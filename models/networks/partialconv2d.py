import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable
from torch import nn
import os
import numpy as np
from PIL import Image
import math
from torch.nn import init


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class InstancePartialConv2d(nn.Module):
    def __init__(self,fin,fout,kw,stride=1,padding=1,return_mask=False):
        self.conv = PartialConv2d(fin,fout,kw,stride,padding,return_mask)

    def forward(self, feat, input_instances):
        n,inst_nc,_,_ = input_instances.size()
        input_inst = F.interpolate(input_instances, size=feat.size()[2:], mode='nearest')
        mask = torch.unsqueeze(input_inst[:,0,:,:],1)
        out = self.conv(feat,mask_in=mask) * mask
        for i in range(1, inst_nc):
            mask = torch.unsqueeze(input_inst[:,i,:,:],1)
            tmp = self.conv(feat,mask_in=mask) * mask
            out += tmp
        return out

class InstanceAwareConv2d(nn.Module):
    def __init__(self, fin, fout, kw, stride=1, padding=1):
        super().__init__()
        self.kw = kw
        self.stride = stride
        self.padding = padding
        self.fin = fin
        self.fout = fout
        self.unfold = nn.Unfold(kw, stride=stride, padding=padding)
        self.weight = nn.Parameter(torch.Tensor(fout, fin, kw, kw))
        self.bias = nn.Parameter(torch.Tensor(fout))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, instances, check=False):
        N,C,H,W = x.size()
        # cal the binary mask from instance map
        instances = F.interpolate(instances, x.size()[2:], mode='nearest') # [n,1,h,w]
        inst_unf = self.unfold(instances)
        # substract the center pixel
        center = torch.unsqueeze(inst_unf[:, self.kw * self.kw // 2, :], 1)
        mask_unf = inst_unf - center
        # clip the absolute value to 0~1
        mask_unf = torch.abs(mask_unf)
        mask_unf = torch.clamp(mask_unf, 0, 1)
        mask_unf = 1.0 - mask_unf # [n,k*k,L]
        # multiply mask_unf and x
        x_unf = self.unfold(x)  # [n,c*k*k,L]
        x_unf = x_unf.view(N, C, -1, x_unf.size()[-1]) # [n,c,,k*k,L]
        mask = torch.unsqueeze(mask_unf,1) # [n,1,k*k,L]
        mask_x = mask * x_unf # [n,c,k*k,L]
        mask_x = mask_x.view(N,-1,mask_x.size()[-1]) # [n,c*k*k,L]
        # conv operation
        weight = self.weight.view(self.fout,-1) # [fout, c*k*k]
        out = torch.einsum('cm,nml->ncl', weight, mask_x)
        # x_unf = torch.unsqueeze(x_unf, 1)  # [n,1,c*k*k,L]
        # out = torch.mul(masked_weight, x_unf).sum(dim=2, keepdim=False) # [n,fout,L]
        bias = torch.unsqueeze(torch.unsqueeze(self.bias,0),-1) # [1,fout,1]
        out = out + bias
        out = out.view(N,self.fout,H//self.stride,W//self.stride)
        # print('weight:',self.weight[0,0,...])
        # print('bias:',self.bias)

        if check:
            out2 = nn.functional.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
            print((out-out2).abs().max())
        return out