import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.partialconv2d import InstanceAwareConv2d

class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar

class instanceAdaptiveEncoder(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        conv_layer = InstanceAwareConv2d

        self.layer1 = conv_layer(3, ndf, kw, stride=2, padding=pw)
        self.norm1 = nn.InstanceNorm2d(ndf)
        self.layer2 = conv_layer(ndf * 1, ndf * 2, kw, stride=2, padding=pw)
        self.norm2 = nn.InstanceNorm2d(ndf * 2)
        self.layer3 = conv_layer(ndf * 2, ndf * 4, kw, stride=2, padding=pw)
        self.norm3 = nn.InstanceNorm2d(ndf * 4)
        self.layer4 = conv_layer(ndf * 4, ndf * 8, kw, stride=2, padding=pw)
        self.norm4 = nn.InstanceNorm2d(ndf * 8)
        
        self.middle = conv_layer(ndf * 8, ndf * 4, kw, stride=1, padding=pw)
        self.norm_middle = nn.InstanceNorm2d(ndf * 4)
        self.up1 = conv_layer(ndf * 8, ndf * 2, kw, stride=1, padding=pw)
        self.norm_up1 = nn.InstanceNorm2d(ndf * 2)
        self.up2 = conv_layer(ndf * 4, ndf * 1, kw, stride=1, padding=pw)
        self.norm_up2 = nn.InstanceNorm2d(ndf)
        self.up3 = conv_layer(ndf * 2, ndf, kw, stride=1, padding=pw)
        self.norm_up3 = nn.InstanceNorm2d(ndf)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.class_nc = opt.semantic_nc if opt.no_instance else opt.semantic_nc-1

        self.scale_conv_mu = conv_layer(ndf, opt.noise_nc, kw, stride=1, padding=pw)
        self.scale_conv_var = conv_layer(ndf, opt.noise_nc, kw, stride=1, padding=pw)
        self.bias_conv_mu = conv_layer(ndf, opt.noise_nc, kw, stride=1, padding=pw)
        self.bias_conv_var = conv_layer(ndf, opt.noise_nc, kw, stride=1, padding=pw)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def instAvgPooling(self, x, instances):
        inst_num = instances.size()[1]
        for i in range(inst_num):
            inst_mask = torch.unsqueeze(instances[:,i,:,:], 1) # [n,1,h,w]
            pixel_num = torch.sum(torch.sum(inst_mask, dim=2, keepdim=True), dim=3, keepdim=True)
            pixel_num[pixel_num==0] = 1
            feat = x * inst_mask
            feat = torch.sum(torch.sum(feat, dim=2, keepdim=True), dim=3, keepdim=True) / pixel_num
            if i == 0:
                out = torch.unsqueeze(feat[:,:,0,0],1) # [n,1,c]
            else:
                out = torch.cat([out,torch.unsqueeze(feat[:,:,0,0],1)],1)
            # inst_pool_feats.append(feat[:,:,0,0]) # [n, 64]
        return out

    def forward(self, x, input_instances):
        # instances [n,1,h,w], input_instances [n,inst_nc,h,w]
        instances = torch.argmax(input_instances, 1, keepdim=True).float()
        x1 = self.actvn(self.norm1(self.layer1(x,instances)))
        x2 = self.actvn(self.norm2(self.layer2(x1,instances)))
        x3 = self.actvn(self.norm3(self.layer3(x2,instances)))
        x4 = self.actvn(self.norm4(self.layer4(x3,instances)))
        y = self.up(self.actvn(self.norm_middle(self.middle(x4,instances))))
        y1 = self.up(self.actvn(self.norm_up1(self.up1(torch.cat([y,x3],1),instances))))
        y2 = self.up(self.actvn(self.norm_up2(self.up2(torch.cat([y1, x2], 1),instances))))
        y3 = self.up(self.actvn(self.norm_up3(self.up3(torch.cat([y2, x1], 1),instances))))

        scale_mu = self.scale_conv_mu(y3,instances)
        scale_var = self.scale_conv_var(y3,instances)
        bias_mu = self.bias_conv_mu(y3,instances)
        bias_var = self.bias_conv_var(y3,instances)

        scale_mus = self.instAvgPooling(scale_mu,input_instances)
        scale_vars = self.instAvgPooling(scale_var,input_instances)
        bias_mus = self.instAvgPooling(bias_mu,input_instances)
        bias_vars = self.instAvgPooling(bias_var,input_instances)

        return scale_mus, scale_vars, bias_mus, bias_vars
