from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import os.path
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class CelebADataset(Pix2pixDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=19)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(dataroot='/home/tzt/HairSynthesis/SPADE/datasets/CelebA-HQ/')
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'test' if opt.phase == 'test' else 'train'

        all_images = make_dataset(os.path.join(root,phase), recursive=True, read_cache=False, write_cache=False)
        image_paths = []
        label_paths = []
        instance_paths = []

        for p in all_images:

            if 'images' in p and p.endswith('.jpg'):
                image_paths.append(p)
            elif 'labels' in p and p.endswith('.png'):
                label_paths.append(p)
            elif 'instances' in p and p.endswith('.png'):
                instance_paths.append(p)

        return label_paths, image_paths, instance_paths

def make_inst_data():
    def mkdir_path(path):
        if not os.path.exists(path):
            os.mkdir(path)
    def reid_instance(inst_tensor):
        inst_tensor = inst_tensor.float()
        ori_idx = torch.sort(torch.unique(inst_tensor))[0]
        new_idx = torch.arange(0, ori_idx.size()[0])
        out_inst_tensor = torch.zeros_like(inst_tensor)
        for idx in range(ori_idx.size()[0]):
            tmp = inst_tensor.clone()
            tmp[tmp != ori_idx[idx]] = -1
            tmp[tmp == ori_idx[idx]] = new_idx[idx]
            tmp[tmp == -1] = 0
            out_inst_tensor += tmp
        return out_inst_tensor
    def make_inst_subset(set='test'):
        assert set == 'test' or set == 'train', "set is test or train"
        print('process ', set)
        root = '/home/tzt/HairSynthesis/SPADE/datasets/CelebA-HQ/'
        src_root = os.path.join(root,set,'labels')
        tag_root = os.path.join(root,set,'instances')
        mkdir_path(tag_root)
        names = sorted(os.listdir(src_root))
        for id in tqdm(range(len(names))):
            name = names[id]
            src_mask = np.array(Image.open(os.path.join(src_root,name)))
            tag_label_tensor = torch.from_numpy(src_mask)
            tag_inst = reid_instance(tag_label_tensor).numpy()
            tag_inst = Image.fromarray(np.uint8(tag_inst))
            tag_inst.save(os.path.join(tag_root,name))

    make_inst_subset('test')
    make_inst_subset('train')
