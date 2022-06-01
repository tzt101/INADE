import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

class DeepfashionDataset(Pix2pixDataset):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=8)
        parser.set_defaults(noise_nc=8)
        parser.set_defaults(dataroot='/home/tzt/HairSynthesis/SPADE/datasets/DeepFashion/')
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'test' if opt.phase == 'test' else opt.phase

        label_dir = os.path.join(root, '%s_label' % phase)
        label_paths = make_dataset(label_dir, recursive=False)

        image_dir = os.path.join(root, '%s_img' % phase)
        image_paths = make_dataset(image_dir, recursive=False)

        if not opt.no_instance:
            instance_dir = os.path.join(root, '%s_inst' % phase)
            instance_paths = make_dataset(instance_dir, recursive=False)
        else:
            instance_paths = []

        if opt.add_sketch:
            sketch_dir = os.path.join(root, '%s_edgeD' % phase)
            sketch_paths = make_dataset(sketch_dir, recursive=False)
        else:
            sketch_paths = []

        return label_paths, image_paths, instance_paths, sketch_paths