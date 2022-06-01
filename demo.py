import os
import sys
import cv2
import time
import numpy as np
from PIL import Image
import glob
import torch
from torchvision.utils import save_image
import torch.nn.functional as F

from ui.deepfashion.demo import demo_for_deepfashion
from options.demo_options import DemoOptions

if __name__ == '__main__':
    print('The demo code')

    opt = DemoOptions().parse()

    if 'inades' in opt.name:
        opt.add_sketch = True

    if opt.dataset_mode == 'deepfashion':
        demo_for_deepfashion(opt)
    else:
        raise ValueError('%s is not a recognized dataset_mode for demo' % opt.dataset_mode)
