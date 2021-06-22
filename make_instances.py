import os.path
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', type=str, help='Path to datasets')
parser.add_argument('--dataset', type=str, default='ade20k', help='which dataset to process')

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
# process for ade20k dataset
def make_inst_for_ade20k(path):
    def make_inst_subset(path, set='validation'):
        assert set == 'validation' or set == 'training', "set is validation or training"
        print('process ', set)
        root = os.path.join(path, 'annotations')
        src_root = os.path.join(root, set)
        tag_root = os.path.join(path, 'instances')
        mkdir_path(tag_root)
        tag_ins_root = os.path.join(tag_root,set)
        mkdir_path(tag_ins_root)
        names = sorted(os.listdir(src_root))
        for id in tqdm(range(len(names))):
            name = names[id]
            src_mask = np.array(Image.open(os.path.join(src_root,name)))
            tag_label_tensor = torch.from_numpy(src_mask)
            tag_inst = reid_instance(tag_label_tensor).numpy()
            tag_inst = Image.fromarray(np.uint8(tag_inst))
            tag_inst.save(os.path.join(tag_ins_root,name))
    make_inst_subset(path, 'validation')
    make_inst_subset(path, 'training')

# process for celeba-mask dataset
def make_inst_for_celeba(path):
    def make_inst_subset(path, set='test'):
        assert set == 'test' or set == 'train', "set is test or train"
        print('process ', set)
        src_root = os.path.join(path,set,'labels')
        tag_root = os.path.join(path,set,'instances')
        mkdir_path(tag_root)
        names = sorted(os.listdir(src_root))
        for id in tqdm(range(len(names))):
            name = names[id]
            src_mask = np.array(Image.open(os.path.join(src_root,name)))
            tag_label_tensor = torch.from_numpy(src_mask)
            tag_inst = reid_instance(tag_label_tensor).numpy()
            tag_inst = Image.fromarray(np.uint8(tag_inst))
            tag_inst.save(os.path.join(tag_root,name))
    make_inst_subset(path, 'test')
    make_inst_subset(path, 'train')

# process for deepfashion dataset
def make_inst_for_deepfashion(path):
    def make_inst_subset(path, set='test'):
        assert set == 'test' or set == 'train', "set is test or train"
        print('process ', set)
        src_root = os.path.join(path,set+'_mask')
        tag_lab_root = os.path.join(path,set+'_label')
        tag_ins_root = os.path.join(path,set+'_inst')
        mkdir_path(tag_lab_root)
        mkdir_path(tag_ins_root)
        names = sorted(os.listdir(src_root))
        for id in tqdm(range(len(names))):
            name = names[id]
            src_mask = np.array(Image.open(os.path.join(src_root,name)))
            tag_label = src_mask[:,:,0]
            tag_label_tensor = torch.from_numpy(tag_label)
            tag_inst = reid_instance(tag_label_tensor).numpy()
            tag_label = Image.fromarray(np.uint8(tag_label))
            tag_inst = Image.fromarray(np.uint8(tag_inst))
            tag_label.save(os.path.join(tag_lab_root,name))
            tag_inst.save(os.path.join(tag_ins_root,name))
    make_inst_subset(path, 'test')
    make_inst_subset(path, 'train')

# process for cityscapes dataset
def reid_instance(inst_tensor):
    inst_tensor = inst_tensor.float()
    ori_idx = torch.unique(inst_tensor)
    new_idx = torch.arange(0,ori_idx.size()[0])
    out_inst_tensor = torch.zeros_like(inst_tensor)
    for idx in range(ori_idx.size()[0]):
        tmp = inst_tensor.clone()
        tmp[tmp!=ori_idx[idx]] = -1
        tmp[tmp==ori_idx[idx]] = new_idx[idx]
        tmp[tmp==-1] = 0
        out_inst_tensor += tmp
    return out_inst_tensor

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def reid_cityscapes_dataset(dir='/home/tzt/dataset/cityscapes/'):
    label_dir = os.path.join(dir, 'gtFine')
    phases = ['val', 'train']
    count = 0
    for phase in phases:
        if 'test' in phase:
            continue
        print('process', phase, 'dataset')
        citys = sorted(os.listdir(os.path.join(label_dir, phase)))
        for city in citys:
            label_path = os.path.join(label_dir, phase, city)
            label_names_all = sorted(os.listdir(label_path))
            instance_names = [p for p in label_names_all if p.endswith('_instanceIds.png')]
            for instance_name in instance_names:
                # print(instance_name)
                inst_np = np.array(Image.open(os.path.join(label_path, instance_name)))
                inst_tensor = torch.from_numpy(inst_np)
                reid_inst_tensor = reid_instance(inst_tensor)
                reid_inst_np = reid_inst_tensor.numpy()
                # print('-------------------')
                # print(np.unique(inst_np))
                # print(np.unique(reid_inst_np))
                save_name = instance_name[:-7]+'ReIds.png'
                reid_inst_save = Image.fromarray(np.uint8(reid_inst_np))
                reid_inst_save.save(os.path.join(label_path,save_name))
                id_num = len(np.unique(reid_inst_np))
                count += (1 if id_num > 255 else 0)
                print('process',instance_name,'with',id_num,'numbers')
    print('Finished! The number of map which more than 255 id is',count)

if __name__ == '__main__':
    print('Start ...')

    args = parser.parse_args()
    if args.dataset == 'ade20k':
        make_inst_for_ade20k(args.path)
    elif args.dataset == 'cityscapes':
        reid_cityscapes_dataset(args.path)
    elif args.dataset == 'deepfashion':
        make_inst_for_deepfashion(args.path)
    elif args.dataset == 'celeba':
        make_inst_for_celeba(args.path)
    else:
        print('Error! dataset must be one of [ade20k|cityscapes|deepfashion|celeba]')