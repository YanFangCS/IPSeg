"""
modified by SSUL
"""

import os
import sys
import torch.utils.data as data
import numpy as np
import json

import torch
from PIL import Image

from utils.tasks import get_dataset_list, get_tasks
def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(0, N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap[254] = np.array([0, 0, 0])       # bg
    cmap[255] = np.array([0, 0, 0])       # bg
    cmap[200] = np.array([192, 192, 192])       # unknown
    cmap = cmap/255 if normalized else cmap
    return cmap

class VOCSegmentation(data.Dataset):
    cmap = voc_cmap()
    
    def __init__(self,
                 opts,
                 image_set='train',
                 transform=None,
                 cil_step=0,
                 mem_size=0):

        self.root=opts.data_root        
        self.task=opts.task
        self.overlap=opts.overlap
        self.unknown=opts.unknown
        self.curr_step=opts.curr_step
        self.num_classes=opts.num_classes
        self.image_set = image_set
        self.transform = transform


        self.curr_cls = sum(self.num_classes[:-1])-2



        voc_root = './datasets/data/voc'
        image_dir = os.path.join(self.root, 'JPEGImages')

        if not os.path.isdir(self.root):
            raise RuntimeError('Dataset not found or corrupted.')
        
        mask_dir = os.path.join(self.root, 'SegmentationClassAug')
        salmap_dir = os.path.join(self.root, 'saliency_map')
        assert os.path.exists(mask_dir), "SegmentationClassAug not found, please refer to README.md and prepare it manually"
            
        self.target_cls = get_tasks('voc', self.task, cil_step)
        self.target_cls += [255] # including ignore index (255)
        
        if image_set=='test':
            file_names = open(os.path.join(self.root, 'ImageSets/Segmentation', 'val.txt'), 'r')
            file_names = file_names.read().splitlines()
            
        elif image_set == 'memory':
            memory_json = os.path.join(voc_root, 'memory.json')

            with open(memory_json, "r") as json_file:
                memory_list = json.load(json_file)

            file_names = memory_list[f"step_{cil_step}"]["memory_list"]
            print("... memory list : ", len(file_names), self.target_cls)
            
            while len(file_names) < opts.batch_size:
                file_names = file_names * 2

        else:
            file_names = get_dataset_list('voc', self.task, cil_step, image_set, self.overlap)

        all_c = open(os.path.join('./datasets/data/voc/', 'all_cls.txt'), 'r')

        all_c = all_c.read().splitlines()
        self.all_c=[]
        for name in file_names:
            temp_cls=torch.zeros(20)
            for line in all_c:
                parts = line.split()
                if parts[0] == name:
                    numbers = [int(num) for num in parts[1:]]
                    temp_cls[numbers] = 1
                    self.all_c.append(temp_cls)
                    break  # Stop searching once the filename is found

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        self.sal_maps = [os.path.join(salmap_dir, x + ".png") for x in file_names]
        self.file_names = file_names
        
        # class re-ordering
        all_steps = get_tasks('voc', self.task)
        all_classes = []
        for i in range(len(all_steps)):
            all_classes += all_steps[i]
            
        self.ordering_map = np.zeros(256, dtype=np.uint8) + 255
        self.ordering_map[:len(all_classes)] = [all_classes.index(x) for x in range(len(all_classes))]

        assert (len(self.images) == len(self.masks))

        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        file_name = self.file_names[index]
        cls=self.all_c[index]
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.image_set == 'train' or self.image_set == 'memory': 
            sal_map = Image.open(self.sal_maps[index])
        else:
            # sal_map is useless for the valdation
            sal_map = Image.fromarray(np.ones(target.size[::-1], dtype=np.uint8))
        
        # re-define target label according to the CIL case
        target = self.gt_label_mapping(target)
        
        if self.transform is not None:
            img, target, sal_map = self.transform(img, target, sal_map)
        
        # add unknown label, background index: 0 -> 1, unknown index: 0
        if (self.image_set == 'train' or self.image_set == 'memory') and self.unknown:
            target = torch.where(target == 255, 
                                 torch.zeros_like(target) + 255,  # keep 255 (uint8)
                                 target+1) # unknown label

            unknown_area = (target == 1) & (sal_map > 0)
            target = torch.where(unknown_area, torch.zeros_like(target), target)


        # IPSeg avoids knowledge leakage from the old class
        if self.image_set == 'train':
            cls[:self.curr_cls] = torch.zeros(self.curr_cls)
        return img, target.long(), file_name, cls


    def __len__(self):
        return len(self.images)

    def gt_label_mapping(self, gt):
        gt = np.array(gt, dtype=np.uint8)
        if self.image_set != 'test':
            gt = np.where(np.isin(gt, self.target_cls), gt, 0)
        gt = self.ordering_map[gt]
        gt = Image.fromarray(gt)
        
        return gt
    
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

