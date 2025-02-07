"""
SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""

import math
import json
import os
import numpy as np
import torch
from PIL import Image

from torch.utils import data
from datasets import VOCSegmentation
from utils import ext_transforms as et
from utils.tasks import get_tasks
import torch.nn.functional as F
def memory_sampling_balanced(opts, prev_model):
    
    fg_idx = 1 if opts.unknown else 0
    
    transform = et.ExtCompose([
        et.ExtResize(opts.crop_size),
        et.ExtCenterCrop(opts.crop_size),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    
    if opts.dataset == 'voc':
        dataset = VOCSegmentation
    # elif opts.dataset == 'ade':
    #     dataset = ADESegmentation
    else:
        raise NotImplementedError
        
    train_dst = dataset(opts=opts, image_set='train', transform=transform, cil_step=opts.curr_step-1)
    
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, 
        shuffle=True, num_workers=4, drop_last=False)
    
    num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1)]
    prev_num_classes = sum(num_classes[:-1])  # 16
    memory_json = f'./datasets/data/{opts.dataset}/memory.json'
    # input(num_classes)
    if opts.curr_step > 1:
        with open(memory_json, "r") as json_file:
            memory_list = json.load(json_file)

        memory_candidates = memory_list[f"step_{opts.curr_step-1}"]["memory_candidates"]

        before_class = sum(num_classes[:-2])-1
        after_class = num_classes[-2]
        num_memory = (opts.mem_size * after_class) // before_class
    else:
        memory_list = {}
        memory_candidates = []
        num_memory = 50000


    
    print("...start memory candidates collection")

    curr_num_memory = 0

    for images, targets, img_names, _ in train_loader:
        if curr_num_memory >=num_memory:
            break

        for b in range(images.size(0)):
            if curr_num_memory >=num_memory:
                break
            img_name = img_names[b]
            target = targets[b]

            labels = torch.unique(target).detach().cpu().numpy()
            labels = (labels - 1).tolist() if opts.unknown else labels.tolist()
            
            if -1 in labels:
                labels.remove(-1)
                
            if 0 in labels:
                labels.remove(0)
            
            objs_num = len(labels)
            objs_ratio = int((target > fg_idx).sum())
            
            memory_candidates.append([img_name, objs_num, objs_ratio, labels])
            curr_num_memory+=1

    print("...end memory candidates collection : ", len(memory_candidates))
    
    ####################################################################################################
    
    print("...start memory list generation")
    curr_memory_list = {f"class_{cls}":[] for cls in range(1, prev_num_classes)} # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    np.random.shuffle(sorted_memory_candidates)

    random_class_order = list(range(1, prev_num_classes))
    np.random.shuffle(random_class_order)
    num_sampled = 0
    sum_class = sum(num_classes[:-1])-1
    num_per_class = np.zeros(sum_class+1)
    avg_num = opts.mem_size //sum_class

    flag = np.ones(len(sorted_memory_candidates))
    for idx, mem in enumerate(sorted_memory_candidates):
        img_name, objs_num, objs_ratio, labels = mem
        for cls in random_class_order:
            if cls in labels and num_per_class[cls]<avg_num:
                curr_memory_list[f"class_{cls}"].append(mem)
                num_sampled += 1
                num_per_class[cls]+=1
                flag[idx]=0
                break
        if opts.mem_size <= num_sampled:
            break

    while opts.mem_size > num_sampled:
        for idx, mem in enumerate(sorted_memory_candidates):
            if flag[idx]==1:
                img_name, objs_num, objs_ratio, labels = mem
                for cls in random_class_order:
                    if cls in labels:
                        curr_memory_list[f"class_{cls}"].append(mem)
                        num_sampled += 1
                        num_per_class[cls]+=1
                        break
            if opts.mem_size <= num_sampled:
                break
    ###################################### 
    
    """ save memory info """
    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory
    
    memory_list[f"step_{opts.curr_step}"] = {"memory_candidates": sampled_memory_list, 
                                                  "memory_list": sorted([mem[0] for mem in sampled_memory_list])
                                                 }    
    
    with open(memory_json, "w") as json_file:
        json.dump(memory_list, json_file)
        
