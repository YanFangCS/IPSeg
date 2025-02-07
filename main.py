"""
SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""
 
from tqdm import tqdm
import network
import utils
import os
import time
import random
import argparse
import numpy as np
#import cv2

from torch.utils import data
from datasets import VOCSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.utils import AverageMeter
from utils.tasks import get_tasks
from utils.memory import memory_sampling_balanced

torch.backends.cudnn.benchmark = True

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/data/DB/VOC2012',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc', choices=['voc', 'ade'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3_resnet101',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet',
                                 'deeplabv3_swin_transformer'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--amp", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--train_epoch", type=int, default=50,
                        help="epoch number")
    parser.add_argument("--curr_itrs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='warm_poly', choices=['poly', 'step', 'warm_poly'],
                        help="learning rate scheduler")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=32,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")

    parser.add_argument("--loss_type", type=str, default='bce_loss',
                        choices=['ce_loss', 'focal_loss', 'bce_loss'], help="loss type")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    
    # CIL options
    parser.add_argument("--pseudo", action='store_true', help="enable pseudo-labeling")
    parser.add_argument("--pseudo_thresh", type=float, default=0.7, help="confidence threshold for pseudo-labeling")
    parser.add_argument("--task", type=str, default='15-1', help="cil task")
    parser.add_argument("--curr_step", type=int, default=0)
    parser.add_argument("--overlap", action='store_true', help="overlap setup (True), disjoint setup (False)")
    parser.add_argument("--mem_size", type=int, default=0, help="size of examplar memory")
    parser.add_argument("--freeze", action='store_true', help="enable network freezing")
    parser.add_argument("--bn_freeze", action='store_true', help="enable batchnorm freezing")
    parser.add_argument("--w_transfer", action='store_true', help="enable weight transfer")
    parser.add_argument("--unknown", action='store_true', help="enable unknown modeling")
    
    return parser

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    
    train_transform = et.ExtCompose([
        #et.ExtResize(size=opts.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        
    if opts.dataset == 'voc':
        dataset = VOCSegmentation
    else:
        raise NotImplementedError
        
    dataset_dict = {}
    dataset_dict['train'] = dataset(opts=opts, image_set='train', transform=train_transform, cil_step=opts.curr_step)
    
    dataset_dict['val'] = dataset(opts=opts,image_set='val', transform=val_transform, cil_step=opts.curr_step)
    
    dataset_dict['test'] = dataset(opts=opts, image_set='test', transform=val_transform, cil_step=opts.curr_step)
    
    if opts.curr_step > 0 and opts.mem_size > 0:
        dataset_dict['memory'] = dataset(opts=opts, image_set='memory', transform=train_transform, 
                                                 cil_step=opts.curr_step, mem_size=opts.mem_size)

    return dataset_dict

import torch.nn.functional as F

def validate(s, opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()

    with torch.no_grad():
        for i, (images, labels, _n, corase_labels) in enumerate(loader):
            
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            
            global_outputs, curr_output, m_outputs = model(images)
  
            input_shape = images.shape[-2:]
            outputs = F.interpolate(global_outputs, size=input_shape, mode='bilinear', align_corners=False)
            if opts.loss_type == 'bce_loss':
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)
                    
            # remove unknown label
            if opts.unknown:
                outputs[:, 1] += outputs[:, 0]
                outputs = outputs[:, 1:]
            
            BS = 0.9 # primary back_scale ratio
            B,C=m_outputs.shape
            m_outputs = torch.sigmoid(m_outputs)
            back_scale = torch.full((B,1), BS,dtype=m_outputs.dtype,device=m_outputs.device)
                
            m_outputs=torch.cat((back_scale, m_outputs),dim=1).unsqueeze(-1).unsqueeze(-1) # B,C+1
            outputs = outputs * m_outputs
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
                
        score = metrics.get_results()
    return score

def main(opts):
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    bn_freeze = opts.bn_freeze if opts.curr_step > 0 else False
        
    target_cls = get_tasks(opts.dataset, opts.task, opts.curr_step)
    opts.num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1)]
    if opts.unknown: # re-labeling: [unknown, background, ...]
        opts.num_classes = [1, 1, opts.num_classes[0]-1] + opts.num_classes[1:]
    fg_idx = 1 if opts.unknown else 0
    
    curr_idx = [
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step)), 
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1))
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("==============================================")
    print(f"  task : {opts.task}")
    print(f"  step : {opts.curr_step}")
    print("  Device: %s" % device)
    print( "  opts : ")
    print(opts)
    print("==============================================")

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    
    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
        'deeplabv3_swin_transformer': network.deeplabv3_swin_transformer
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, bn_freeze=bn_freeze)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
        
    if opts.curr_step > 0:
        """ load previous model """
        model_prev = model_map[opts.model](num_classes=opts.num_classes[:-1], output_stride=opts.output_stride, bn_freeze=bn_freeze)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model_prev.classifier)
        utils.set_bn_momentum(model_prev.backbone, momentum=0.01)
    else:
        model_prev = None
    
    # Set up metrics
    metrics = StreamSegMetrics(sum(opts.num_classes)-1 if opts.unknown else sum(opts.num_classes), dataset=opts.dataset)

    # print(model.classifier.head)
    
    # Set up optimizer & parameters
    if opts.freeze and opts.curr_step > 0:
        for param in model_prev.parameters():
            param.requires_grad = False

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.mining_head1[-1].parameters(): # classifier for new class
            param.requires_grad = True
        training_params = [{'params': model.classifier.mining_head1[-1].parameters(), 'lr': opts.lr}]
        for param in model.classifier.mining_head2[-1].parameters():
                param.requires_grad = True
        training_params.append({'params': model.classifier.mining_head2[-1].parameters(), 'lr': opts.lr})
        for param in model.classifier.mining_head3[-1].parameters():
                param.requires_grad = True
        training_params.append({'params': model.classifier.mining_head3[-1].parameters(), 'lr': opts.lr})

        
        for param in model.classifier.fc.parameters(): # IP branch
            param.requires_grad = True
        training_params.append({'params': model.classifier.fc.parameters(), 'lr': opts.lr})
        for param in model.classifier.scale_head.parameters(): # IP branch
            param.requires_grad = True
        training_params.append({'params': model.classifier.scale_head.parameters(), 'lr': opts.lr})

        if opts.unknown:
            for param in model.classifier.mining_head1[0].parameters(): # unknown
                param.requires_grad = True
            training_params.append({'params': model.classifier.mining_head1[0].parameters(), 'lr': opts.lr * 0.1})
            for param in model.classifier.mining_head2[0].parameters(): # unknown
                param.requires_grad = True
            training_params.append({'params': model.classifier.mining_head2[0].parameters(), 'lr': opts.lr * 0.1})
            for param in model.classifier.mining_head3[0].parameters(): # unknown
                param.requires_grad = True
            training_params.append({'params': model.classifier.mining_head3[0].parameters(), 'lr': opts.lr * 0.1})
            

        
    else:
        training_params = [{'params': model.backbone.parameters(), 'lr': 0.001},
                           {'params': model.classifier.parameters(), 'lr': 0.01}]
        
    optimizer = torch.optim.SGD(params=training_params, 
                                lr=opts.lr, 
                                momentum=0.9, 
                                weight_decay=opts.weight_decay,
                                nesterov=True)
    
    print("----------- trainable parameters --------------")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    print("-----------------------------------------------")
    
    def save_ckpt(path):
        torch.save({
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    utils.mkdir('checkpoints')
    # Restore
    best_score = -1
    cur_itrs = 0
    cur_epochs = 0
    
    if opts.overlap:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_overlap.pth"
    else:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_disjoint.pth"
    
    if opts.curr_step > 0: # previous step checkpoint
        opts.ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step-1)
        
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))["model_state"]
        model_prev.load_state_dict(checkpoint, strict=True)
        
        if opts.unknown and opts.w_transfer:
            # weight transfer : from unknown to new-class
            print("... weight transfer")
            curr_head_num = len(model.classifier.mining_head1) - 1

            checkpoint[f"classifier.mining_head1.{curr_head_num}.0.weight"] = checkpoint["classifier.mining_head1.0.0.weight"]
            checkpoint[f"classifier.mining_head1.{curr_head_num}.0.bias"] = checkpoint["classifier.mining_head1.0.0.bias"]
            checkpoint[f"classifier.mining_head1.{curr_head_num}.1.weight"] = checkpoint["classifier.mining_head1.0.1.weight"]
            checkpoint[f"classifier.mining_head1.{curr_head_num}.1.bias"] = checkpoint["classifier.mining_head1.0.1.bias"]
            checkpoint[f"classifier.mining_head1.{curr_head_num}.1.running_mean"] = checkpoint["classifier.mining_head1.0.1.running_mean"]
            checkpoint[f"classifier.mining_head1.{curr_head_num}.1.running_var"] = checkpoint["classifier.mining_head1.0.1.running_var"]

            checkpoint[f"classifier.mining_head2.{curr_head_num}.0.weight"] = checkpoint["classifier.mining_head2.0.0.weight"]
            checkpoint[f"classifier.mining_head2.{curr_head_num}.0.bias"] = checkpoint["classifier.mining_head2.0.0.bias"]
            checkpoint[f"classifier.mining_head2.{curr_head_num}.1.weight"] = checkpoint["classifier.mining_head2.0.1.weight"]
            checkpoint[f"classifier.mining_head2.{curr_head_num}.1.bias"] = checkpoint["classifier.mining_head2.0.1.bias"]
            checkpoint[f"classifier.mining_head2.{curr_head_num}.1.running_mean"] = checkpoint["classifier.mining_head2.0.1.running_mean"]
            checkpoint[f"classifier.mining_head2.{curr_head_num}.1.running_var"] = checkpoint["classifier.mining_head2.0.1.running_var"]

            checkpoint[f"classifier.mining_head3.{curr_head_num}.0.weight"] = checkpoint["classifier.mining_head3.0.0.weight"]
            checkpoint[f"classifier.mining_head3.{curr_head_num}.0.bias"] = checkpoint["classifier.mining_head3.0.0.bias"]
            checkpoint[f"classifier.mining_head3.{curr_head_num}.1.weight"] = checkpoint["classifier.mining_head3.0.1.weight"]
            checkpoint[f"classifier.mining_head3.{curr_head_num}.1.bias"] = checkpoint["classifier.mining_head3.0.1.bias"]
            checkpoint[f"classifier.mining_head3.{curr_head_num}.1.running_mean"] = checkpoint["classifier.mining_head3.0.1.running_mean"]
            checkpoint[f"classifier.mining_head3.{curr_head_num}.1.running_var"] = checkpoint["classifier.mining_head3.0.1.running_var"]


            last_conv_weight = model.state_dict()[f"classifier.mining_head3.{curr_head_num}.3.weight"]
            last_conv_bias = model.state_dict()[f"classifier.mining_head3.{curr_head_num}.3.bias"]

            last_conv_weight[:2] = checkpoint["classifier.mining_head3.0.3.weight"]
            last_conv_bias[:2] = checkpoint["classifier.mining_head3.0.3.bias"]
            for i in range(2, opts.num_classes[-1]):
                last_conv_weight[i] = last_conv_weight[0]
                last_conv_bias[i] = last_conv_bias[0]

            checkpoint[f"classifier.mining_head3.{curr_head_num}.3.weight"] = last_conv_weight
            checkpoint[f"classifier.mining_head3.{curr_head_num}.3.bias"] = last_conv_bias

        model.load_state_dict(checkpoint, strict=False)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")

    model = nn.DataParallel(model)
    mode = model.to(device)
    mode.train()
    
    if opts.curr_step > 0:
        model_prev = nn.DataParallel(model_prev)
        model_prev = model_prev.to(device)
        model_prev.eval()

        if opts.mem_size > 0:
            memory_sampling_balanced(opts, model_prev)
            
        # Setup dataloader
    if not opts.crop_val:
        opts.val_batch_size = 1
    
    dataset_dict = get_dataset(opts)
    train_loader = data.DataLoader(
        dataset_dict['train'], batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = data.DataLoader(
        dataset_dict['val'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(
        dataset_dict['test'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
          (opts.dataset, len(dataset_dict['train']), len(dataset_dict['val']), len(dataset_dict['test'])))
    
    if opts.curr_step > 0 and opts.mem_size > 0:
        memory_loader = data.DataLoader(
            dataset_dict['memory'], batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    total_itrs = opts.train_epoch * len(train_loader)
    val_interval = max(100, total_itrs // (opts.train_epoch//2))
    print(f"... train epoch : {opts.train_epoch} , iterations : {total_itrs} , val_interval : {val_interval}")
        
    #==========   Train Loop   ==========#
    if opts.test_only:
        model.eval()
        test_score = validate(opts=opts, model=model, loader=test_loader, 
                              device=device, metrics=metrics)
        
        print(metrics.to_str(test_score))
        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())

        first_cls = len(get_tasks(opts.dataset, opts.task, 0)) # 15-1 task -> first_cls=16
        print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
        print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))
        return

    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    elif opts.lr_policy=='warm_poly':
        warmup_iters = int(total_itrs*0.1)
        scheduler = utils.WarmupPolyLR(optimizer, total_itrs, warmup_iters=warmup_iters, power=0.9)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'ce_loss':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'bce_loss':
        criterion = utils.BCEWithLogitsLossWithIgnoreIndex(ignore_index=255, 
                                                           reduction='mean')
        
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
    
    avg_loss_sum = AverageMeter()
    avg_loss_global = AverageMeter()
    avg_loss_curr = AverageMeter()
    avg_loss_scale = AverageMeter()
    avg_time = AverageMeter()
    
    model.train()
    save_ckpt(ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step))
    
    # =====  Train  =====
    while cur_itrs < total_itrs:
        cur_itrs += 1
        optimizer.zero_grad()
        end_time = time.time()
        
        """ data load """
        try:
            images, labels, _ , corase_labels = train_iter.next()
        except:
            train_iter = iter(train_loader)
            images, labels, _ , corase_labels = train_iter.next()
            cur_epochs += 1
            avg_loss_sum.reset()
            avg_loss_global.reset()
            avg_loss_curr.reset()
            avg_loss_scale.reset()
            avg_time.reset()
        
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        corase_labels = corase_labels.to(device, dtype=torch.float, non_blocking=True)

        """ memory """
        if opts.curr_step > 0 and opts.mem_size > 0:
            try:
                m_images, m_labels, _ , m_corase_labels = mem_iter.next()
            except:
                mem_iter = iter(memory_loader)
                m_images, m_labels, _ , m_corase_labels = mem_iter.next()

            m_images = m_images.to(device, dtype=torch.float32, non_blocking=True)
            m_labels = m_labels.to(device, dtype=torch.long, non_blocking=True)
            m_corase_labels = m_corase_labels.to(device, dtype=torch.float, non_blocking=True)
            

            ori_m_images = m_images.clone()
            ori_m_corase_labels = m_corase_labels.clone()
            num_classes = opts.num_classes
            rand_index = torch.randperm(opts.batch_size)[:max(1,opts.batch_size * num_classes[-1] // (sum(num_classes)-2))].cuda()
            ori_m_images[rand_index, ...] = images[rand_index, ...]
            ori_m_corase_labels[rand_index, ...] = corase_labels[rand_index, ...]

            rand_index = torch.randperm(opts.batch_size)[:opts.batch_size // 2].cuda()
            images[rand_index, ...] = m_images[rand_index, ...]
            labels[rand_index, ...] = m_labels[rand_index, ...]

            del m_images, m_labels, m_corase_labels

        
        """ forwarding and optimization """
        with torch.cuda.amp.autocast(enabled=opts.amp):

            global_outputs, curr_outputs, corase_outputs = model(images)
            input_shape = images.shape[-2:]
            global_outputs = F.interpolate(global_outputs, size=input_shape, mode='bilinear', align_corners=False)
            curr_outputs = F.interpolate(curr_outputs, size=input_shape, mode='bilinear', align_corners=False)
            if opts.pseudo and opts.curr_step > 0:
                """ pseudo labeling """
                with torch.no_grad():
                    global_output_p, _, _ = model_prev(images)
                    outputs_prev = F.interpolate(global_output_p, size=input_shape, mode='bilinear', align_corners=False)
                if opts.loss_type == 'bce_loss':
                    pred_prob = torch.sigmoid(outputs_prev).detach()
                else:
                    pred_prob = torch.softmax(outputs_prev, 1).detach()
                    
                pred_scores, pred_labels = torch.max(pred_prob, dim=1)
                # pseudo labels for main branch
                pseudo_labels = torch.where( (labels <= fg_idx) & (pred_labels > fg_idx) & (pred_scores >= opts.pseudo_thresh), 
                                            pred_labels, 
                                            labels)
                
                # pseudo labels for IP branch
                p_labels=pseudo_labels.clone()
                p_labels[p_labels == 255] = 1
                num_classes = 22
                class_counts = torch.zeros((16, num_classes), dtype=torch.int64).cuda()
                for i in range(16):
                    unique_labels, counts = p_labels[i].unique(return_counts=True)
                    class_counts[i, unique_labels] = counts
                image_level_labels = class_counts > 1000



                loss_global = criterion(global_outputs, pseudo_labels) * 0.5
                loss_curr = criterion(curr_outputs, labels) * 0.5
            else:
                loss_global = criterion(global_outputs, labels) * 0.5
                loss_curr = criterion(curr_outputs, labels) * 0.5

            if opts.curr_step > 0 and opts.mem_size > 0:
                a = 1
                _, _, ori_m_outputs = model(ori_m_images)
                B,C=ori_m_outputs.shape
                targets = ori_m_corase_labels[:,:C]


                image_level_labels=image_level_labels[:,2:]
                for i in range(B):
                    for j in range(C-1):
                        if image_level_labels[i,j]==True and targets[i,C-1]==1:
                            targets[i,j]=1

                loss_scale = nn.BCEWithLogitsLoss()(ori_m_outputs, targets) * a
                loss_sum = loss_global + loss_curr + loss_scale

            elif opts.curr_step == 0 and opts.mem_size > 0:
                a = 1
                B,C=corase_outputs.shape
                targets = corase_labels[:,:C]
                loss_scale = nn.BCEWithLogitsLoss()(corase_outputs, targets) * a
                loss_sum = loss_global + loss_curr + loss_scale

        scaler.scale(loss_sum).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        avg_loss_sum.update(loss_sum.item())
        avg_loss_global.update(loss_global.item())
        avg_loss_curr.update(loss_curr.item())
        avg_loss_scale.update(loss_scale.item())
        avg_time.update(time.time() - end_time)
        end_time = time.time()

        if (cur_itrs) % 10 == 0:
            print("[%s / step %d] Epoch %d, Itrs %d/%d, Loss=%6f, Loss_global=%6f, Loss_curr=%6f, Loss_scale=%6f, Time=%.2f , LR=%.8f, %.2f hours left" %
                  (opts.task, opts.curr_step, cur_epochs, cur_itrs, total_itrs, 
                   avg_loss_sum.avg, avg_loss_global.avg, avg_loss_curr.avg, avg_loss_scale.avg, avg_time.avg*1000, optimizer.param_groups[0]['lr'], (total_itrs - cur_itrs) * avg_time.avg / 3600))

        if val_interval > 0 and (cur_itrs) % val_interval == 0:
            print("validation...")
            model.eval()
            val_score = validate('val', opts=opts, model=model, loader=val_loader, 
                                 device=device, metrics=metrics)
            print(metrics.to_str(val_score))
            
            model.train()
            
            class_iou = list(val_score['Class IoU'].values())
            val_score = np.mean( class_iou[curr_idx[0]:curr_idx[1]] + [class_iou[0]])
            curr_score = np.mean( class_iou[curr_idx[0]:curr_idx[1]] )
            print("curr_val_score : %.4f" % (curr_score))
            print()
            
            if curr_score > best_score:  # save best model
                print("... save best ckpt : ", curr_score)
                best_score = curr_score
                save_ckpt(ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step))


    print("... Training Done")
    
    if opts.curr_step > 0:

        print("... Testing Best Model interpolate sigmoid")
        best_ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step)
        
        checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        
        test_score = validate('test', opts=opts, model=model, loader=test_loader, 
                              device=device, metrics=metrics)
        print(metrics.to_str(test_score))

        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())
        first_cls = len(get_tasks(opts.dataset, opts.task, 0))

        print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
        print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))


import datetime
if __name__ == '__main__':
            
    opts = get_argparser().parse_args()
        
    start_step = 0
    total_step = len(get_tasks(opts.dataset, opts.task))

    start_time = datetime.datetime.now()
    print("start:", start_time)

    for step in range(start_step, total_step):
        opts.curr_step = step
        t1=datetime.datetime.now()
        main(opts)
        t2 = datetime.datetime.now()
        print(f"step{step}spend:", t2 - t1)

    end_time = datetime.datetime.now()
    print("end:", end_time)
    print("total",end_time - start_time)
        