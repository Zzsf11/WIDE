import datetime
import os
import time
import math

import torch
import torch.utils.data

from collections import OrderedDict
import numpy as np

import utils

from dataset import dataset_dict
from model import model_dict
import loss
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import torchvision

def get_scheduler_function(name, total_iters, final_lr=0):
    print("LR Scheduler: {}".format(name))
    if name == 'cosine':
        return lambda step: ((1 + math.cos(step * math.pi / total_iters)) / 2) * (1 - final_lr) + final_lr
    elif name == 'linear':
        return lambda step: 1 - (1 - final_lr) / total_iters * step
    elif name == 'exp':
        return lambda step: (1 - step / total_iters) ** 0.9
    elif name == 'none':
        return lambda step: 1
    else:
        raise ValueError(name)
                  
def warmup(num_iter, num_warmup, optimizer):
    if num_iter < num_warmup:
        # warm up
        xi = [0, num_warmup]
        for j, x in enumerate(optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            #x['lr'] = np.interp(num_iter, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
            x['lr'] = np.interp(num_iter, xi, [0.1 if j == 2 else 0.0, x['lr']])
            if 'momentum' in x:
                x['momentum'] = np.interp(num_iter, xi, [0.8, 0.9])

def fix_BN_stat(module):
    classname = module.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        module.eval()
    #if classname.find('LayerNorm') != -1:
    #    module.eval()

def freeze_BN_stat(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        if hasattr(model.module, 'backbone'):
            print("freeze backbone BN stat")
            model.module.backbone.apply(fix_BN_stat)
    


def CD_evaluate(model, data_loader, device, save_imgs_dir=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    coco_metric_names = {
        'mAP': 0,
        'mAP_50': 1,
        'mAP_75': 2,
        'mAP_s': 3,
        'mAP_m': 4,
        'mAP_l': 5,
        'AR@100': 6,
        'AR@300': 7,
        'AR@1000': 8,
        'AR_s@1000': 9,
        'AR_m@1000': 10,
        'AR_l@1000': 11
    }
    # metric_logger.add_meter('Prec', utils.SmoothedValue(window_size=1, fmt='{value:.3f} ({global_avg:.3f})'))
    header = 'Test:'
    with torch.no_grad():
        coco_gt = COCO()
        coco_dt = COCO()
        gt_annotations = []
        dt_annotations = []
        gt_ann_id = 1  # Initialize ground truth annotation ID
        dt_ann_id = 1  # Initialize detection annotation ID
        idx = 0
        boxes = {}
        for image, target in metric_logger.log_every(data_loader, 50, header):
            image = image.to(device)
            for t_img in target:
                for t in t_img:
                    x1, y1, w, h = t.pop('bbox')
                    t['boxes'] = [x1, y1, x1+w, y1+h]
                    t['boxes'] = torch.tensor(t['boxes']).to(device).unsqueeze(0)
                    t['labels'] = torch.tensor(t['category_id']).to(device).unsqueeze(0)
                    gt_annotations.append({
                        'id': gt_ann_id,
                        'image_id': t['image_id'],
                        'category_id': t['category_id'],
                        'bbox': [x1, y1, w, h],
                        'area': w * h,
                        'iscrowd': 0
                    })
                    gt_ann_id += 1
                    ###
                    # if t[0]['category_id'] == 1:
                    #     boxes[t[0]['image_id']] = [x1, y1, x1+w, y1+h]
            
            a = target[0]
            target = [t[0] for t in target]
            detections = model(image)
            for i, det in enumerate(detections):
                ###
                # if target[i]['image_id'] in boxes:
                #     box = [int(b) for b in boxes[target[i]['image_id']]]
                #     score = 1
                #     label = 1
                #     dt_annotations.append({
                #         'id': dt_ann_id,
                #         'image_id': target[i]['image_id'],
                #         'category_id': label,
                #         'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                #         'score': score
                #     })
                #     detections[0]['boxes']=torch.tensor([box]).to(device)
                #     detections[0]['scores']=torch.tensor([score]).to(device)
                #     detections[0]['labels']=torch.tensor([label]).to(device)
                #     continue
                for j in range(len(det['boxes'])):
                    box = det['boxes'][j].cpu().numpy()
                    score = det['scores'][j].cpu().numpy()
                    label = int(det['labels'][j].cpu().numpy())
                    dt_annotations.append({
                        'id': dt_ann_id,
                        'image_id': target[0]['image_id'],
                        'category_id': label,
                        'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        'score': score
                    })
                    dt_ann_id += 1
                    
            metric_logger.update()
            if save_imgs_dir:
                save_visualization(image, a, model, save_imgs_dir, idx, train=False, dt=detections)
                idx+=1

        # gt_annotations = [
        #     {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [50, 50, 100, 100], 'area': 10000, 'iscrowd': 0},
        #     {'id': 2, 'image_id': 1, 'category_id': 1, 'bbox': [150, 150, 100, 100], 'area': 10000, 'iscrowd': 0}
        # ]
        # dt_annotations = [
        #     {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [50, 50, 100, 100], 'score': 0.9},
        #     {'id': 2, 'image_id': 1, 'category_id': 1, 'bbox': [150, 150, 100, 100], 'score': 0.8},
        #     {'id': 3, 'image_id': 1, 'category_id': 1, 'bbox': [200, 200, 100, 100], 'score': 0.7}  # False positive
        # ]   
        # Add required COCO dataset structure
    if utils.get_world_size() > 1:
        # Gather all annotations from different ranks
        all_gt_annotations = utils.all_gather(gt_annotations)
        all_dt_annotations = utils.all_gather(dt_annotations)
        
        if utils.get_rank() == 0:
            # Flatten gathered lists and update annotation IDs
            gt_annotations = []
            dt_annotations = []
            gt_ann_id = 1
            dt_ann_id = 1
            
            for gt_anns in all_gt_annotations:
                for ann in gt_anns:
                    ann['id'] = gt_ann_id
                    gt_annotations.append(ann)
                    gt_ann_id += 1
                    
            for dt_anns in all_dt_annotations:
                for ann in dt_anns:
                    ann['id'] = dt_ann_id 
                    dt_annotations.append(ann)
                    dt_ann_id += 1
        
    coco_gt.dataset = {
        'annotations': gt_annotations,
        'images': [{'id': ann['image_id']} for ann in gt_annotations], # 所有的ann中提及的image_id
        'categories': [{'id': i} for i in range(1, max(ann['category_id'] for ann in gt_annotations) + 1)]
    }
    coco_gt.createIndex()
    
    if len(dt_annotations) == 0:
        # If no detections, create empty detection results with same structure
        coco_dt = COCO()
        coco_dt.dataset = {
            'annotations': [],
            'images': coco_gt.dataset['images'],
            'categories': coco_gt.dataset['categories']
        }
        coco_dt.createIndex()
    else:
        coco_dt = coco_gt.loadRes(dt_annotations)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    eval_results = OrderedDict()
    coco_eval.params.catIds = data_loader.dataset.cat_ids
    # coco_eval.params.catIds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    coco_eval.params.iouThrs = np.array([0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.80, 0.85, 0.9, 0.95])  # 数值类型导致iou没法读取
    coco_eval.params.useCats = False
    metric_items = [
        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
    ]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    for metric_item in metric_items:
        key = f'bbox_{metric_item}'
        val = coco_eval.stats[coco_metric_names[metric_item]]
        eval_results[key] = float(f'{round(val, 3)}')
    
    return eval_results['bbox_mAP']

def SS_evaluate(model, data_loader, device, save_imgs_dir=None):
    model.eval()
    num_classes = data_loader.dataset.num_classes
    if hasattr(data_loader.dataset, 'class_mask'):
        class_mask = data_loader.dataset.class_mask
    else:
        class_mask = None
    confmat = utils.ConfusionMatrix(num_classes, class_mask)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    count = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            if isinstance(output, OrderedDict):
                output = output['out']
            pred = output.argmax(1)
            confmat.update(target.flatten(), pred.flatten())
            if save_imgs_dir:
                if count > 100:
                    continue
                output_pil = data_loader.dataset.get_pil(image[0], target[0], pred[0])
                output_pil.save(os.path.join(save_imgs_dir, "{}_{}.png".format(utils.get_rank(), count)))
                count += 1
        confmat.reduce_from_all_processes()
    print("{} {} confmat: {}".format(
        header,
        data_loader.dataset.name,
        confmat
    ))
    acc_global, acc, iu = confmat.compute()
    mIoU = confmat.mIoU(iu)
    return mIoU


def save_visualization(image, target, model, save_dir, idx, train=True, dt=None):
    # Get first image from batch for visualization
    img_ori = image[0].cpu()
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = img_ori[-3:] * std + mean
    ref = img_ori[:-3] * std + mean
    img = (img * 255).to(torch.uint8)
    ref = (ref * 255).to(torch.uint8)
    
    # Save original image
    torchvision.utils.save_image(ref/255.0, os.path.join(save_dir, f'ref_{utils.get_rank()}_{idx}.png'))
    
    # Draw bounding boxes on image copy
    img_with_boxes = img.permute(1,2,0).numpy().copy()
    # t = target[0]
    # box = t['boxes'][0].cpu().numpy()
    # label = t['labels'][0].cpu().numpy()
    # cv2.rectangle(img_with_boxes, 
    #             (int(box[0]), int(box[1])), 
    #             (int(box[2]), int(box[3])),
    #             (0,255,0), 2)
    # cv2.putText(img_with_boxes, f'GT: {label}',
    #             (int(box[0]), int(box[1]-10)),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    for t in target:
        for box, label in zip(t['boxes'], t['labels']):
            box = box.cpu().numpy()
            label = label.cpu().numpy()
            cv2.rectangle(img_with_boxes, 
                          (int(box[0]), int(box[1])), 
                          (int(box[2]), int(box[3])),
                          (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f'GT: {label}',
                        (int(box[0]), int(box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save image with ground truth boxes
    cv2.imwrite(os.path.join(save_dir, f'gt_{utils.get_rank()}_{idx}.png'), 
                cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    
    
    if train:
        # Get predictions and draw them
        with torch.no_grad():
            model.eval()
            pred = model(image[0:1])
            if isinstance(pred, list):
                pred = pred[0]
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            img_with_pred = img.permute(1,2,0).numpy().copy()
            for box, score, label in zip(boxes, scores, labels):
                if score > 0:  # Only show confident predictions
                    cv2.rectangle(img_with_pred,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (255,0,0), 2)
                    cv2.putText(img_with_pred, f'Pred: {label} ({score:.2f})',
                                (int(box[0]), int(box[1]-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            
            # Save image with predictions
            cv2.imwrite(os.path.join(save_dir, f'pred_{utils.get_rank()}_{idx}.png'),
                        cv2.cvtColor(img_with_pred, cv2.COLOR_RGB2BGR))
        model.train()
    else:
        pred = dt
        if isinstance(pred, list):
            pred = pred[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        img_with_pred = img.permute(1,2,0).numpy().copy()
        for box, score, label in zip(boxes, scores, labels):
            if score > 0:  # Only show confident predictions
                cv2.rectangle(img_with_pred,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (255,0,0), 2)
                cv2.putText(img_with_pred, f'Pred: {label} ({score:.2f})',
                            (int(box[0]), int(box[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        
        # Save image with predictions
        cv2.imwrite(os.path.join(save_dir, f'pred_{utils.get_rank()}_{idx}.png'),
                    cv2.cvtColor(img_with_pred, cv2.COLOR_RGB2BGR))

def CD_train_one_epoch(model, criterion, optimizer, scaler, data_loader, lr_scheduler, num_warmup, device, epoch, print_freq):
    model.train()
    freeze_BN_stat(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    # metric_logger.add_meter('f1score', utils.SmoothedValue(window_size=1, fmt='{value:.4f} ({global_avg:.4f})'))
    header = 'Epoch: [{}]'.format(epoch)
    warmup(lr_scheduler._step_count, num_warmup, optimizer)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device)
        for t in target:
            x1, y1, w, h = t[0].pop('bbox')
            t[0]['boxes'] = [x1, y1, x1+w, y1+h]
            t[0]['boxes'] = torch.tensor(t[0]['boxes']).to(device).unsqueeze(0)
            t[0]['labels'] = torch.tensor(t[0]['category_id']).to(device).unsqueeze(0)
            # print(t[0]['labels'])
        target = [t[0] for t in target]
        loss_dict = model(image, target)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        #loss.backward()
        #optimizer.step()
        lr_scheduler.step()
        warmup(lr_scheduler._step_count, num_warmup, optimizer)
        
        ############################ Visualize ground truth and predictions########################################
        # save_dir = os.path.join('visualization', f'epoch_{epoch}')
        # os.makedirs(save_dir, exist_ok=True)
        # save_visualization(image, target, model, save_dir)
        ####################            
        
        metric_logger.update(loss=losses.item(), lr=optimizer.param_groups[0]["lr"])

def SS_train_one_epoch(model, criterion, optimizer, scaler, data_loader, lr_scheduler, num_warmup, device, epoch, print_freq):
    model.train()
    freeze_BN_stat(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = "Epoch: [{}]".format(epoch)
    warmup(lr_scheduler._step_count, num_warmup, optimizer)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        if isinstance(output, OrderedDict):
            output = output['out']
        target = target.squeeze()
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()

        lr_scheduler.step()
        warmup(lr_scheduler._step_count, num_warmup, optimizer)
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


def create_dataloader(args):
    dataset = dataset_dict[args.train_dataset](args, train=True)
    dataset_test = dataset_dict[args.test_dataset](args, train=False)
    if args.test_dataset2:
        dataset_test2 = dataset_dict[args.test_dataset2](args, train=False)
    else:
        dataset_test2 = None
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        test_sampler2 = torch.utils.data.distributed.DistributedSampler(dataset_test2) if dataset_test2 else None
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        test_sampler2 = torch.utils.data.SequentialSampler(dataset_test2) if dataset_test2 else None

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, drop_last=True, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn,
        pin_memory=True)

    if dataset_test2:
        data_loader_test2 = torch.utils.data.DataLoader(
            dataset_test2, batch_size=1,
            sampler=test_sampler2, num_workers=args.workers,
            collate_fn=utils.collate_fn)
    else:
        data_loader_test2 = None

    return dataset, train_sampler, data_loader, dataset_test, data_loader_test, dataset_test2, data_loader_test2
    

def prepare_train(args, model_without_ddp, dataset, data_loader):
    if "fcn" in args.model or "deeplabv3" in args.model:
        params_to_optimize = [
            {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
        ]
    else:
        params_to_optimize = model_without_ddp.parameters()
        
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, betas=(args.momentum, 0.999), weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(args.opt)

    if args.loss_weight:
        print("computing loss weight")
        ratio = dataset.get_mask_ratio()
        loss_weight = torch.tensor(ratio).cuda()
        print("loss weight {}".format(loss_weight))
    else:
        loss_weight = None
    criterion = loss.get_loss(args.loss, loss_weight)
    lambda_lr = get_scheduler_function(args.lr_scheduler, args.epochs * len(data_loader), final_lr=0.2*args.lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    if args.warmup:
        num_warmup = max(round(5 * len(data_loader)), 1000)
    else:
        num_warmup = 0
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    return optimizer, criterion, lr_scheduler, scaler, num_warmup

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    dataset, train_sampler, data_loader, dataset_test, data_loader_test, dataset_test2, data_loader_test2 = create_dataloader(args)

    if args.train_dataset in ['COCO', 'ChangeSim_Multi', 'ChangeSim_Binary', 'ChangeSim_Semantic']:
        train_one_epoch = SS_train_one_epoch
        evaluate = SS_evaluate
    else:
        train_one_epoch = CD_train_one_epoch
        evaluate = CD_evaluate

    args.num_classes = dataset.num_classes
    model = model_dict[args.model](args)
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer, criterion, lr_scheduler, scaler, num_warmup = prepare_train(args, model_without_ddp, dataset, data_loader)    

    if args.pretrained:
        utils.load_model(model_without_ddp, args.pretrained)

    if args.resume:
        print("load from: {}".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        sd = checkpoint['model']
        ret = model_without_ddp.load_state_dict(sd, strict=not args.test_only)
        print("load ret: {}".format(ret))
        if not args.test_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        if args.save_imgs:
            save_imgs_dir = os.path.join(args.output_dir, 'img')
            os.makedirs(save_imgs_dir, exist_ok=True)
        else:
            save_imgs_dir = None
        f1score = evaluate(model, data_loader_test, device=device, save_imgs_dir=save_imgs_dir)
        print(f1score)
        return

    best = -1
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, scaler, data_loader, lr_scheduler, num_warmup, device, epoch, args.print_freq)
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args
        }
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, f'{epoch}checkpoint.pth'))
        if epoch % args.eval_every == 0:
            evaluate(model, data_loader_test, device=device)
            if dataset_test2:
                _ = evaluate(model, data_loader_test2, device=device)
        # print(f"Memory Allocated: {torch.cuda.memory_allocated()} bytes")
        # print(f"Memory Reserved: {torch.cuda.memory_reserved()} bytes")
        # import gc

        # # 打印所有未被垃圾回收的对象
        # gc.collect()
        # print(gc.garbage)
        # # del checkpoint
        # import gc
        # gc.collect()
        # torch.cuda.empty_cache()
        
        # if f1score > best:
        #     best = f1score
        #     utils.save_on_master(
        #         checkpoint,
        #         os.path.join(args.output_dir, 'best.pth'))

    if args.train_dataset in ['COCO']:
        save_imgs_dir = None
    else:
        save_imgs_dir = os.path.join(args.output_dir, '{}_img'.format(dataset_test.name))
        os.makedirs(save_imgs_dir, exist_ok=True)
    _ = evaluate(model, data_loader_test, device=device, save_imgs_dir=save_imgs_dir)
    if dataset_test2:
        save_imgs_dir = os.path.join(args.output_dir, '{}_img'.format(dataset_test2.name))
        os.makedirs(save_imgs_dir, exist_ok=True)
        _ = evaluate(model, data_loader_test2, device=device, save_imgs_dir=save_imgs_dir)
    utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch change detection', add_help=add_help)
    parser.add_argument('--train-dataset', default='VL_CMU_CD', help='dataset name')
    parser.add_argument('--test-dataset', default='VL_CMU_CD', help='dataset name')
    parser.add_argument('--test-dataset2', default='', help='dataset name')
    parser.add_argument('--input-size', default=448, type=int, metavar='N',
                        help='the input-size of images')
    parser.add_argument('--randomflip', default=0.5, type=float, help='random flip input')
    parser.add_argument('--randomrotate', dest="randomrotate", action="store_true", help='random rotate input')
    parser.add_argument('--randomcrop', dest="randomcrop", action="store_true", help='random crop input')
    parser.add_argument('--data-cv', default=0, type=int, metavar='N',
                        help='the number of cross validation')

    parser.add_argument('--model', default='resnet18_mtf_msf_deeplabv3', help='model')
    parser.add_argument('--mtf', default='iade', help='choose branches to use')
    parser.add_argument('--msf', default=4, type=int, help='the number of MSF layers')
    
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=12, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--loss', default='bi', type=str, help='the training loss')
    parser.add_argument('--loss-weight', action="store_true", help='add weight for loss')
    parser.add_argument('--opt', default='sgd', type=str, help='the optimizer')
    parser.add_argument('--lr-scheduler', default='cosine', type=str, help='the lr scheduler')
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--warmup', dest="warmup", action="store_true", help='warmup the lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--pretrained", default='', help='pretrain checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval-every', default=1, type=int, metavar='N',
                        help='eval the model every n epoch')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--save-imgs", dest="save_imgs", action="store_true",
                        help="save the predicted mask")

    
    parser.add_argument("--save-local", dest="save_local", help="save logs to local", action="store_true")
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--e', default=7, type=int,
                        help='number of epoch')
    return parser


if __name__ == "__main__":
    #os.environ["TORCH_HOME"] = '/Pretrained'
    args = get_args_parser().parse_args()
    output_dir = 'output'
    # save_path = "{}_{}_{}/{date:%Y-%m-%d_%H:%M:%S}".format(
    #     args.model, args.train_dataset, args.data_cv, date=datetime.datetime.now())
    save_path = "{}_{}_{}/{}".format(
        args.model, args.train_dataset, args.data_cv, args.e)
    args.output_dir = os.path.join(output_dir, save_path)

    main(args)