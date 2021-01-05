import sys
import time
import math

import torch

import torchvision.models

from datasets.coco_eval import CocoEvaluator

import utils.misc as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images, targets = images.to(device), targets.to(device)

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f'Loss is {loss_value}, stopping training')
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ['bbox']
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append('segm')
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append('keypoints')
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, base_ds, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'

    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 20, header):
        images = images.to(device)

        model_time = time.time()
        target_sizes = torch.stack([t['orig_size'] for t in targets], dim=0).to(device)
        results = model(images, target_sizes=target_sizes)

        model_time = time.time() - model_time

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f'Averaged stats: {metric_logger}')
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator
