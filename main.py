# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Zhiqiang Wang (zhiqwang@foxmail.com)

import datetime
import os
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc as utils

from datasets import build_dataset, get_coco_api_from_dataset
from models import yolov5s
from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('You only look once detector', add_help=False)

    parser.add_argument('--arch', default='ssd_lite_mobilenet_v2',
                        help='model architecture')
    parser.add_argument('--return-criterion', action='store_true',
                        help='Should be enabled in training mode')
    parser.add_argument('--data-path', default='./data-bin',
                        help='dataset')
    parser.add_argument('--dataset-file', default='coco',
                        help='dataset')
    parser.add_argument('--dataset-mode', default='instances',
                        help='dataset mode')
    parser.add_argument('--dataset-year', default=['2017'], nargs='+',
                        help='dataset year')
    parser.add_argument('--train-set', default='train',
                        help='set of train')
    parser.add_argument('--val-set', default='val',
                        help='set of val')
    parser.add_argument('--model', default='ssd',
                        help='model')
    parser.add_argument("--masks", action="store_true",
                        help="semantic segmentation")
    parser.add_argument('--device', default='cuda',
                        help='device')
    parser.add_argument('--score-thresh', default=0.01, type=float,
                        help='inference score threshold')
    parser.add_argument('--image-size', default=300, type=int,
                        help='input size of models')
    parser.add_argument('--num-classes', default=21, type=int,
                        help='number classes of datasets')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--num-workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--lr-backbone', default=1e-5, type=float)
    parser.add_argument('--lr-scheduler', default='cosine',
                        help='Scheduler for SGD, It can be chosed to multi-step or cosine')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--lr-step-size', default=8, type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 70], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--t-max', default=200, type=int,
                        help='T_max value for Cosine Annealing Scheduler')
    parser.add_argument('--print-freq', default=20, type=int,
                        help='print frequency')
    parser.add_argument('--output-dir', default='.',
                        help='path where to save')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test the model')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pre-trained models from the modelzoo')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print('Loading data')
    dataset_train = build_dataset(args.train_set, args.dataset_year, args)
    dataset_val = build_dataset(args.val_set, args.dataset_year, args)
    base_ds = get_coco_api_from_dataset(dataset_val)

    print('Creating data loaders')
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True,
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    print('Creating model, always set args.return_criterion be True')
    args.return_criterion = True
    model = yolov5s()
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
        )
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.lr_scheduler == 'multi-step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.lr_steps,
            gamma=args.lr_gamma,
        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.t_max)
    else:
        raise ValueError(f'scheduler {args.lr_scheduler} not supported')

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, data_loader_val, base_ds, device)
        return

    print('Start training')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, args.print_freq)

        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master(
                {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                    'epoch': epoch,
                },
                os.path.join(output_dir, 'model_{}.pth'.format(epoch)),
            )

        # evaluate after every epoch
        # evaluate(model, criterion, data_loader_val, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('YOLOv5 training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
