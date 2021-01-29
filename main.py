# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Zhiqiang Wang (zhiqwang@foxmail.com)

import datetime
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

import pytorch_lightning as pl

from datasets import build_dataset, get_coco_api_from_dataset, collate_fn
from models import YOLOLitWrapper


def get_args_parser():
    parser = argparse.ArgumentParser('You only look once detector', add_help=False)

    parser.add_argument('--data_path', default='./data-bin',
                        help='dataset')
    parser.add_argument('--dataset_file', default='coco',
                        help='dataset')
    parser.add_argument('--dataset_mode', default='instances',
                        help='dataset mode')
    parser.add_argument('--dataset_year', default=['2017'], nargs='+',
                        help='dataset year')
    parser.add_argument('--train_set', default='train',
                        help='set of train')
    parser.add_argument('--val_set', default='val',
                        help='set of val')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='print frequency')
    parser.add_argument('--output_dir', default='.',
                        help='path where to save')
    return parser


def main(args):

    # Data loading code
    print('Loading data')
    dataset_train = build_dataset(args.train_set, args.dataset_year, args)
    dataset_val = build_dataset(args.val_set, args.dataset_year, args)
    base_ds = get_coco_api_from_dataset(dataset_val)

    print('Creating data loaders')
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True,
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    print('Creating model, always set args.return_criterion be True')
    args.return_criterion = True

    # Load model
    model = YOLOLitWrapper()
    model.train()

    # train
    # trainer = pl.Trainer().from_argparse_args(args)
    trainer = pl.Trainer(max_epochs=1, gpus=1)
    trainer.fit(model, data_loader_train, data_loader_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('YOLOv5 training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
