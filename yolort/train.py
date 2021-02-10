# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Zhiqiang Wang (zhiqwang@foxmail.com)
import argparse
from pathlib import Path

import pytorch_lightning as pl

from .datasets import VOCDetectionDataModule
from . import models


def get_args_parser():
    parser = argparse.ArgumentParser('You only look once detector', add_help=False)

    parser.add_argument('--arch', default='yolov5s',
                        help='model structure to train')
    parser.add_argument('--data_path', default='./data-bin',
                        help='dataset')
    parser.add_argument('--dataset_type', default='coco',
                        help='dataset')
    parser.add_argument('--dataset_mode', default='instances',
                        help='dataset mode')
    parser.add_argument('--years', default=['2017'], nargs='+',
                        help='dataset year')
    parser.add_argument('--train_set', default='train',
                        help='set of train')
    parser.add_argument('--val_set', default='val',
                        help='set of val')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--max_epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--num_gpus', default=1, type=int, metavar='N',
                        help='number of gpu utilizing (default: 1)')
    parser.add_argument('--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='print frequency')
    parser.add_argument('--output_dir', default='.',
                        help='path where to save')
    return parser


def main(args):
    # Load the data
    datamodule = VOCDetectionDataModule.from_argparse_args(args)

    # Build the model
    model = models.__dict__[args.arch](num_classes=datamodule.num_classes)

    # Create the trainer. Run twice on data
    trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=args.num_gpus)

    # Train the model
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('YOLOv5 training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
