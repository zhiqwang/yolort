# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import argparse
from pathlib import Path

import pytorch_lightning as pl

from .data import COCODataModule
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
    parser.add_argument('--val_set', default='val',
                        help='set of validation')
    parser.add_argument('--image_size', default=640, type=int,
                        help='image size to predict')
    parser.add_argument('--score_thresh', default=0.01, type=float,
                        help='threshold of the score')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--num_gpus', default=1, type=int, metavar='N',
                        help='number of gpu utilizing (default: 1)')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--output_dir', default='.',
                        help='path where to save')
    return parser


def main(args):
    # Load the data module
    data_module = COCODataModule.from_argparse_args(args)

    # Build the model
    model = models.__dict__[args.arch](
        pretrained=True,
        min_size=args.image_size,
        max_size=args.image_size,
        score_thresh=args.score_thresh,
    )

    # test step
    trainer = pl.Trainer(max_epochs=1)
    trainer.test(model, test_dataloaders=data_module.val_dataloader)
    # test epoch end
    results = model.evaluator.compute()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser('YOLOv5 evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
