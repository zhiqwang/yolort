import argparse
from pathlib import Path

import pytorch_lightning as pl

from .data import COCODetectionDataModule
from . import models


def get_args_parser():
    parser = argparse.ArgumentParser('You only look once detector', add_help=False)

    parser.add_argument('--arch', default='yolov5s',
                        help='model structure to train')
    parser.add_argument('--max_epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--num_gpus', default=1, type=int, metavar='N',
                        help='number of gpu utilizing (default: 1)')

    parser.add_argument('--data_path', default='./data-bin',
                        help='root path of the dataset')
    parser.add_argument('--anno_path', default=None,
                        help='root path of annotation files')
    parser.add_argument('--data_task', default='instances',
                        help='dataset mode')
    parser.add_argument('--train_set', default='train2017',
                        help='name of train dataset')
    parser.add_argument('--val_set', default='val2017',
                        help='name of val dataset')
    parser.add_argument('--skip_train_set', action='store_true',
                        help='Skip train set')
    parser.add_argument('--skip_val_set', action='store_true',
                        help='Skip val set')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--output_dir', default='.',
                        help='path where to save')
    return parser


def main(args):
    # Load the data
    datamodule = COCODetectionDataModule.from_argparse_args(args)

    # Build the model
    model = models.__dict__[args.arch](num_classes=datamodule.num_classes)

    # Create the trainer. Run twice on data
    trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=args.num_gpus)

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Save it!
    trainer.save_checkpoint("object_detection_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('YOLOv5 training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
