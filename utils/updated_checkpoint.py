# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import argparse
import torch

from hubconf import yolov5


def main(args):
    model = yolov5(cfg_path=args.cfg_path)
    model.update_ultralytics(args.checkpoint_path)

    torch.save(model.state_dict(), args.updated_checkpoint_path)


def get_args_parser():
    parser = argparse.ArgumentParser('YOLO checkpoint configures', add_help=False)
    parser.add_argument('--checkpoint-path', default='./yolov5s.pt',
                        help='Path of ultralytics trained yolov5 checkpoint model')
    parser.add_argument('--cfg-path', default='./models/yolov5s.yaml',
                        help='Path of yolov5 configures model')
    parser.add_argument('--updated-checkpoint-path', default='./checkpoints/yolov5/yolov5s.pt',
                        help='Path of updated yolov5 checkpoint model')

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Update checkpoint from ultralytics yolov5', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
