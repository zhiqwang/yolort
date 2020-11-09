# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import argparse
import torch

from hubconf import yolov5


def update_ultralytics(model, checkpoint_path_ultralytics):

    state_dict = torch.load(checkpoint_path_ultralytics, map_location="cpu")

    # Update body features
    for name, params in model.backbone.body.named_parameters(prefix='model'):
        params.data.copy_(state_dict[name])

    for name, buffers in model.backbone.body.named_buffers(prefix='model'):
        buffers.copy_(state_dict[name])

    # Update box heads
    for name, params in model.head.named_parameters(prefix='model.24'):
        params.data.copy_(state_dict[name.replace('head', 'm')])

    for name, buffers in model.head.named_buffers(prefix='model.24'):
        buffers.copy_(state_dict[name.replace('head', 'm')])

    return model


def main(args):
    model = yolov5(cfg_path=args.cfg_path)
    model = update_ultralytics(model, args.checkpoint_path)

    torch.save(model.state_dict(), args.updated_checkpoint_path)


def get_args_parser():
    parser = argparse.ArgumentParser('YOLO checkpoint configures', add_help=False)
    parser.add_argument('--checkpoint_path', default='./yolov5s.pt',
                        help='Path of ultralytics trained yolov5 checkpoint model')
    parser.add_argument('--cfg_path', default='./models/yolov5s.yaml',
                        help='Path of yolov5 configures model')
    parser.add_argument('--updated_checkpoint_path', default='./checkpoints/yolov5/yolov5s.pt',
                        help='Path of updated yolov5 checkpoint model')

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Update checkpoint from ultralytics yolov5', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
