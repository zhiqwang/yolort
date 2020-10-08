# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import argparse
import torch

from hubconf import yolov5


def main(args):
    model = yolov5(cfg_path=args.cfg_path)

    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model'].float().state_dict()  # to FP32

    body_state_dict = {
        f'body.features.{k[6:]}': v for k, v in state_dict.items() if (
            (k[6:].split('.')[0] in model.body.features.keys()) and (
                model.body.features.state_dict()[k[6:]].shape == v.shape))}

    head_state_dict = {
        f'box_head.{k[9:]}': v for k, v in state_dict.items() if (
            k[9:] in model.box_head.state_dict().keys())}  # filter

    model_state_dict = {**body_state_dict, **head_state_dict}

    model.load_state_dict(model_state_dict)

    torch.save(model.state_dict(), args.updated_checkpoint_path)


def get_args_parser():
    parser = argparse.ArgumentParser('YOLO checkpoint configures', add_help=False)
    parser.add_argument('--checkpoint-path', default='./yolov5s.pt',
                        help='Path of ultralytics trained yolov5 checkpoint model')
    parser.add_argument('--cfg-path', default='./models/yolov5s.yaml',
                        help='Path of yolov5 configures model')
    parser.add_argument('--updated-checkpoint-path', default='./checkpoints/yolov5/yolov5s_updated.pt',
                        help='Path of updated yolov5 checkpoint model')

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Update checkpoint from ultralytics yolov5', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
