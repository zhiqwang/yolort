# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import argparse
import torch

from ..models import yolov5m


def update_ultralytics_checkpoints(model, checkpoint_path_ultralytics):
    """
    It's limited that ultralytics saved model must load in their root path.
    So a very important thing is to desensitize the path befor updating
    ultralytics's trained model as following:

        >>> checkpoints_ = torch.load(weights, map_location='cpu')['model']
        >>> torch.save(checkpoints_.state_dict(), './checkpoints/yolov5s_ultralytics.pt')
    """
    state_dict = torch.load(checkpoint_path_ultralytics, map_location="cpu")

    # Update backbone features
    for name, params in model.backbone.body.named_parameters(prefix='model'):
        params.data.copy_(state_dict[name])

    for name, buffers in model.backbone.body.named_buffers(prefix='model'):
        buffers.copy_(state_dict[name])

    inner_block_maps = {'0': '9', '1': '10', '3': '13', '4': '14'}
    layer_block_maps = {'0': '17', '1': '18', '2': '20', '3': '21', '4': '23'}

    # Update PAN features
    for name, params in model.backbone.pan.inner_blocks.named_parameters():
        state_key = name.split('.')
        params.data.copy_(state_dict[f"model.{'.'.join([inner_block_maps[state_key[0]]] + state_key[1:])}"])

    for name, buffers in model.backbone.pan.inner_blocks.named_buffers():
        state_key = name.split('.')
        buffers.copy_(state_dict[f"model.{'.'.join([inner_block_maps[state_key[0]]] + state_key[1:])}"])

    for name, params in model.backbone.pan.layer_blocks.named_parameters():
        state_key = name.split('.')
        params.data.copy_(state_dict[f"model.{'.'.join([layer_block_maps[state_key[0]]] + state_key[1:])}"])

    for name, buffers in model.backbone.pan.layer_blocks.named_buffers():
        state_key = name.split('.')
        buffers.copy_(state_dict[f"model.{'.'.join([layer_block_maps[state_key[0]]] + state_key[1:])}"])

    # Update box heads
    for name, params in model.head.named_parameters(prefix='model.24'):
        params.data.copy_(state_dict[name.replace('head', 'm')])

    for name, buffers in model.head.named_buffers(prefix='model.24'):
        buffers.copy_(state_dict[name.replace('head', 'm')])

    return model


def main(args):
    model = yolov5m(pretrained=False, score_thresh=0.25)
    model = update_ultralytics_checkpoints(model, args.checkpoint_path_ultralytics)
    model = model.half()
    torch.save(model.state_dict(), args.checkpoint_path_rt_stack)


def get_args_parser():
    parser = argparse.ArgumentParser('YOLO checkpoint configures', add_help=False)
    parser.add_argument('--checkpoint_path_ultralytics', default='.checkpoints/yolov5s_ultralytics.pt',
                        help='Path of ultralytics trained yolov5 checkpoint model')
    parser.add_argument('--checkpoint_path_rt_stack', default='./checkpoints/yolov5s_rt.pt',
                        help='Path of updated yolov5 checkpoint model')

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Update checkpoint from ultralytics yolov5', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
