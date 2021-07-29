# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import argparse
import torch
from .yolort_deploy_friendly import yolov5s_r40_deploy_ncnn


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt',
                        help='weights path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640],
                        help='image (height, width)')
    parser.add_argument('--num_classes', type=int, default=80,
                        help='number of classes')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true',
                        help='FP16 half-precision export')
    parser.add_argument('--dynamic', action='store_true',
                        help='ONNX: dynamic axes')
    parser.add_argument('--simplify', action='store_true',
                        help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX: opset version')
    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    export_onnx(args)


def export_onnx(args):

    model = yolov5s_r40_deploy_ncnn(
        pretrained=True,
        num_classes=args.num_classes,
    )
    inputs = torch.rand(args.batch_size, 3, 320, 320)
    outputs = model(inputs)
    print([out.shape for out in outputs])


if __name__ == "__main__":
    cli_main()
