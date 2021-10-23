# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import argparse

from yolort.utils import load_from_ultralytics


def get_parser():
    parser = argparse.ArgumentParser(
        "Convert checkpoint weights trained by yolov5 to yolort", add_help=True
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="The path of checkpoint weights",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="r6.0",
        help="Upstream version released by the ultralytics/yolov5, Possible "
        "values are ['r3.1', 'r4.0', 'r6.0']. Default: 'r6.0'.",
    )
    # Dataset Configuration
    parser.add_argument(
        "--image_path",
        default="./data-bin/coco128/images/train2017",
        help="Root path of the dataset containing images",
    )

    parser.add_argument("--output_dir", default=".", help="Path where to save")
    return parser


def convert_yolov5_to_yolort(checkpoint_path, version):
    model_info = load_from_ultralytics(checkpoint_path, version=version)
    return model_info


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    print(f"Command Line Args: {args}")
    convert_yolov5_to_yolort(args)


if __name__ == "__main__":
    cli_main()
