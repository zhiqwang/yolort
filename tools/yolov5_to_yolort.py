# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import argparse
from pathlib import Path

from yolort.utils import convert_yolov5_to_yolort


def get_parser():
    parser = argparse.ArgumentParser("Convert checkpoints from yolov5 to yolort", add_help=True)

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path of the checkpoint weights",
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
        type=str,
        default="./test/assets/zidane.jpg",
        help="Path of the test image",
    )

    parser.add_argument("--output_path", type=str, default=None, help="Path where to save")
    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    print(f"Command Line Args: {args}")
    checkpoint_path = Path(args.checkpoint_path)
    assert checkpoint_path.exists(), f"Not found checkpoint file at '{checkpoint_path}'"

    if args.output_path is None:
        args.output_path = checkpoint_path.parent
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    convert_yolov5_to_yolort(checkpoint_path, output_path, version=args.version)


if __name__ == "__main__":
    cli_main()
