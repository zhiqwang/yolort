# Copyright (c) 2022, yolort team. All rights reserved.

import argparse

from yolort.utils.yolo2coco import YOLO2COCO


def get_parser():
    parser = argparse.ArgumentParser("Datasets converter from yolo to coco", add_help=True)

    parser.add_argument("--data_source", default="./coco128", help="Dataset root path")
    parser.add_argument("--class_names", default="./coco.name", help="Path of the label names")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split part")
    parser.add_argument("--year", default=2017, type=int, help="Year of the dataset")

    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    print(f"Command Line Args: {args}")

    converter = YOLO2COCO(args.data_source, args.class_names, split=args.split, year=args.year)
    converter.generate()


if __name__ == "__main__":
    cli_main()
