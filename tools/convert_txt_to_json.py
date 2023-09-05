# Copyright (c) 2022, yolort team. All rights reserved.

import argparse

from yolort.utils import AnnotationsConverter


def get_parser():
    parser = argparse.ArgumentParser("Annotations converter from yolo to coco", add_help=True)

    parser.add_argument("--data_source", default="./coco128", help="Root path of the datasets")
    parser.add_argument("--class_names", default="./coco.name", help="Path of the label names")
    parser.add_argument("--image_dir", default=None, help="Name of the path to be replaced")
    parser.add_argument("--label_dir", default=None, help="Name of the replaced path for desired labels")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split part")
    parser.add_argument("--year", default=2017, type=int, help="Year of the dataset")

    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    print(f"Command Line Args: {args}")

    converter = AnnotationsConverter(
        args.data_source,
        args.class_names,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        split=args.split,
        year=args.year,
    )
    converter.generate()


if __name__ == "__main__":
    cli_main()