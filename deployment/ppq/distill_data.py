import argparse
import os
import shutil
from pathlib import Path

import onnx
import torch
import torchvision

from PIL import Image

from utils import calibrate, collate_fn, get_distill_data, get_parser, make_model, prepare_data_loaders
from yolort.models import YOLOv5


def main():

    # parser
    parser = get_parser()
    args = parser.parse_args()
    print(f"Command Line Args: {args}")

    # model initilize
    checkpoint_path = Path(args.checkpoint_path)
    assert checkpoint_path.exists(), f"Not found checkpoint file at '{checkpoint_path}'"
    model = make_model(checkpoint_path, args.version)
    model = model.to(args.device)
    model.eval()

    # distill data
    if args.distill_data:
        distilled_data_path = Path(args.distilled_data_path)
        args.calibration_data_path = distilled_data_path
        if args.regenerate_data and os.path.exists(distilled_data_path):
            shutil.rmtree(distilled_data_path)
        if not os.path.exists(distilled_data_path):
            os.makedirs(distilled_data_path)
        imgs_lists = os.listdir(distilled_data_path)
        sorted(imgs_lists)
        if len(imgs_lists) < args.num_of_batches:
            args.num_of_batches = args.num_of_batches - len(imgs_lists)
            get_distill_data(
                args.distilled_data_path,
                model,
                args.input_size,
                args.batch_size,
                len(imgs_lists) + 1,
                args.distill_iterations,
                args.num_of_batches,
            )


if __name__ == "__main__":
    main()
