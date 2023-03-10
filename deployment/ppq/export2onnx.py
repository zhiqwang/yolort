import onnx
import torch
import torchvision
from onnx import helper, version_converter
from onnxsim import simplify

import argparse
import os
import shutil
from pathlib import Path

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

    if args.export2onnx:
        # torch to onnx
        dummy_inputs = torch.randn([1] + args.input_size, device=args.device)
        torch.onnx.export(
            model,
            dummy_inputs,
            args.onnx_output_path,
            args.opset_version,
            do_constant_folding=False,
            input_names=[args.onnx_input_name],
            output_names=[args.onnx_output_name],
        )


if __name__ == "__main__":
    main()
