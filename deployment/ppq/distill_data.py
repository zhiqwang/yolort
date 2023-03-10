import argparse
import os
import shutil
from pathlib import Path

from create_onnx import make_model

from utils import get_distill_data


def main():
    parser = argparse.ArgumentParser("ptq tool.", add_help=True)

    parser.add_argument(
        "--checkpoint_path", type=str, default="yolov5s.pt", help="The path of checkpoint weights"
    )
    parser.add_argument("--input_size", default=[3, 640, 640], type=int, help="input size")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--version", type=str, default="r6.0", help="opset version")
    parser.add_argument("--threshold", type=float, default=0.25, help="threshold")
    parser.add_argument("--device", type=str, default="cuda", help="opset version")
    parser.add_argument(
        "--regenerate_data",
        type=int,
        default=1,
        help="if you wangt to generate new data in place of old data",
    )
    parser.add_argument(
        "--distilled_data_path", type=str, default="./distilled_data/", help="The path of distilled data"
    )
    parser.add_argument(
        "--calibration_data_path",
        type=str,
        default="./distilled_data/",
        help="The path of calibration data, if zeroq is not used, you should set it",
    )
    parser.add_argument("--distill_iterations", type=int, default=50, help="distill iterations")
    parser.add_argument("--num_batches", type=int, default=10, help="num of batches")

    args = parser.parse_args()
    print(f"Command Line Args: {args}")

    # model initilize
    checkpoint_path = Path(args.checkpoint_path)
    assert checkpoint_path.exists(), f"Not found checkpoint file at '{checkpoint_path}'"
    model = make_model(checkpoint_path, args.version)
    model = model.to(args.device)
    model.eval()

    # distill data
    distilled_data_path = Path(args.distilled_data_path)
    args.calibration_data_path = distilled_data_path
    if args.regenerate_data and os.path.exists(distilled_data_path):
        shutil.rmtree(distilled_data_path)
    if not os.path.exists(distilled_data_path):
        os.makedirs(distilled_data_path)
    imgs_lists = os.listdir(distilled_data_path)
    sorted(imgs_lists)

    if len(imgs_lists) < args.num_batches:
        args.num_batches = args.num_batches - len(imgs_lists)
        get_distill_data(
            args.distilled_data_path,
            model,
            args.input_size,
            args.batch_size,
            len(imgs_lists) + 1,
            args.distill_iterations,
            args.num_batches,
        )


if __name__ == "__main__":
    main()
