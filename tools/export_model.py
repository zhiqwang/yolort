import argparse
from pathlib import Path

from yolort.runtime.ort_helper import export_onnx


def get_parser():
    parser = argparse.ArgumentParser("CLI tool for exporting models.", add_help=True)

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="The path of checkpoint weights",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        help="The path of the exported ONNX models",
    )
    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="Export the vanilla YOLO model.",
    )
    parser.add_argument(
        "--score_thresh",
        default=0.25,
        type=float,
        help="Score threshold used for postprocessing the detections.",
    )
    parser.add_argument(
        "--nms_thresh",
        default=0.45,
        type=float,
        help="IOU threshold used for doing the NMS.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="r6.0",
        help="Upstream version released by the ultralytics/yolov5, Possible "
        "values are ['r3.1', 'r4.0', 'r6.0']. Default: 'r6.0'.",
    )
    parser.add_argument(
        "--image_size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="Image size for inferencing (default: 640, 640).",
    )
    parser.add_argument("--size_divisible", type=int, default=32, help="Stride for pre-processing.")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size for pre-processing.")
    parser.add_argument("--opset", default=11, type=int, help="Opset version for exporing ONNX models")
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model.")

    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    print(f"Command Line Args: {args}")
    checkpoint_path = Path(args.checkpoint_path)
    assert checkpoint_path.exists(), f"Not found checkpoint file at '{checkpoint_path}'"

    # Save the ONNX model path in the same directory of the checkpoint if not determined
    onnx_path = args.onnx_path
    onnx_path = onnx_path or checkpoint_path.with_suffix(".onnx")

    export_onnx(
        onnx_path,
        checkpoint_path=checkpoint_path,
        size=tuple(args.image_size),
        size_divisible=args.size_divisible,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        version=args.version,
        skip_preprocess=args.skip_preprocess,
        opset_version=args.opset,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    cli_main()
