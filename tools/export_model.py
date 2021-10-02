import argparse
from pathlib import Path

import onnx
import torch
from torchvision.ops._register_onnx_ops import _onnx_opset_version as DEFAULT_OPSET

try:
    import onnxsim
except ImportError:
    onnxsim = None

from yolort.models import YOLOv5


def get_parser():
    parser = argparse.ArgumentParser(
        "CLI tool for exporting ONNX models", add_help=True
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="The path of checkpoint weights",
    )
    # Model architecture
    parser.add_argument(
        "--arch",
        choices=["yolov5s", "yolov5m", "yolov5l"],
        default="yolov5s",
        help="Model architecture to export",
    )
    parser.add_argument(
        "--num_classes", default=80, type=int, help="The number of classes"
    )
    parser.add_argument(
        "--score_thresh",
        default=0.25,
        type=float,
        help="Score threshold used for postprocessing the detections.",
    )
    parser.add_argument(
        "--export_friendly", action="store_true", help="Replace torch.nn.silu with Silu"
    )
    parser.add_argument(
        "--image_size",
        default=640,
        type=int,
        help="Image size for evaluation (default: 640)",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument(
        "--opset", default=DEFAULT_OPSET, type=int, help="opset_version"
    )
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")

    return parser


def export_onnx(model, inputs, export_onnx_path, opset_version, enable_simplify):
    torch.onnx.export(
        model,
        inputs,
        export_onnx_path,
        do_constant_folding=True,
        opset_version=opset_version,
        input_names=["images_tensors"],
        output_names=["scores", "labels", "boxes"],
        dynamic_axes={
            "images_tensors": [0, 1, 2],
            "boxes": [0, 1],
            "labels": [0],
            "scores": [0],
        },
    )

    if enable_simplify:
        input_shapes = {"images_tensors": list(inputs[0][0].shape)}
        simplify_onnx(export_onnx_path, input_shapes)


def simplify_onnx(onnx_path, input_shapes):
    if onnxsim is None:
        raise ImportError("onnx-simplifier not found and is required by yolort")

    print(f"Simplifing with onnx-simplifier {onnxsim.__version__}...")

    # Load onnx mode
    onnx_model = onnx.load(onnx_path)

    # Simlify the ONNX model
    model_sim, check = onnxsim.simplify(
        onnx_model,
        input_shapes=input_shapes,
        dynamic_input_shape=True,
    )

    assert check, "Simplified ONNX model could not be validated"
    export_onnx_sim_path = onnx_path.with_suffix(".sim.onnx")
    onnx.save(model_sim, export_onnx_sim_path)


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    print(f"Command Line Args: {args}")
    checkpoint_path = Path(args.checkpoint_path)
    assert checkpoint_path.is_file(), f"Not found checkpoint: {checkpoint_path}"

    # input data
    images = [torch.rand(3, args.image_size, args.image_size)]
    inputs = (images,)

    model = YOLOv5.load_from_yolov5(checkpoint_path, score_thresh=args.score_thresh)
    model.eval()

    # export ONNX models
    export_onnx_path = checkpoint_path.with_suffix(".onnx")
    opset_version = args.opset
    enable_simplify = args.simplify
    export_onnx(model, inputs, export_onnx_path, opset_version, enable_simplify)


if __name__ == "__main__":
    cli_main()
