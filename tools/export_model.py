import argparse
from pathlib import Path

import onnx
import torch
from torchvision.ops._register_onnx_ops import _onnx_opset_version as DEFAULT_OPSET

try:
    import onnxsim
except ImportError:
    onnxsim = None

from yolort.models import YOLO, YOLOv5


def get_parser():
    parser = argparse.ArgumentParser("CLI tool for exporting models.", add_help=True)

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="The path of checkpoint weights",
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
        "--version",
        type=str,
        default="r6.0",
        help="Upstream version released by the ultralytics/yolov5, Possible "
        "values are ['r3.1', 'r4.0', 'r6.0']. Default: 'r6.0'.",
    )
    parser.add_argument(
        "--export_friendly",
        action="store_true",
        help="Replace torch.nn.SiLU with SiLU.",
    )
    parser.add_argument(
        "--image_size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="Image size for evaluation (default: 640, 640).",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
    parser.add_argument("--opset", default=DEFAULT_OPSET, type=int, help="opset_version")
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model.")

    return parser


def export_onnx(
    model,
    inputs,
    export_onnx_path,
    dynamic_axes,
    input_names=["images_tensors"],
    output_names=["scores", "labels", "boxes"],
    opset_version=11,
    enable_simplify=False,
):
    """
    Export the yolort models.

    Args:
        model (nn.Module): The model to be exported.
        inputs (Tuple[torch.Tensor]): The inputs to the model.
        export_onnx_path (str): A string containg a file name. A binary Protobuf
            will be written to this file.
        dynamic_axes (dict): A dictionary of dynamic axes.
        input_names (str): A names list of input names.
        output_names (str): A names list of output names.
        opset_version (int, default is 11): By default we export the model to the
            opset version of the onnx submodule.
        enable_simplify (bool, default is False): Whether to enable simplification
            of the ONNX model.
    """
    torch.onnx.export(
        model,
        inputs,
        export_onnx_path,
        do_constant_folding=True,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    if enable_simplify:
        input_shapes = {input_names[0]: list(inputs[0][0].shape)}
        simplify_onnx(export_onnx_path, input_shapes)


def simplify_onnx(onnx_path, input_shapes):
    if onnxsim is None:
        raise ImportError("onnx-simplifier not found and is required by yolort.")

    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")

    # Load onnx mode
    onnx_model = onnx.load(onnx_path)

    # Simlify the ONNX model
    model_sim, check = onnxsim.simplify(
        onnx_model,
        input_shapes=input_shapes,
        dynamic_input_shape=True,
    )

    assert check, "There is something error when simplifying ONNX model"
    export_onnx_sim_path = onnx_path.with_suffix(".sim.onnx")
    onnx.save(model_sim, export_onnx_sim_path)


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    print(f"Command Line Args: {args}")
    checkpoint_path = Path(args.checkpoint_path)
    assert checkpoint_path.exists(), f"Not found checkpoint file at '{checkpoint_path}'"

    image_size = args.image_size * 2 if len(args.image_size) == 1 else 1  # expand
    if args.skip_preprocess:
        # input data
        inputs = torch.rand(args.batch_size, 3, *image_size)
        dynamic_axes = {
            "images_tensors": {0: "batch", 2: "height", 3: "width"},
            "boxes": {0: "batch", 1: "num_objects"},
            "labels": {0: "batch", 1: "num_objects"},
            "scores": {0: "batch", 1: "num_objects"},
        }
        input_names = ["images_tensors"]
        output_names = ["scores", "labels", "boxes"]
        model = YOLO.load_from_yolov5(
            checkpoint_path,
            score_thresh=args.score_thresh,
            version=args.version,
        )
        model.eval()
    else:
        # input data
        images = [torch.rand(3, *image_size)]
        inputs = (images,)
        dynamic_axes = {
            "images_tensors": {1: "height", 2: "width"},
            "boxes": {0: "num_objects"},
            "labels": {0: "num_objects"},
            "scores": {0: "num_objects"},
        }
        input_names = ["images_tensors"]
        output_names = ["scores", "labels", "boxes"]
        model = YOLOv5.load_from_yolov5(
            checkpoint_path,
            size=tuple(image_size),
            core_thresh=args.score_thresh,
            version=args.version,
        )
        model.eval()

    # export ONNX models
    export_onnx_path = checkpoint_path.with_suffix(".onnx")

    export_onnx(
        model,
        inputs,
        export_onnx_path,
        dynamic_axes,
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
        enable_simplify=args.simplify,
    )


if __name__ == "__main__":
    cli_main()
