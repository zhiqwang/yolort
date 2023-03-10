from pathlib import Path

import torch
from torch import nn

from yolort.models._checkpoint import load_from_ultralytics
from yolort.models.backbone_utils import darknet_pan_backbone
from yolort.models.box_head import YOLOHead


class YOLO(nn.Module):
    def __init__(self, backbone: nn.Module, strides, num_anchors, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = YOLOHead(backbone.out_channels, num_anchors, strides, num_classes)

    def forward(self, samples):

        # get the features from the backbone
        features = self.backbone(samples)

        # compute the yolo heads outputs using the features
        head_outputs = self.head(features)
        return head_outputs


class ModelWrapper(torch.nn.Module):
    """
    Wrapper class for model with dict/list rvalues.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Init call.
        """
        super().__init__()
        self.model = model

    def forward(self, input_x):
        """
        Wrap forward call.
        """
        data = self.model(input_x)

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)  # type: ignore

        elif isinstance(data, list):
            data = tuple(data)

        return data


def make_model(checkpoint_path, version):

    model_info = load_from_ultralytics(checkpoint_path, version=version)

    backbone_name = f"darknet_{model_info['size']}_{version.replace('.', '_')}"
    depth_multiple = model_info["depth_multiple"]
    width_multiple = model_info["width_multiple"]
    use_p6 = model_info["use_p6"]
    backbone = darknet_pan_backbone(
        backbone_name, depth_multiple, width_multiple, version=version, use_p6=use_p6
    )
    strides = model_info["strides"]
    num_anchors = len(model_info["anchor_grids"][0]) // 2
    num_classes = model_info["num_classes"]
    model = YOLO(backbone, strides, num_anchors, num_classes)

    model.load_state_dict(model_info["state_dict"])
    model = ModelWrapper(model)

    model = model.eval()

    return model


def main():
    import argparse

    parser = argparse.ArgumentParser("ptq tool.", add_help=True)

    parser.add_argument(
        "--checkpoint_path", type=str, default="yolov5s.pt", help="The path of checkpoint weights"
    )
    parser.add_argument("--version", type=str, default="r6.0", help="opset version")
    parser.add_argument("--threshold", type=float, default=0.25, help="threshold")
    parser.add_argument("--device", type=str, default="cuda", help="opset version")
    parser.add_argument("--input_size", default=[3, 640, 640], type=int, help="input size")
    parser.add_argument("--opset_version", type=int, default=11, help="opset version")
    parser.add_argument("--onnx_input_name", type=str, default="dummy_input", help="onnx input name")
    parser.add_argument("--onnx_output_name", type=str, default="dummy_output", help="onnx output name")
    parser.add_argument("--onnx_output_path", type=str, default="yolov5.onnx", help="onnx output name")

    args = parser.parse_args()

    print(f"Command Line Args: {args}")

    # model initilize
    checkpoint_path = Path(args.checkpoint_path)
    assert checkpoint_path.exists(), f"Not found checkpoint file at '{checkpoint_path}'"
    model = make_model(checkpoint_path, args.version)
    model = model.to(args.device)
    model.eval()

    # Export torch checkpoint to ONNX
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
