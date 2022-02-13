# Copyright (c) 2022, yolort team. All rights reserved.

from pathlib import Path
from typing import Optional

import torch
from yolort.models import YOLO, YOLOv5


def export_onnx(
    checkpoint_path,
    score_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    version: str = "r6.0",
    onnx_path: Optional[str] = None,
    batch_size: int = 1,
    skip_preprocess: bool = False,
    opset_version: int = 11,
) -> None:
    """
    Export to ONNX models that can be used for ONNX Runtime inferencing.

    Args:
        checkpoint_path (string): Path of the custom trained YOLOv5 checkpoint.
        score_thresh (float): Score threshold used for postprocessing the detections.
            Default: 0.25
        nms_thresh (float): NMS threshold used for postprocessing the detections. Default: 0.45
        version (string): Upstream YOLOv5 version. Default: 'r6.0'
        batch_size (int): Batch size for exporting YOLOv5.
        skip_preprocess (bool): Skip the preprocessing transformation when exporting the ONNX
            models. Default: False
        opset_version (int): Opset version for exporting ONNX models. Default: 11
    """
    onnx_builder = ONNXBuilder(
        checkpoint_path=checkpoint_path,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        version=version,
        batch_size=batch_size,
        skip_preprocess=skip_preprocess,
        opset_version=opset_version,
    )

    onnx_builder.to_onnx(onnx_path)


class ONNXBuilder:
    """
    YOLOv5 wrapper for exporting ONNX models.

    Args:
        checkpoint_path (string): Path of the custom trained YOLOv5 checkpoint.
        score_thresh (float): Score threshold used for postprocessing the detections.
            Default: 0.25
        nms_thresh (float): NMS threshold used for postprocessing the detections. Default: 0.45
        version (string): Upstream YOLOv5 version. Default: 'r6.0'
        batch_size (int): Batch size for exporting YOLOv5.
        skip_preprocess (bool): Skip the preprocessing transformation when exporting the ONNX
            models. Default: False
        opset_version (int): Opset version for exporting ONNX models. Default: 11
    """

    def __init__(
        self,
        checkpoint_path: str,
        score_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        version: str = "r6.0",
        batch_size: int = 1,
        skip_preprocess: bool = False,
        opset_version: int = 11,
    ) -> None:

        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.skip_preprocess = skip_preprocess
        self.batch_size = batch_size
        self.version = version
        self.opset_version = opset_version

        self.model = self.build_model()
        self.input_names = ["images"]
        self.output_names = ["scores", "labels", "boxes"]
        self.input_sample = self.get_input_sample()
        self.dynamic_axes = self.get_dynamic_axes()

    def build_model(self):
        if self.skip_preprocess:
            model = YOLO.load_from_yolov5(
                self.checkpoint_path,
                score_thresh=self.score_thresh,
                nms_thresh=self.nms_thresh,
                version=self.version,
            )
        else:
            model = YOLOv5.load_from_yolov5(
                self.checkpoint_path,
                score_thresh=self.score_thresh,
                nms_thresh=self.nms_thresh,
                version=self.version,
            )

        model = model.eval()
        return model

    def get_dynamic_axes(self):
        if self.skip_preprocess:
            return {
                "images": {0: "batch", 2: "height", 3: "width"},
                "boxes": {0: "batch", 1: "num_objects"},
                "labels": {0: "batch", 1: "num_objects"},
                "scores": {0: "batch", 1: "num_objects"},
            }
        else:
            return {
                "images": {1: "height", 2: "width"},
                "boxes": {0: "num_objects"},
                "labels": {0: "num_objects"},
                "scores": {0: "num_objects"},
            }

    def get_input_sample(self):
        if self.skip_preprocess:
            return torch.rand(self.batch_size, 3, 640, 640)
        else:
            return [torch.rand(3, 640, 640)] * self.batch_size

    @torch.no_grad()
    def to_onnx(self, onnx_path: Optional[str], **kwargs):
        """
        Saves the model in ONNX format.

        Args:
            onnx_path (string, optional): The path to the ONNX graph to load. Default: None
            **kwargs: Will be passed to torch.onnx.export function.
        """
        onnx_path = onnx_path or Path(self.checkpoint_path).with_suffix(".onnx")

        torch.onnx.export(
            self.model,
            self.input_sample,
            onnx_path,
            do_constant_folding=True,
            opset_version=self.opset_version,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            **kwargs,
        )
