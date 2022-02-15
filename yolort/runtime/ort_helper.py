# Copyright (c) 2022, yolort team. All rights reserved.

from typing import Optional, Tuple

import torch
from torch import nn
from yolort.models import YOLO, YOLOv5


def export_onnx(
    onnx_path: str,
    checkpoint_path: Optional[str] = None,
    model: Optional[nn.Module] = None,
    size: Tuple[int, int] = (640, 640),
    size_divisible: int = 32,
    score_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    version: str = "r6.0",
    skip_preprocess: bool = False,
    opset_version: int = 11,
) -> None:
    """
    Export to ONNX models that can be used for ONNX Runtime inferencing.

    Args:
        onnx_path (string): The path to the ONNX graph to be exported.
        checkpoint_path (string, optional): Path of the custom trained YOLOv5 checkpoint.
            Default: None
        model (nn.Module): The defined PyTorch module to be exported. Default: None
        size: (Tuple[int, int]): the minimum and maximum size of the image to be rescaled.
            Default: (640, 640)
        size_divisible (int): Stride in the preprocessing. Default: 32
        score_thresh (float): Score threshold used for postprocessing the detections.
            Default: 0.25
        nms_thresh (float): NMS threshold used for postprocessing the detections. Default: 0.45
        version (string): Upstream YOLOv5 version. Default: 'r6.0'
        skip_preprocess (bool): Skip the preprocessing transformation when exporting the ONNX
            models. Default: False
        opset_version (int): Opset version for exporting ONNX models. Default: 11
    """

    onnx_builder = ONNXBuilder(
        checkpoint_path=checkpoint_path,
        model=model,
        size=size,
        size_divisible=size_divisible,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        version=version,
        skip_preprocess=skip_preprocess,
        opset_version=opset_version,
    )

    onnx_builder.to_onnx(onnx_path)


class ONNXBuilder:
    """
    YOLOv5 wrapper for exporting ONNX models.

    Args:
        checkpoint_path (string): Path of the custom trained YOLOv5 checkpoint.
        model (nn.Module): The defined PyTorch module to be exported. Default: None
        size: (Tuple[int, int]): the minimum and maximum size of the image to be rescaled.
            Default: (640, 640)
        size_divisible (int): Stride in the preprocessing. Default: 32
        score_thresh (float): Score threshold used for postprocessing the detections.
            Default: 0.25
        nms_thresh (float): NMS threshold used for postprocessing the detections. Default: 0.45
        version (string): Upstream YOLOv5 version. Default: 'r6.0'
        skip_preprocess (bool): Skip the preprocessing transformation when exporting the ONNX
            models. Default: False
        opset_version (int): Opset version for exporting ONNX models. Default: 11
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        size: Tuple[int, int] = (640, 640),
        size_divisible: int = 32,
        score_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        version: str = "r6.0",
        skip_preprocess: bool = False,
        opset_version: int = 11,
    ) -> None:

        super().__init__()
        self._checkpoint_path = checkpoint_path
        self._version = version
        # For post-processing
        self._score_thresh = score_thresh
        self._nms_thresh = nms_thresh
        self._skip_preprocess = skip_preprocess
        # For pre-processing
        self._size = size
        self._size_divisible = size_divisible
        # Define the module
        if model is None:
            model = self._build_model()
        self.model = model

        self.opset_version = opset_version
        self.input_names = ["images"]
        self.output_names = ["scores", "labels", "boxes"]
        self.input_sample = self._get_input_sample()
        self.dynamic_axes = self._get_dynamic_axes()

    def _build_model(self):
        if self._skip_preprocess:
            model = YOLO.load_from_yolov5(
                self._checkpoint_path,
                score_thresh=self._score_thresh,
                nms_thresh=self._nms_thresh,
                version=self._version,
            )
        else:
            model = YOLOv5.load_from_yolov5(
                self._checkpoint_path,
                size=self._size,
                size_divisible=self._size_divisible,
                score_thresh=self._score_thresh,
                nms_thresh=self._nms_thresh,
                version=self._version,
            )

        model = model.eval()
        return model

    def _get_dynamic_axes(self):
        if self._skip_preprocess:
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

    def _get_input_sample(self):
        if self._skip_preprocess:
            return torch.rand(1, 3, 640, 640)
        else:
            return [torch.rand(3, 640, 640)]

    @torch.no_grad()
    def to_onnx(self, onnx_path: str, **kwargs):
        """
        Saves the model in ONNX format.

        Args:
            onnx_path (string): The path to the ONNX graph to be exported.
            **kwargs: Will be passed to torch.onnx.export function.
        """

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
