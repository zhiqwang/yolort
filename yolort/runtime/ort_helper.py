# Copyright (c) 2022, yolort team. All rights reserved.

from typing import Optional, Tuple

import torch
from torch import nn
from yolort.models import YOLO, YOLOv5


def export_onnx(
    onnx_path: str,
    *,
    checkpoint_path: Optional[str] = None,
    model: Optional[nn.Module] = None,
    size: Tuple[int, int] = (640, 640),
    size_divisible: int = 32,
    score_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    version: str = "r6.0",
    skip_preprocess: bool = False,
    opset_version: int = 11,
    batch_size: int = 1,
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
        batch_size (int): Only used for models that include pre-processing, you need to specify
            the batch sizes and ensure that the number of input images is the same as the batches
            when inferring if you want to export multiple batches ONNX models. Default: 1
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
        batch_size=batch_size,
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
        batch_size (int): Only used for models that include pre-processing, you need to specify
            the batch sizes and ensure that the number of input images is the same as the batches
            when inferring if you want to export multiple batches ONNX models. Default: 1
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
        batch_size: int = 1,
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
        self._batch_size = batch_size
        # Define the module
        if model is None:
            model = self._build_model()
        self.model = model

        # For exporting ONNX model
        self._opset_version = opset_version
        self.input_names = self._set_input_names()
        self.output_names = self._set_output_names()
        self.input_sample = self._set_input_sample()
        self.dynamic_axes = self._set_dynamic_axes()

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

    def _set_input_names(self):
        if self._skip_preprocess:
            return ["images"]
        if self._batch_size == 1:
            return ["image"]

        input_names = []
        for i in range(self._batch_size):
            input_names.append(f"image{i + 1}")
        return input_names

    def _set_output_names(self):
        if self._skip_preprocess:
            return ["scores", "labels", "boxes"]
        if self._batch_size == 1:
            return ["score", "label", "box"]

        output_names = []
        for i in range(self._batch_size):
            output_names.extend([f"score{i + 1}", f"label{i + 1}", f"box{i + 1}"])
        return output_names

    def _set_dynamic_axes(self):
        if self._skip_preprocess:
            return {
                "images": {0: "batch", 2: "height", 3: "width"},
                "boxes": {0: "batch", 1: "num_objects"},
                "labels": {0: "batch", 1: "num_objects"},
                "scores": {0: "batch", 1: "num_objects"},
            }
        if self._batch_size == 1:
            return {
                "image": {1: "height", 2: "width"},
                "box": {0: "num_objects"},
                "label": {0: "num_objects"},
                "score": {0: "num_objects"},
            }

        dynamic_axes = {}
        for i in range(self._batch_size):
            dynamic_axes[f"image{i + 1}"] = {1: "height", 2: "width"}
            dynamic_axes[f"box{i + 1}"] = {0: "num_objects"}
            dynamic_axes[f"label{i + 1}"] = {0: "num_objects"}
            dynamic_axes[f"score{i + 1}"] = {0: "num_objects"}
        return dynamic_axes

    def _set_input_sample(self):
        if self._skip_preprocess:
            return torch.rand(1, 3, 640, 640)
        if self._batch_size == 1:
            return [torch.rand(3, 640, 640)]

        return [torch.rand(3, 640, 640)] * self._batch_size

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
            opset_version=self._opset_version,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            **kwargs,
        )
