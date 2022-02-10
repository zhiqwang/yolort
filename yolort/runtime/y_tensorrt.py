# Copyright (c) 2021, yolort team. All rights reserved.
#
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from collections import OrderedDict, namedtuple
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor

try:
    import tensorrt as trt
except ImportError:
    trt = None

logging.basicConfig(level=logging.INFO)
logging.getLogger("PredictorTRT").setLevel(logging.INFO)
logger = logging.getLogger("PredictorTRT")


class PredictorTRT:
    """
    Create a simple end-to-end predictor with the given checkpoint that runs on
    single device for a single input image.

    Args:
        engine_path (string): Path of the ONNX checkpoint.
        device (torch.device): The CUDA device to be used for inferencing.
        precision (string): The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.

    Examples:
        >>> import cv2
        >>> import numpy as np
        >>> import torch
        >>> from yolort.runtime import PredictorTRT
        >>>
        >>> engine_path = 'yolov5n6.engine'
        >>> device = torch.device('cuda')
        >>> runtime = PredictorTRT(engine_path, device)
        >>>
        >>> img_path = 'bus.jpg'
        >>> image = cv2.imread(img_path)
        >>> image = cv2.resize(image, (320, 320))
        >>> image = image.transpose((2, 0, 1))[::-1]  # Convert HWC to CHW, BGR to RGB
        >>> image = np.ascontiguousarray(image)
        >>>
        >>> image = runtime.preprocessing(image)
        >>> detections = runtime.run_on_image(image)
    """

    def __init__(
        self,
        engine_path: str,
        device: torch.device = torch.device("cuda"),
        precision: str = "fp32",
    ) -> None:
        self.engine_path = engine_path
        self.device = device
        self.named_binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        self.stride = 32
        self.names = [f"class{i}" for i in range(1000)]  # assign defaults

        self.engine = self._build_engine()
        self._set_context()

        if precision == "fp32":
            self.half = False
        elif precision == "fp16":
            self.half = True
        else:
            raise NotImplementedError(f"Currently not supports precision: {precision}")

    def _build_engine(self):
        logger.info(f"Loading {self.engine_path} for TensorRT inference...")
        trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(trt_logger, namespace="")
        with open(self.engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        return engine

    def _set_context(self):
        self.bindings = OrderedDict()
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            shape = tuple(self.engine.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            self.bindings[name] = self.named_binding(name, dtype, shape, data, int(data.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.engine.create_execution_context()

    def preprocessing(self, image):
        image = torch.from_numpy(image).to(self.device)
        image = image.half() if self.half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0
        if len(image.shape) == 3:
            image = image[None]  # expand for batch dim
        return image

    def __call__(self, image: Tensor):
        """
        Args:
            image (Tensor): an image of shape (N, C, H, W).

        Returns:
            predictions (Tuple[Tensor, Tensor, Tensor, Tensor]):
                stands for boxes, scores, labels and number of boxes respectively.
        """
        assert image.shape == self.bindings["images"].shape, (image.shape, self.bindings["images"].shape)
        self.binding_addrs["images"] = int(image.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        num_dets = self.bindings["num_detections"].data
        boxes = self.bindings["detection_boxes"].data
        scores = self.bindings["detection_scores"].data
        labels = self.bindings["detection_classes"].data
        return boxes, scores, labels, num_dets

    def run_on_image(self, image: Tensor):
        """
        Run the TensorRT engine for one image only.

        Args:
            image (Tensor): an image of shape (N, C, H, W).
        """
        boxes, scores, labels, num_dets = self(image)

        detections = self.postprocessing(boxes, scores, labels, num_dets)
        return detections

    @staticmethod
    def postprocessing(all_boxes, all_scores, all_labels, all_num_dets):
        detections: List[Dict[str, Tensor]] = []

        for boxes, scores, labels, num_dets in zip(all_boxes, all_scores, all_labels, all_num_dets):
            keep = num_dets.item()
            boxes, scores, labels = boxes[:keep], scores[:keep], labels[:keep]
            detections.append({"scores": scores, "labels": labels, "boxes": boxes})

        return detections

    def warmup(self, img_size=(1, 3, 320, 320)):
        # Warmup model by running inference once
        # only warmup GPU models
        if isinstance(self.device, torch.device) and self.device.type != "cpu":
            image = torch.zeros(*img_size).to(self.device).type(torch.half if self.half else torch.float)
            self(image)

    def run_wo_postprocessing(self, image: Tensor):
        """
        Run the TensorRT engine for one image only.

        Args:
            image (Tensor): an image of shape (N, C, H, W).
        """
        assert image.shape == self.bindings["images"].shape, (image.shape, self.bindings["images"].shape)
        self.binding_addrs["images"] = int(image.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        boxes = self.bindings["boxes"].data
        scores = self.bindings["scores"].data
        return boxes, scores
