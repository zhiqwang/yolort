# Copyright (c) 2021, yolort team. All rights reserved.
#
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from collections import OrderedDict, namedtuple

import numpy as np
import torch
from torch import Tensor

from typing import Dict, List
import torch
from torch import Tensor
from torchvision.ops import box_convert, boxes as box_ops

try:
    import tensorrt as trt
except ImportError:
    trt = None

logging.basicConfig(level=logging.INFO)
logging.getLogger("PredictorTRT").setLevel(logging.INFO)
log = logging.getLogger("PredictorTRT")


class PredictorTRT:
    """
    Create a simple end-to-end predictor with the given checkpoint that runs on
    single device for a single input image.

    Args:
        engine_path (str): Path of the ONNX checkpoint.

    Examples:
        >>> from yolort.runtime import PredictorTRT
        >>>
        >>> engine_path = 'yolov5s.engine'
        >>> detector = PredictorTRT(engine_path)
        >>>
        >>> img_path = 'bus.jpg'
        >>> scores, class_ids, boxes = detector.run_on_image(img_path)
    """

    def __init__(
        self,
        engine_path: str,
        device: torch.device = torch.device("cuda"),
        score_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        detections_per_img: int = 100,
    ) -> None:
        self.device = device
        self.named_binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        self.logger = trt.Logger(trt.Logger.INFO)
        self.stride = 32
        self.names = [f'class{i}' for i in range(1000)]  # assign defaults
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.detections_per_img = detections_per_img

        self.engine = None
        log.info(f"Loading {engine_path} for TensorRT inference...")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.bindings = OrderedDict()
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            shape = tuple(self.engine.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            self.bindings[name] = self.named_binding(name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.engine.create_execution_context()
        self.batch_size = self.bindings["images"].shape[0]

    def __call__(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (Tuple[List[float], List[int], List[float, float]]):
                stands for scores, labels and boxes respectively.
        """
        assert image.shape == self.bindings["images"].shape, (image.shape, self.bindings["images"].shape)
        self.binding_addrs["images"] = int(image.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        pred_logits = self.bindings["output"].data
        return pred_logits

    def run_on_image(self, image):
        """
        Run the TensorRT engine for one image only.

        Args:
            image_path (str): The image path to be predicted.
        """
        pred_logits = self(image)
        detections = self.postprocessing(pred_logits)
        return detections

    @staticmethod
    def _decode_pred_logits(pred_logits: Tensor):
        """
        Decode the prediction logit from the PostPrecess.
        """
        # Compute conf
        # box_conf x class_conf, w/ shape: num_anchors x num_classes
        scores = pred_logits[:, 5:] * pred_logits[:, 4:5]
        boxes = box_convert(pred_logits[:, :4], in_fmt="cxcywh", out_fmt="xyxy")

        return boxes, scores

    def postprocessing(self, pred_logits: Tensor):
        batch_size = pred_logits.shape[0]
        detections: List[Dict[str, Tensor]] = []

        for idx in range(batch_size):  # image idx, image inference
            # Decode the predict logits
            boxes, scores = self._decode_pred_logits(pred_logits[idx])

            # remove low scoring boxes
            inds, labels = torch.where(scores > self.score_thresh)
            boxes, scores = boxes[inds], scores[inds, labels]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, labels, self.iou_thresh)
            # Keep only topk scoring head_outputs
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            detections.append({"scores": scores, "labels": labels, "boxes": boxes})

        return detections

    def warmup(self, img_size=(1, 3, 320, 320), half=False):
        # Warmup model by running inference once
        # only warmup GPU models
        if isinstance(self.device, torch.device) and self.device.type != 'cpu':
            im = torch.zeros(*img_size).to(self.device).type(torch.half if half else torch.float)
            self(im)
