# Copyright (c) 2021, yolort team. All rights reserved.
#
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from collections import OrderedDict, namedtuple

import cv2
import numpy as np
import torch
from torch import Tensor

try:
    import tensorrt as trt
except ImportError:
    trt = None

from yolort.utils import read_image_to_tensor
from yolort.v5 import letterbox

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

    def __init__(self, engine_path: str) -> None:
        self.device = torch.device("cuda")
        self.named_binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        self.logger = trt.Logger(trt.Logger.INFO)
        self.engine_path = engine_path

        self._runtime = None
        log.info(f"Loading {self.engine_path} for TensorRT inference...")
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self._runtime = runtime.deserialize_cuda_engine(f.read())

        self.bindings = OrderedDict()
        for index in range(self._runtime.num_bindings):
            name = self._runtime.get_binding_name(index)
            dtype = trt.nptype(self._runtime.get_binding_dtype(index))
            shape = tuple(self._runtime.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            self.bindings[name] = self.named_binding(name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self._runtime.create_execution_context()
        self.batch_size = self.bindings["images"].shape[0]

    def _preprocessing(self, image: np.ndarray) -> Tensor:
        blob = letterbox(image, new_shape=(320, 320), auto=False)[0]
        blob = read_image_to_tensor(blob)
        blob = blob[None]
        return blob

    def __call__(self, image: np.ndarray):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (Tuple[List[float], List[int], List[float, float]]):
                stands for scores, labels and boxes respectively.
        """
        device = torch.device("cuda")
        blob = self._preprocessing(image)
        print(blob.shape)
        blob = blob.to(device)

        assert blob.shape == self.bindings["images"].shape, (blob.shape, self.bindings["images"].shape)
        self.binding_addrs["images"] = int(blob.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        predictions = self.bindings["output"].data
        return predictions

    def run_on_image(self, image_path):
        """
        Run the TensorRT engine for one image only.

        Args:
            image_path (str): The image path to be predicted.
        """
        img = cv2.imread(image_path)
        predictions = self(img)

        return predictions
