# Copyright (c) 2022, yolort team. All rights reserved.

import logging
from collections import OrderedDict, namedtuple
from typing import Any, Dict, List, Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torchvision.io import read_image
from yolort.data import contains_any_tensor
from yolort.models.transform import YOLOTransform

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
    single device for input images with TensorRT.

    Args:
        engine_path (string): Path of the serialized TensorRT engine.
        device (torch.device): The CUDA device to be used for inferencing.
        precision (string): The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        enable_dynamic (bool): Whether to specify axes of tensors as dynamic. Default: False.
        size: (Tuple[int, int]): the minimum and maximum size of the image to be rescaled.
            Default: (640, 640)
        size_divisible (int): stride of the models. Default: 32
        fixed_shape (Tuple[int, int], optional): Padding mode for letterboxing. If set to `True`,
            the image will be padded to shape `fixed_shape` if specified. Instead the image will
            be padded to a minimum rectangle to match `min_size / max_size` and each of its edges
            is divisible by `size_divisible` if it is not specified. Default: None
        fill_color (int): fill value for padding. Default: 114

    Example:

        Demo pipeline for deploying TensorRT.

        .. code-block:: python

            import torch
            from yolort.runtime import PredictorTRT

            # Load the exported TensorRT engine
            engine_path = 'yolov5n6.engine'
            device = torch.device('cuda')
            y_runtime = PredictorTRT(engine_path, device=device)

            # Perform inference on an image file
            predictions = y_runtime.predict('bus.jpg')
    """

    def __init__(
        self,
        engine_path: str,
        device: torch.device = torch.device("cuda"),
        precision: str = "fp32",
        enable_dynamic: bool = False,
        size: Tuple[int, int] = (640, 640),
        size_divisible: int = 32,
        fixed_shape: Optional[Tuple[int, int]] = None,
        fill_color: int = 114,
    ) -> None:

        self._engine_path = engine_path
        self._device = device

        # Build the inference engine
        self.named_binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        self.engine = self._build_engine()
        self.bindings = OrderedDict()
        self.binding_addrs = None
        self.context = None
        self._set_context()

        if precision == "fp32":
            self._half = False
        elif precision == "fp16":
            self._half = True
        else:
            raise NotImplementedError(f"Currently not supports precision: {precision}")

        self._dtype = torch.float16 if self._half else torch.float32

        # Set pre-processing transform for TensorRT inference
        self._enable_dynamic = enable_dynamic
        self._size = size
        self._size_divisible = size_divisible
        self._fixed_shape = fixed_shape
        self._fill_color = fill_color
        self._img_size = None
        self.transform = None
        self._set_preprocessing()

        # Visualization
        self._names = [f"class{i}" for i in range(1000)]  # assign defaults

    def _build_engine(self):
        logger.info(f"Loading {self._engine_path} for TensorRT inference...")
        if trt is not None:
            trt_logger = trt.Logger(trt.Logger.INFO)
        else:
            trt_logger = None
            raise ImportError("TensorRT is not installed, please install trt firstly.")

        trt.init_libnvinfer_plugins(trt_logger, namespace="")
        with open(self._engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        return engine

    def _set_context(self):
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            shape = tuple(self.engine.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self._device)
            self.bindings[name] = self.named_binding(name, dtype, shape, data, int(data.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.engine.create_execution_context()

    def _set_preprocessing(self):
        if self._enable_dynamic:
            raise NotImplementedError("Currently only supports static shape inference in TensorRT.")

        export_onnx_shape = self.bindings["images"].shape
        self._img_size = export_onnx_shape

        size = export_onnx_shape[-2:]
        self.transform = YOLOTransform(
            size[0],
            size[1],
            size_divisible=self._size_divisible,
            fixed_shape=size,
            fill_color=self._fill_color,
        )

    def warmup(self):
        # Warmup model by running inference once and only warmup GPU models
        if isinstance(self._device, torch.device) and self._device.type != "cpu":
            image = torch.zeros(*self._img_size).to(dtype=self._dtype, device=self._device)
            self(image)

    def __call__(self, image: Tensor):
        """
        Args:
            image (Tensor): an image of shape (N, C, H, W).

        Returns:
            predictions (Tuple[Tensor, Tensor, Tensor, Tensor]):
                stands for boxes, scores, labels and number of boxes respectively.
        """
        self.binding_addrs["images"] = int(image.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        num_dets = self.bindings["num_detections"].data
        boxes = self.bindings["detection_boxes"].data
        scores = self.bindings["detection_scores"].data
        labels = self.bindings["detection_classes"].data
        return boxes, scores, labels, num_dets

    @staticmethod
    def parse_output(all_boxes, all_scores, all_labels, all_num_dets):
        detections: List[Dict[str, Tensor]] = []

        for boxes, scores, labels, num_dets in zip(all_boxes, all_scores, all_labels, all_num_dets):
            keep = num_dets.item()
            boxes, scores, labels = boxes[:keep], scores[:keep], labels[:keep]
            detections.append({"scores": scores, "labels": labels, "boxes": boxes})

        return detections

    def forward(self, inputs: List[Tensor]):
        """
        Wrapper the TensorRT inference engine with Pre-Processing Module.

        Args:
            inputs (list[Tensor]): images to be processed
        """
        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []

        for img in inputs:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # Pre-Processing
        samples, _ = self.transform(inputs)
        # Inference on TensorRT
        boxes, scores, labels, num_dets = self(samples.tensors)
        results = self.parse_output(boxes, scores, labels, num_dets)

        # Rescale coordinate
        im_shape = torch.tensor(samples.tensors.shape[-2:])
        detections = self.transform.postprocess(results, im_shape, original_image_sizes)

        return detections

    def predict(self, x: Any, image_loader: Optional[Callable] = None) -> List[Dict[str, Tensor]]:
        """
        Predict function for raw data or processed data
        Args:
            x: Input to predict. Can be raw data or processed data.
            image_loader: Utility function to convert raw data to Tensor.

        Returns:
            The post-processed model predictions.
        """
        image_loader = image_loader or self.default_loader
        images = self.collate_images(x, image_loader)
        return self.forward(images)

    def default_loader(self, img_path: str) -> Tensor:
        """
        Default loader of read a image path.

        Args:
            img_path (str): a image path

        Returns:
            Tensor, processed tensor for prediction.
        """
        return read_image(img_path) / 255.0

    def collate_images(self, samples: Any, image_loader: Callable) -> List[Tensor]:
        """
        Prepare source samples for inference.

        Args:
            samples (Any): samples source, support the following various types:
                - str or List[str]: a image path or list of image paths.
                - Tensor or List[Tensor]: a tensor or list of tensors.

        Returns:
            List[Tensor], The processed image samples.
        """
        if isinstance(samples, Tensor):
            return [samples.to(dtype=self._dtype, device=self._device)]

        if contains_any_tensor(samples):
            return [sample.to(dtype=self._dtype, device=self._device) for sample in samples]

        if isinstance(samples, str):
            samples = [samples]

        if isinstance(samples, (list, tuple)) and all(isinstance(p, str) for p in samples):
            outputs = []
            for sample in samples:
                output = image_loader(sample).to(dtype=self._dtype, device=self._device)
                outputs.append(output)
            return outputs

        raise NotImplementedError(
            f"The type of the sample is {type(samples)}, we currently don't support it now, the "
            "samples should be either a tensor, list of tensors, a image path or list of image paths."
        )
