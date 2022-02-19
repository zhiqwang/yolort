# Copyright (c) 2021, yolort team. All rights reserved.

import logging
from typing import Any, Dict, List, Callable, Optional

import numpy as np
from yolort.data import contains_any_tensor

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)


class PredictorORT:
    """
    Create a simple end-to-end predictor with the given checkpoint that runs on
    single device for input images with ONNX Runtime.

    Args:
        engine_path (string): Path of the serialized ONNX model.
        device (string): The device to be used for inferencing.

    Example:

        Demo pipeline for deploying yolort with ONNX Runtime.

        .. code-block:: python

            from yolort.runtime import PredictorORT

            # Load the serialized ONNX model
            engine_path = 'yolov5n6.onnx'
            device = 'cpu'
            y_runtime = PredictorORT(engine_path, device=device)

            # Perform inference on an image file
            predictions = y_runtime.predict('bus.jpg')
    """

    def __init__(self, engine_path: str, device: str = "cpu") -> None:
        self.engine_path = engine_path
        self.device = device
        self._providers = self._set_providers()
        self._runtime = self._build_runtime()
        self._input_names = self._runtime.get_inputs()[0].name

    def _set_providers(self):
        logger.info("Providers was initialized.")
        if ort is not None:
            ort_device = ort.get_device()
        else:
            raise ImportError("ONNXRuntime is not installed, please install onnxruntime firstly.")
        providers = None

        enable_gpu = True if self.device == "cuda" else False
        if ort_device == "GPU" and enable_gpu:
            providers = ["CUDAExecutionProvider"]
            logger.info("Set inference device to GPU")
        elif enable_gpu:
            logger.info("GPU is not supported by your ONNXRuntime build. Fallback to CPU.")
        else:
            providers = ["CPUExecutionProvider"]
            logger.info("Set inference device to CPU")

        return providers

    def _build_runtime(self):
        runtime = ort.InferenceSession(self.engine_path, providers=self._providers)
        return runtime

    def default_loader(self, img_path: str) -> np.ndarray:
        """
        Default loader of read a image path.

        Args:
            img_path (str): the path to the image

        Returns:
            np.ndarray, processed ndarray for prediction.
        """
        import cv2

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, axes=(2, 0, 1))

        return image

    def __call__(self, inputs: List[np.ndarray]):
        """
        Args:
            inputs (List[np.ndarray]): a list of images with shape (C, H, W) (in RGB order).

        Returns:
            predictions (Tuple[List[float], List[int], List[float, float]]):
                stands for scores, labels and boxes respectively.
        """
        inputs = dict((self._runtime.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
        predictions = self._runtime.run(output_names=None, input_feed=inputs)
        return predictions

    def predict(self, x: Any, image_loader: Optional[Callable] = None) -> List[Dict[str, np.ndarray]]:
        """
        Predict function for raw data or processed data

        Args:
            x: Input to predict. Can be raw data or processed data.
            image_loader: Utility function to convert raw data to Numpy ndarray.

        Returns:
            The post-processed model predictions.
        """
        image_loader = image_loader or self.default_loader
        images = self.collate_images(x, image_loader)
        return self(images)

    def collate_images(self, samples: Any, image_loader: Callable) -> List[np.ndarray]:
        """
        Prepare source samples for inference.

        Args:
            samples (Any): samples source, support the following various types:
                - str or List[str]: a image path or list of image paths.
                - np.ndarray or List[np.ndarray]: a ndarray or list of ndarray.

        Returns:
            List[np.ndarray], The processed image samples.
        """
        if isinstance(samples, np.ndarray):
            return [samples]

        if contains_any_tensor(samples, dtype=np.ndarray):
            return [sample for sample in samples]

        if isinstance(samples, str):
            samples = [samples]

        if isinstance(samples, (list, tuple)) and all(isinstance(p, str) for p in samples):
            outputs = []
            for sample in samples:
                output = image_loader(sample)
                outputs.append(output)
            return outputs

        raise NotImplementedError(
            f"The type of the sample is {type(samples)}, we currently don't support it now, the "
            "samples should be either a ndarray, list of np.ndarray, a image path or list of image paths."
        )
