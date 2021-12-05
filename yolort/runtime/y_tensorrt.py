import logging

import cv2
import numpy as np

try:
    import tensorrt as trt
except ImportError:
    trt = None

logger = logging.getLogger(__name__)


class PredictorTRT:
    """
    Create a simple end-to-end TensorRT predictor with the given checkpoint that
    runs on single device for a single input image.

    Attributes:
        checkpoint_path (str): Path of the ONNX checkpoint.
        enable_gpu (bool, default is False): Whether to enable GPU device.

    Examples:
        >>> from yolort.runtime import PredictorTRT
        >>>
        >>> checkpoint_path = "yolort.onnx"
        >>> detector = PredictorTRT(checkpoint_path)
        >>>
        >>> img_path = "bus.jpg"
        >>> scores, class_ids, boxes = detector.run_on_image(img_path)
    """

    def __init__(self, checkpoint_path: str, enable_gpu: bool = False) -> None:
        self.checkpoint_path = checkpoint_path
        self.enable_gpu = enable_gpu
        self._providers = self._set_providers()
        self._runtime = self._build_runtime()
        self._input_names = self._runtime.get_inputs()[0].name

    def _set_providers(self):
        logger.info("Providers was initialized.")
        pass

    def _build_runtime(self):
        pass

    def _preprocessing(self, image: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, image: np.ndarray):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (Tuple[List[float], List[int], List[float, float]]):
                stands for scores, labels and boxes respectively.
        """
        pass

    def run_on_image(self, image_path):
        """
        Run the ORT model for one image only.

        Args:
            image_path (str): The image path to be predicted.
        """
        img = cv2.imread(image_path)
        scores, class_ids, boxes = self(img)
        return scores.tolist(), class_ids.tolist(), boxes.tolist()
