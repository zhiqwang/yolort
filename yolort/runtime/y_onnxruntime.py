import logging

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)


class PredictorORT:
    """
    Create a simple end-to-end predictor with the given checkpoint that runs on
    single device for a single input image.

    Attributes:
        checkpoint_path (str): Path of the ONNX checkpoint.
        enable_gpu (bool, default is False): Whether to enable GPU device.

    Examples:
        >>> from yolort.runtime import PredictorORT
        >>>
        >>> checkpoint_path = 'yolov5s.sim.onnx'
        >>> detector = PredictorORT(checkpoint_path)
        >>>
        >>> img_path = 'bus.jpg'
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
        if ort is not None:
            ort_device = ort.get_device()
        else:
            raise ImportError("ONNXRuntime is not installed, please install onnxruntime firstly.")
        providers = None

        if ort_device == "GPU" and self.enable_gpu:
            providers = ["CUDAExecutionProvider"]
            logger.info("Set inference device to GPU")
        elif self.enable_gpu:
            logger.info("GPU is not supported by your ONNXRuntime build. Fallback to CPU.")
        else:
            providers = ["CPUExecutionProvider"]
            logger.info("Set inference device to CPU")

        return providers

    def _build_runtime(self):
        runtime = ort.InferenceSession(self.checkpoint_path, providers=self._providers)
        return runtime

    def _preprocessing(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blob = image.astype(np.float32) / 255.0
        blob = np.transpose(blob, axes=(2, 0, 1))

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
        blob = self._preprocessing(image)
        predictions = self._runtime.run(
            output_names=None,
            input_feed={self._input_names: blob},
        )
        return predictions

    def run_on_image(self, image_path):
        """
        Run the ORT model for one image only.

        Args:
            image_path (str): The image path to be predicted.
        """
        img = cv2.imread(image_path)
        scores, class_ids, boxes = self(img)
        return scores.tolist(), class_ids.tolist(), boxes.tolist()
