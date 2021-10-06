import cv2
import numpy as np

try:
    from pyppl import common as pplcommon, nn as pplnn
except ImportError:
    pplcommon, pplnn = None, None


class PredictorPPL:
    """
    Create a simple end-to-end predictor with the given checkpoint that runs on
    single device for a single input image.

    Attributes:
        checkpoint_path (str): Path of the ONNX checkpoint.
        engine_type (str, default is x86): which engine to use x86 or cuda.

    Examples:
        >>> from yolort.runtime import PredictorPPL
        >>>
        >>> checkpoint_path = 'yolov5s.sim.onnx'
        >>> detector = PredictorPPL(checkpoint_path)
        >>>
        >>> img_path = 'bus.jpg'
        >>> scores, class_ids, boxes = detector.run_on_image(img_path)
    """

    def __init__(self, checkpoint_path: str, engine_type: str = "x86"):
        providers = self._set_providers(engine_type)
        self._runtime = self._build_runtime(checkpoint_path, providers)

    def _set_providers(self, engine_type):
        providers = []
        if engine_type == "x86":
            engine = self._build_x86_engine()
        elif engine_type == "cuda":
            engine = self._build_cuda_engine()
        else:
            raise NotImplementedError(f"Not supported this engine type: {engine_type}")
        providers.append(engine)
        return providers

    def _build_x86_engine(self):
        x86_options = pplnn.X86EngineOptions()
        x86_engine = pplnn.X86EngineFactory.Create(x86_options)
        return pplnn.Engine(x86_engine)

    def _build_cuda_engine(self):
        cuda_options = pplnn.CudaEngineOptions()
        cuda_options.device_id = 0
        cuda_engine = pplnn.CudaEngineFactory.Create(cuda_options)
        return pplnn.Engine(cuda_engine)

    @staticmethod
    def _build_runtime(checkpoint_path, providers):
        runtime_builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(checkpoint_path, providers)
        if not runtime_builder:
            raise RuntimeError("Create RuntimeBuilder failed.")

        runtime = runtime_builder.CreateRuntime()
        if not runtime:
            raise RuntimeError("Create Runtime instance failed.")

        return runtime

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
        tensor = self._runtime.GetInputTensor(0)
        status = tensor.ConvertFromHost(image)
        if status != pplcommon.RC_SUCCESS:
            raise RuntimeError(
                f"Copy data to tensor[{tensor.GetName()}] failed: {pplcommon.GetRetCodeStr(status)}"
            )

        status = self._runtime.Run()
        if status != pplcommon.RC_SUCCESS:
            raise RuntimeError(f"Run() failed: {pplcommon.GetRetCodeStr(status)}")

        for i in range(self._runtime.GetOutputCount()):
            tensor = self._runtime.GetOutputTensor(i)
            blob = tensor.ConvertToHost()
            if not blob:
                raise RuntimeError(f"Copy data from tensor[{tensor.GetName()}] failed.")

            if tensor.GetName() == "boxes":
                boxes = np.array(blob, copy=False)
                boxes = boxes.squeeze()
            if tensor.GetName() == "labels":
                class_ids = np.array(blob, copy=False)
                class_ids = class_ids.squeeze()
            if tensor.GetName() == "scores":
                scores = np.array(blob, copy=False)

        return boxes, class_ids, scores

    def run_on_image(self, image_path):
        """
        Run the PPL model for one image only.

        Args:
            image_path: input data file (binary data)
        """
        image = cv2.imread(image_path)
        return self(image)
