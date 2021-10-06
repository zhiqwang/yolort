import numpy as np

try:
    from pyppl import nn as pplnn, common as pplcommon
except ImportError:
    pplnn, pplcommon = None, None


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
        self._engines = self._set_providers(engine_type)
        self._build_runtime(checkpoint_path)

    def _set_providers(self, engine_type):
        engines = []
        if engine_type == "x86":
            engine = self._build_x86_engine()
        elif engine_type == "cuda":
            engine = self._build_cuda_engine()
        else:
            raise NotImplementedError(f"Not supported this engine type: {engine_type}")
        engines.append(engine)

    def _build_x86_engine(self):
        x86_options = pplnn.X86EngineOptions()
        x86_engine = pplnn.X86EngineFactory.Create(x86_options)
        return pplnn.Engine(x86_engine)

    def _build_cuda_engine(self):
        cuda_options = pplnn.CudaEngineOptions()
        cuda_options.device_id = 0
        cuda_engine = pplnn.CudaEngineFactory.Create(cuda_options)
        return pplnn.Engine(cuda_engine)

    def _build_runtime(self, checkpoint_path):
        runtime_builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(
            checkpoint_path, self._engines,
        )
        if not runtime_builder:
            raise RuntimeError("Create RuntimeBuilder failed.")

        self._runtime = runtime_builder.CreateRuntime()
        if not self._runtime:
            raise RuntimeError("Create Runtime instance failed.")

    def _preprocessing(self, image_path):
        tensor = self._runtime.GetInputTensor(0)
        in_data = np.fromfile(image_path, dtype=np.float32).reshape((1, 3, 800, 1200))
        status = tensor.ConvertFromHost(in_data)
        if status != pplcommon.RC_SUCCESS:
            raise RuntimeError(
                f"Copy data to tensor[{tensor.GetName()}] failed: {pplcommon.GetRetCodeStr(status)}"
            )

    def _prepare_output(self):
        for i in range(self._runtime.GetOutputCount()):
            tensor = self._runtime.GetOutputTensor(i)
            tensor_data = tensor.ConvertToHost()
            if not tensor_data:
                raise RuntimeError(
                    f"Copy data from tensor[{tensor.GetName()}] failed."
                )
            if tensor.GetName() == 'dets':
                dets_data = np.array(tensor_data, copy=False)
                dets_data = dets_data.squeeze()
            if tensor.GetName() == 'labels':
                labels_data = np.array(tensor_data, copy=False)
                labels_data = labels_data.squeeze()
            if tensor.GetName() == 'masks':
                masks_data = np.array(tensor_data, copy=False)
                masks_data = masks_data.squeeze()
        return dets_data, labels_data, masks_data

    def run_on_image(self, image_path):
        """
        Run the PPL model for one image only.

        Args:
            image_path: input data file (binary data)
        """
        self._preprocessing(image_path)
        status = self._runtime.Run()
        if status != pplcommon.RC_SUCCESS:
            raise RuntimeError(f"Run() failed: {pplcommon.GetRetCodeStr(status)}")
        dets_data, labels_data, masks_data = self._prepare_output()
        return dets_data, labels_data, masks_data
