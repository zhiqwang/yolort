try:
    from openvino.inference_engine import IECore
except ImportError:
    IECore = None

import logging
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime
from yolort.relay import YOLOTRTInference

logging.basicConfig(level=logging.INFO)
logging.getLogger("OpenvinoHelper").setLevel(logging.INFO)
logger = logging.getLogger("OpenvinoHelper")


class OpenvinoExport:
    def __init__(self, checkpoint_path, device="cpu"):
        self.checkpoint_path = checkpoint_path
        self.model = YOLOTRTInference(self.checkpoint_path)
        self.device = device
        self.providers = (
            ["CPUExecutionProvider"]
            if self.device == "cpu"
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    def export(self, onnx_path, openvino_path, enable_dynamic=False):
        self.model.eval()
        self.model.to_onnx(onnx_path, enable_dynamic=enable_dynamic)
        cmd = f"mo --input_model {onnx_path} --output_dir {openvino_path}"
        subprocess.check_output(cmd, shell=True)
        logger.info("Finish export openvino")

    def check_outputs(self, onnx_path, openvino_path, input_shape=(1, 3, 640, 640), times=10):
        for _ in range(times):
            test_inputs = {"images": np.random.randn(*input_shape).astype("float32")}
            session = onnxruntime.InferenceSession(onnx_path, providers=self.providers)
            ie = IECore()
            net_ir = ie.read_network(model=(Path(openvino_path) / Path(onnx_path).name).with_suffix(".xml"))
            exec_net_ir = ie.load_network(network=net_ir, device_name=self.device.upper())
            out_ort = session.run(output_names=None, input_feed=test_inputs)
            out_ie = exec_net_ir.infer(inputs=test_inputs)
            assert out_ie["boxes"].shape == out_ort[0].shape
            assert out_ie["scores"].shape == out_ort[1].shape
            # np.testing.assert_allclose(out_ie["boxes"],out_ort[0])
            # np.testing.assert_allclose(out_ie["scores"],out_ort[1])
        logger.info(f"Check {onnx_path} and {openvino_path} run with {input_shape} {times} times ok")
        return


if __name__ == "__main__":
    onnx_path = "../../yolov5s.onnx"
    openvino_path = "../../yolov5s_openvino/"
    OE = OpenvinoExport("../../yolov5s.pt")
    # OE.export("../../yolov5s.onnx", openvino_path)
    OE.check_outputs(onnx_path, openvino_path)
    print("finish")
