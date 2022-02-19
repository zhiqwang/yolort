"""
Test for exporting model to ONNX and inference with ONNX Runtime
"""
import io
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torchvision.io import read_image
from torchvision.ops._register_onnx_ops import _onnx_opset_version
from yolort import models
from yolort.runtime import PredictorORT
from yolort.runtime.ort_helper import export_onnx
from yolort.utils.image_utils import to_numpy

# In environments without onnxruntime we prefer to
# invoke all tests in the repo and have this one skipped rather than fail.
onnxruntime = pytest.importorskip("onnxruntime")


class TestONNXExporter:
    def run_model(self, model, inputs_list):
        """
        The core part of exporting model to ONNX and inference with ONNX Runtime
        Copy-paste from <https://github.com/pytorch/vision/blob/07fb8ba/test/test_onnx.py#L34>
        """
        model = model.eval()

        onnx_io = io.BytesIO()

        # export to onnx models
        batch_size = len(inputs_list[0])
        export_onnx(onnx_io, model=model, opset_version=_onnx_opset_version, batch_size=batch_size)

        # validate the exported model with onnx runtime
        for test_inputs in inputs_list:
            with torch.no_grad():
                if isinstance(test_inputs, Tensor) or isinstance(test_inputs, list):
                    test_inputs = (test_inputs,)
                test_outputs = model(*test_inputs)
                if isinstance(test_outputs, Tensor):
                    test_outputs = (test_outputs,)
            self.ort_validate(onnx_io, test_inputs, test_outputs)

    def ort_validate(self, onnx_io, inputs, outputs):

        inputs, _ = torch.jit._flatten(inputs)
        outputs, _ = torch.jit._flatten(outputs)

        inputs = list(map(to_numpy, inputs))
        outputs = list(map(to_numpy, outputs))

        y_runtime = PredictorORT(onnx_io.getvalue())
        # Inference on ONNX Runtime
        ort_outs = y_runtime.predict(inputs)

        for i in range(0, len(outputs)):
            torch.testing.assert_allclose(outputs[i], ort_outs[i], rtol=1e-03, atol=1e-05)

    def get_image(self, img_name):

        img_path = Path(__file__).parent.resolve() / "assets" / img_name
        image = read_image(str(img_path)) / 255

        return image

    def get_test_images(self):
        return self.get_image("bus.jpg"), self.get_image("zidane.jpg")

    @pytest.mark.parametrize(
        "arch, fixed_size, upstream_version",
        [
            ("yolov5s", True, "r3.1"),
            ("yolov5m", False, "r4.0"),
            ("yolov5n", True, "r6.0"),
            ("yolov5n", False, "r6.0"),
            ("yolov5n6", False, "r6.0"),
        ],
    )
    def test_onnx_export_single_image(self, arch, fixed_size, upstream_version):
        img_one, img_two = self.get_test_images()
        img_dummy = torch.ones(3, 1080, 720) * 0.3

        size = (640, 640) if arch[-1] == "6" else (320, 320)
        model = models.__dict__[arch](
            upstream_version=upstream_version,
            pretrained=True,
            size=size,
            fixed_shape=size if fixed_size else None,
            score_thresh=0.45,
        )
        model = model.eval()
        model([img_one])
        # Test exported model on images of different size, or dummy input
        self.run_model(model, [[img_one], [img_two], [img_dummy]])

    @pytest.mark.parametrize("arch", ["yolov5n6"])
    def test_onnx_export_multi_batches(self, arch):
        img_one, img_two = self.get_test_images()
        img_dummy = torch.ones(3, 1080, 720) * 0.3

        size = (640, 640) if arch[-1] == "6" else (320, 320)
        model = models.__dict__[arch](pretrained=True, size=size, score_thresh=0.45)
        model = model.eval()
        model([img_one, img_two])

        # Test exported model on images of different size, or dummy input
        inputs_list = [
            [img_one, img_two],
            [img_two, img_one],
            [img_dummy, img_one],
            [img_one, img_one],
            [img_two, img_dummy],
            [img_dummy, img_two],
        ]
        self.run_model(model, inputs_list)

    @pytest.mark.parametrize("arch", ["yolov5n"])
    def test_onnx_export_misbatch(self, arch):
        img_one, img_two = self.get_test_images()
        img_dummy = torch.ones(3, 640, 480) * 0.3

        size = (640, 640) if arch[-1] == "6" else (320, 320)
        model = models.__dict__[arch](pretrained=True, size=size, score_thresh=0.45)
        model = model.eval()
        model([img_one, img_two])

        # Test exported model on images of misbatch
        with pytest.raises(IndexError, match="list index out of range"):
            self.run_model(model, [[img_one, img_two], [img_two, img_one, img_dummy]])

        # Test exported model on images of misbatch
        with pytest.raises(ValueError, match="Model requires 3 inputs. Input Feed contains 2"):
            self.run_model(model, [[img_two, img_one, img_dummy], [img_one, img_two]])
