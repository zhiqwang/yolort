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
from yolort.utils.image_utils import to_numpy

# In environments without onnxruntime we prefer to
# invoke all tests in the repo and have this one skipped rather than fail.
onnxruntime = pytest.importorskip("onnxruntime")


class TestONNXExporter:
    def run_model(
        self,
        model,
        inputs_list,
        do_constant_folding=True,
        input_names=None,
        output_names=None,
        dynamic_axes=None,
    ):
        """
        The core part of exporting model to ONNX and inference with ONNX Runtime
        Copy-paste from <https://github.com/pytorch/vision/blob/07fb8ba/test/test_onnx.py#L34>
        """
        model = model.eval()

        onnx_io = io.BytesIO()
        if isinstance(inputs_list[0][-1], dict):
            torch_onnx_input = inputs_list[0] + ({},)
        else:
            torch_onnx_input = inputs_list[0]
        # export to onnx with the first input
        torch.onnx.export(
            model,
            torch_onnx_input,
            onnx_io,
            do_constant_folding=do_constant_folding,
            opset_version=_onnx_opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
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

        ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
        # Inference on ONNX Runtime
        ort_inputs = dict((ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
        ort_outs = ort_session.run(None, ort_inputs)

        for i in range(0, len(outputs)):
            torch.testing.assert_allclose(outputs[i], ort_outs[i], rtol=1e-03, atol=1e-05)

    def get_image(self, img_name):

        img_path = Path(__file__).parent.resolve() / "assets" / img_name
        image = read_image(str(img_path)) / 255

        return image

    def get_test_images(self):
        return [self.get_image("bus.jpg")], [self.get_image("zidane.jpg")]

    @pytest.mark.parametrize(
        "arch, fixed_size, upstream_version",
        [
            ("yolov5s", False, "r3.1"),
            ("yolov5m", True, "r4.0"),
            ("yolov5m", False, "r4.0"),
            ("yolov5n", True, "r6.0"),
            ("yolov5n", False, "r6.0"),
            ("yolov5n6", True, "r6.0"),
            ("yolov5n6", False, "r6.0"),
        ],
    )
    def test_yolort_onnx_export(self, arch, fixed_size, upstream_version):
        images_one, images_two = self.get_test_images()
        images_dummy = [torch.ones(3, 1080, 720) * 0.3]

        model = models.__dict__[arch](
            upstream_version=upstream_version,
            pretrained=True,
            size=(640, 640),
            fixed_shape=(640, 640) if fixed_size else None,
            score_thresh=0.45,
        )
        model = model.eval()
        model(images_one)
        # Test exported model on images of different size, or dummy input
        self.run_model(
            model,
            [(images_one,), (images_two,), (images_dummy,)],
            input_names=["images"],
            output_names=["scores", "labels", "boxes"],
            dynamic_axes={
                "images": [1, 2],
                "boxes": [0, 1],
                "labels": [0],
                "scores": [0],
            },
        )
        # Test exported model for an image with no detections on other images
        self.run_model(
            model,
            [(images_dummy,), (images_one,)],
            input_names=["images"],
            output_names=["scores", "labels", "boxes"],
            dynamic_axes={
                "images": [1, 2],
                "boxes": [0, 1],
                "labels": [0],
                "scores": [0],
            },
        )
