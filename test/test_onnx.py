"""
Test for exporting model to ONNX and inference with ONNXRuntime
"""
import io
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.ops._register_onnx_ops import _onnx_opset_version
from yolort import models

# In environments without onnxruntime we prefer to
# invoke all tests in the repo and have this one skipped rather than fail.
onnxruntime = pytest.importorskip("onnxruntime")


class TestONNXExporter:
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)

    def run_model(
        self,
        model,
        inputs_list,
        tolerate_small_mismatch=False,
        do_constant_folding=True,
        dynamic_axes=None,
        output_names=None,
        input_names=None,
    ):
        """
        The core part of exporting model to ONNX and inference with ONNXRuntime
        Copy-paste from <https://github.com/pytorch/vision/blob/07fb8ba/test/test_onnx.py#L34>
        """
        model.eval()

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
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            output_names=output_names,
        )
        # validate the exported model with onnx runtime
        for test_inputs in inputs_list:
            with torch.no_grad():
                if isinstance(test_inputs, Tensor) or isinstance(test_inputs, list):
                    test_inputs = (test_inputs,)
                test_ouputs = model(*test_inputs)
                if isinstance(test_ouputs, Tensor):
                    test_ouputs = (test_ouputs,)
            self.ort_validate(onnx_io, test_inputs, test_ouputs, tolerate_small_mismatch)

    def ort_validate(self, onnx_io, inputs, outputs, tolerate_small_mismatch=False):

        inputs, _ = torch.jit._flatten(inputs)
        outputs, _ = torch.jit._flatten(outputs)

        def to_numpy(tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.cpu().numpy()

        inputs = list(map(to_numpy, inputs))
        outputs = list(map(to_numpy, outputs))

        ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
        # compute onnxruntime output prediction
        ort_inputs = dict((ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
        ort_outs = ort_session.run(None, ort_inputs)

        for i in range(0, len(outputs)):
            try:
                torch.testing.assert_close(outputs[i], ort_outs[i], rtol=1e-03, atol=1e-05)
            except AssertionError as error:
                if tolerate_small_mismatch:
                    self.assertIn("(0.00%)", str(error), str(error))
                else:
                    raise

    def get_image(self, img_name, size):

        img_path = Path(__file__).parent.resolve() / "assets" / img_name
        image = Image.open(img_path).convert("RGB").resize(size, Image.BILINEAR)

        return transforms.ToTensor()(image)

    def get_test_images(self):
        return (
            [self.get_image("bus.jpg", (416, 320))],
            [self.get_image("zidane.jpg", (352, 480))],
        )

    @pytest.mark.parametrize(
        "arch, upstream_version",
        [
            ("yolov5s", "r3.1"),
            ("yolov5m", "r4.0"),
            # ("yolov5ts", "r4.0"),
        ],
    )
    def test_yolort_export_onnx(self, arch, upstream_version):
        images_one, images_two = self.get_test_images()
        images_dummy = [torch.ones(3, 100, 100) * 0.3]

        model = models.__dict__[arch](
            upstream_version=upstream_version,
            export_friendly=True,
            pretrained=True,
            size=(640, 640),
            score_thresh=0.45,
        )
        model.eval()
        model(images_one)
        # Test exported model on images of different size, or dummy input
        self.run_model(
            model,
            [(images_one,), (images_two,), (images_dummy,)],
            input_names=["images_tensors"],
            output_names=["outputs"],
            dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
            tolerate_small_mismatch=True,
        )
        # Test exported model for an image with no detections on other images
        self.run_model(
            model,
            [(images_dummy,), (images_one,)],
            input_names=["images_tensors"],
            output_names=["outputs"],
            dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
            tolerate_small_mismatch=True,
        )
