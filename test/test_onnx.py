# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
"""
Test for exporting model to ONNX and inference with ONNXRuntime
"""
import io
import unittest

try:
    # This import should be before that of torch if you are using PyTorch lower than 1.5.0
    # see <https://github.com/onnx/onnx/issues/2394#issuecomment-581638840>
    import onnxruntime
except ImportError:
    onnxruntime = None

import torch
from torchvision.ops._register_onnx_ops import _onnx_opset_version

from yolort.models import yolov5s, yolov5m, yolotr
from yolort.utils import get_image_from_url, read_image_to_tensor


@unittest.skipIf(onnxruntime is None, 'ONNX Runtime unavailable')
class ONNXExporterTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)

    def run_model(self, model, inputs_list, tolerate_small_mismatch=False,
                  do_constant_folding=True, dynamic_axes=None,
                  output_names=None, input_names=None):
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
                if isinstance(test_inputs, torch.Tensor) or isinstance(test_inputs, list):
                    test_inputs = (test_inputs,)
                test_ouputs = model(*test_inputs)
                if isinstance(test_ouputs, torch.Tensor):
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
                torch.testing.assert_allclose(outputs[i], ort_outs[i], rtol=1e-03, atol=1e-05)
            except AssertionError as error:
                if tolerate_small_mismatch:
                    self.assertIn("(0.00%)", str(error), str(error))
                else:
                    raise

    def get_test_images(self):
        image_url = "https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg"
        image = get_image_from_url(image_url)
        image = read_image_to_tensor(image, is_half=False)

        image_url2 = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"
        image2 = get_image_from_url(image_url2)
        image2 = read_image_to_tensor(image2, is_half=False)

        images_one = [image]
        images_two = [image2]
        return images_one, images_two

    def test_yolov5s_r31(self):
        images_one, images_two = self.get_test_images()
        images_dummy = [torch.ones(3, 100, 100) * 0.3]
        model = yolov5s(upstream_version='r3.1', export_friendly=True, pretrained=True, score_thresh=0.45)
        model.eval()
        model(images_one)
        # Test exported model on images of different size, or dummy input
        self.run_model(model, [(images_one,), (images_two,), (images_dummy,)], input_names=["images_tensors"],
                       output_names=["outputs"],
                       dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
                       tolerate_small_mismatch=True)
        # Test exported model for an image with no detections on other images
        self.run_model(model, [(images_dummy,), (images_one,)], input_names=["images_tensors"],
                       output_names=["outputs"],
                       dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
                       tolerate_small_mismatch=True)

    def test_yolov5m_r40(self):
        images_one, images_two = self.get_test_images()
        images_dummy = [torch.ones(3, 100, 100) * 0.3]
        model = yolov5m(upstream_version='r4.0', export_friendly=True, pretrained=True, score_thresh=0.45)
        model.eval()
        model(images_one)
        # Test exported model on images of different size, or dummy input
        self.run_model(model, [(images_one,), (images_two,), (images_dummy,)], input_names=["images_tensors"],
                       output_names=["outputs"],
                       dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
                       tolerate_small_mismatch=True)
        # Test exported model for an image with no detections on other images
        self.run_model(model, [(images_dummy,), (images_one,)], input_names=["images_tensors"],
                       output_names=["outputs"],
                       dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
                       tolerate_small_mismatch=True)

    def test_yolotr(self):
        images_one, images_two = self.get_test_images()
        images_dummy = [torch.ones(3, 100, 100) * 0.3]
        model = yolotr(upstream_version='r4.0', export_friendly=True, pretrained=True, score_thresh=0.45)
        model.eval()
        model(images_one)
        # Test exported model on images of different size, or dummy input
        self.run_model(model, [(images_one,), (images_two,), (images_dummy,)], input_names=["images_tensors"],
                       output_names=["outputs"],
                       dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
                       tolerate_small_mismatch=True)
        # Test exported model for an image with no detections on other images
        self.run_model(model, [(images_dummy,), (images_one,)], input_names=["images_tensors"],
                       output_names=["outputs"],
                       dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
                       tolerate_small_mismatch=True)
