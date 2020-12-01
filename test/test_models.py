import random
import warnings

import functools
import operator
import unittest

import numpy as np
import torch
from torchvision import models

from common_utils import TestCase, map_nested_tensor_object


def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_available_yolo_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


# If 'unwrapper' is provided it will be called with the script model outputs
# before they are compared to the eager model outputs. This is useful if the
# model outputs are different between TorchScript / Eager mode
script_model_unwrapper = {
    "yolov5s": lambda x: x[1],
}


# The following models exhibit flaky numerics under autocast in _test_*_model harnesses.
# This may be caused by the harness environment (e.g. num classes, input initialization
# via torch.rand), and does not prove autocast is unsuitable when training with real data
# (autocast has been used successfully with real data for some of these models).
# TODO:  investigate why autocast numerics are flaky in the harnesses.
#
# For the following models, _test_*_model harnesses skip numerical checks on outputs when
# trying autocast. However, they still try an autocasted forward pass, so they still ensure
# autocast coverage suffices to prevent dtype errors in each model.
autocast_flaky_numerics = (
    "inception_v3",
    "resnet101",
    "resnet152",
    "wide_resnet101_2",
)


class ModelTester(TestCase):
    def _test_detection_model(self, name, dev):
        set_rng_seed(0)
        kwargs = {}
        if "retinanet" in name:
            # Reduce the default threshold to ensure the returned boxes are not empty.
            kwargs["score_thresh"] = 0.01

        model = models.__dict__[name](num_classes=50, pretrained_backbone=False, **kwargs)
        model.eval().to(device=dev)
        input_shape = (3, 300, 300)
        # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
        x = torch.rand(input_shape).to(device=dev)
        model_input = [x]
        out = model(model_input)
        self.assertIs(model_input[0], x)

        def check_out(out):
            self.assertEqual(len(out), 1)

            def compact(tensor):
                size = tensor.size()
                elements_per_sample = functools.reduce(operator.mul, size[1:], 1)
                if elements_per_sample > 30:
                    return compute_mean_std(tensor)
                else:
                    return subsample_tensor(tensor)

            def subsample_tensor(tensor):
                num_elems = tensor.size(0)
                num_samples = 20
                if num_elems <= num_samples:
                    return tensor

                ith_index = num_elems // num_samples
                return tensor[ith_index - 1::ith_index]

            def compute_mean_std(tensor):
                # can't compute mean of integral tensor
                tensor = tensor.to(torch.double)
                mean = torch.mean(tensor)
                std = torch.std(tensor)
                return {"mean": mean, "std": std}

            output = map_nested_tensor_object(out, tensor_map_fn=compact)
            prec = 0.01
            strip_suffix = f"_{dev}"
            try:
                # We first try to assert the entire output if possible. This is not
                # only the best way to assert results but also handles the cases
                # where we need to create a new expected result.
                self.assertExpected(output, prec=prec, strip_suffix=strip_suffix)
            except AssertionError:
                # Unfortunately detection models are flaky due to the unstable sort
                # in NMS. If matching across all outputs fails, use the same approach
                # as in NMSTester.test_nms_cuda to see if this is caused by duplicate
                # scores.
                expected_file = self._get_expected_file(strip_suffix=strip_suffix)
                expected = torch.load(expected_file)
                self.assertEqual(output[0]["scores"], expected[0]["scores"], prec=prec)

                # Note: Fmassa proposed turning off NMS by adapting the threshold
                # and then using the Hungarian algorithm as in DETR to find the
                # best match between output and expected boxes and eliminate some
                # of the flakiness. Worth exploring.
                return False  # Partial validation performed

            return True  # Full validation performed

        full_validation = check_out(out)
        self.check_jit_scriptable(model, ([x],), unwrapper=script_model_unwrapper.get(name, None))

        if dev == torch.device("cuda"):
            with torch.cuda.amp.autocast():
                out = model(model_input)
                # See autocast_flaky_numerics comment at top of file.
                if name not in autocast_flaky_numerics:
                    full_validation &= check_out(out)

        if not full_validation:
            msg = (f"The output of {self._testMethodName} could only be partially validated. "
                   "This is likely due to unit-test flakiness, but you may "
                   "want to do additional manual checks if you made "
                   "significant changes to the codebase.")
            warnings.warn(msg, RuntimeWarning)
            raise unittest.SkipTest(msg)

    def _test_model_validation(self, name):
        set_rng_seed(0)
        model = models.__dict__[name](num_classes=50, pretrained_backbone=False)
        input_shape = (3, 300, 300)
        x = [torch.rand(input_shape)]

        # validate that targets are present in training
        self.assertRaises(ValueError, model, x)

        # validate type
        targets = [{'boxes': 0.}]
        self.assertRaises(ValueError, model, x, targets=targets)

        # validate boxes shape
        for boxes in (torch.rand((4,)), torch.rand((1, 5))):
            targets = [{'boxes': boxes}]
            self.assertRaises(ValueError, model, x, targets=targets)

        # validate that no degenerate boxes are present
        boxes = torch.tensor([[1, 3, 1, 4], [2, 4, 3, 4]])
        targets = [{'boxes': boxes}]
        self.assertRaises(ValueError, model, x, targets=targets)

    def test_yolov5_double(self):
        model = models.detection.fasterrcnn_resnet50_fpn(num_classes=50, pretrained_backbone=False)
        model.double()
        model.eval()
        input_shape = (3, 300, 300)
        x = torch.rand(input_shape, dtype=torch.float64)
        model_input = [x]
        out = model(model_input)
        self.assertIs(model_input[0], x)
        self.assertEqual(len(out), 1)
        self.assertTrue("boxes" in out[0])
        self.assertTrue("scores" in out[0])
        self.assertTrue("labels" in out[0])

    @unittest.skipIf(not torch.cuda.is_available(), 'needs GPU')
    def test_fasterrcnn_switch_devices(self):
        def checkOut(out):
            self.assertEqual(len(out), 1)
            self.assertTrue("boxes" in out[0])
            self.assertTrue("scores" in out[0])
            self.assertTrue("labels" in out[0])

        model = models.detection.fasterrcnn_resnet50_fpn(num_classes=50, pretrained_backbone=False)
        model.cuda()
        model.eval()
        input_shape = (3, 300, 300)
        x = torch.rand(input_shape, device=torch.device("cuda"))
        model_input = [x]
        out = model(model_input)
        self.assertIs(model_input[0], x)

        checkOut(out)

        with torch.cuda.amp.autocast():
            out = model(model_input)

        checkOut(out)

        # now switch to cpu and make sure it works
        model.cpu()
        x = x.cpu()
        out_cpu = model([x])

        checkOut(out_cpu)


_devs = [torch.device("cpu"), torch.device("cuda")] if torch.cuda.is_available() else [torch.device("cpu")]


for model_name in get_available_yolo_models():
    for dev in _devs:
        # for-loop bodies don't define scopes, so we have to save the variables
        # we want to close over in some way
        def do_test(self, model_name=model_name, dev=dev):
            self._test_detection_model(model_name, dev)

        setattr(ModelTester, f"test_{model_name}_{dev}", do_test)

    def do_validation_test(self, model_name=model_name):
        self._test_detection_model_validation(model_name)

    setattr(ModelTester, "test_" + model_name + "_validation", do_validation_test)


if __name__ == '__main__':
    unittest.main()
