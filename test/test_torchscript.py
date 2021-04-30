# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import unittest

import torch

from yolort.models import yolov5s, yolov5m, yolov5l, yolotr


class TorchScriptTester(unittest.TestCase):
    def test_yolov5s_script(self):
        model = yolov5s(pretrained=True)
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.eval()

        x = [torch.rand(3, 416, 320), torch.rand(3, 480, 352)]

        out = model(x)
        out_script = scripted_model(x)
        self.assertTrue(out[0]["scores"].equal(out_script[1][0]["scores"]))
        self.assertTrue(out[0]["labels"].equal(out_script[1][0]["labels"]))
        self.assertTrue(out[0]["boxes"].equal(out_script[1][0]["boxes"]))

    def test_yolov5m_script(self):
        model = yolov5m(pretrained=True)
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.eval()

        x = [torch.rand(3, 416, 320), torch.rand(3, 480, 352)]

        out = model(x)
        out_script = scripted_model(x)
        self.assertTrue(out[0]["scores"].equal(out_script[1][0]["scores"]))
        self.assertTrue(out[0]["labels"].equal(out_script[1][0]["labels"]))
        self.assertTrue(out[0]["boxes"].equal(out_script[1][0]["boxes"]))

    def test_yolov5l_script(self):
        model = yolov5l(pretrained=True)
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.eval()

        x = [torch.rand(3, 416, 320), torch.rand(3, 480, 352)]

        out = model(x)
        out_script = scripted_model(x)
        self.assertTrue(out[0]["scores"].equal(out_script[1][0]["scores"]))
        self.assertTrue(out[0]["labels"].equal(out_script[1][0]["labels"]))
        self.assertTrue(out[0]["boxes"].equal(out_script[1][0]["boxes"]))

    def test_yolotr_script(self):
        model = yolotr(pretrained=True)
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.eval()

        x = [torch.rand(3, 416, 320), torch.rand(3, 480, 352)]

        out = model(x)
        out_script = scripted_model(x)
        self.assertTrue(out[0]["scores"].equal(out_script[1][0]["scores"]))
        self.assertTrue(out[0]["labels"].equal(out_script[1][0]["labels"]))
        self.assertTrue(out[0]["boxes"].equal(out_script[1][0]["boxes"]))
