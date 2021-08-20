# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import torch

from yolort.models import yolov5s, yolov5m, yolov5l, yolotr


class TestTorchScript:
    def test_yolov5s_script(self):
        model = yolov5s(pretrained=True, size=(640, 640), score_thresh=0.45)
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.eval()

        x = [torch.rand(3, 416, 320), torch.rand(3, 480, 352)]

        out = model(x)
        out_script = scripted_model(x)

        torch.testing.assert_allclose(out[0]["scores"], out_script[1][0]["scores"], rtol=0., atol=0.)
        torch.testing.assert_allclose(out[0]["labels"], out_script[1][0]["labels"], rtol=0., atol=0.)
        torch.testing.assert_allclose(out[0]["boxes"], out_script[1][0]["boxes"], rtol=0., atol=0.)

    def test_yolov5m_script(self):
        model = yolov5m(pretrained=True, size=(640, 640), score_thresh=0.45)
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.eval()

        x = [torch.rand(3, 416, 320), torch.rand(3, 480, 352)]

        out = model(x)
        out_script = scripted_model(x)
        torch.testing.assert_allclose(out[0]["scores"], out_script[1][0]["scores"], rtol=0., atol=0.)
        torch.testing.assert_allclose(out[0]["labels"], out_script[1][0]["labels"], rtol=0., atol=0.)
        torch.testing.assert_allclose(out[0]["boxes"], out_script[1][0]["boxes"], rtol=0., atol=0.)

    def test_yolov5l_script(self):
        model = yolov5l(pretrained=True, size=(640, 640), score_thresh=0.45)
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.eval()

        x = [torch.rand(3, 416, 320), torch.rand(3, 480, 352)]

        out = model(x)
        out_script = scripted_model(x)
        torch.testing.assert_allclose(out[0]["scores"], out_script[1][0]["scores"], rtol=0., atol=0.)
        torch.testing.assert_allclose(out[0]["labels"], out_script[1][0]["labels"], rtol=0., atol=0.)
        torch.testing.assert_allclose(out[0]["boxes"], out_script[1][0]["boxes"], rtol=0., atol=0.)

    def test_yolotr_script(self):
        model = yolotr(pretrained=True, size=(640, 640), score_thresh=0.45)
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.eval()

        x = [torch.rand(3, 416, 320), torch.rand(3, 480, 352)]

        out = model(x)
        out_script = scripted_model(x)
        torch.testing.assert_allclose(out[0]["scores"], out_script[1][0]["scores"], rtol=0., atol=0.)
        torch.testing.assert_allclose(out[0]["labels"], out_script[1][0]["labels"], rtol=0., atol=0.)
        torch.testing.assert_allclose(out[0]["boxes"], out_script[1][0]["boxes"], rtol=0., atol=0.)
