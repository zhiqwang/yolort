# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import pytest
import torch

from yolort import models


@pytest.mark.parametrize('arch', ['yolov5s', 'yolov5m', 'yolov5l', 'yolotr'])
def test_yolov5s_script(arch):
    model = models.__dict__[arch](pretrained=True, size=(320, 320), score_thresh=0.45)
    model.eval()

    scripted_model = torch.jit.script(model)
    scripted_model.eval()

    x = [torch.rand(3, 288, 320), torch.rand(3, 300, 256)]

    out = model(x)
    out_script = scripted_model(x)

    torch.testing.assert_allclose(out[0]["scores"], out_script[1][0]["scores"], rtol=0., atol=0.)
    torch.testing.assert_allclose(out[0]["labels"], out_script[1][0]["labels"], rtol=0., atol=0.)
    torch.testing.assert_allclose(out[0]["boxes"], out_script[1][0]["boxes"], rtol=0., atol=0.)
