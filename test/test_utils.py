# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from pathlib import Path
import numpy as np
import torch
from torch import nn, Tensor

from yolort.utils import (
    update_module_state_from_ultralytics,
    read_image_to_tensor,
    get_image_from_url,
)


def test_update_module_state_from_ultralytics():
    yolov5s_r40_path = Path('yolov5s.pt')

    if not yolov5s_r40_path.is_file():
        yolov5s_r40_url = 'https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt'
        torch.hub.download_url_to_file(yolov5s_r40_url, yolov5s_r40_path, hash_prefix='9ca9a642')

    model = update_module_state_from_ultralytics(
        str(yolov5s_r40_path),
        arch='yolov5s',
        feature_fusion_type='PAN',
        num_classes=80,
    )
    assert isinstance(model, nn.Module)


def test_read_image_to_tensor():
    N, H, W = 3, 720, 360
    img = np.random.randint(0, 255, (H, W, N), dtype='uint8')  # As a dummy image
    out = read_image_to_tensor(img)

    assert isinstance(out, Tensor)
    assert tuple(out.shape) == (N, H, W)


def test_get_image_from_url():
    url = 'https://raw.githubusercontent.com/zhiqwang/yolov5-rt-stack/master/test/assets/zidane.jpg'
    img = get_image_from_url(url)
    assert isinstance(img, np.ndarray)
    assert tuple(img.shape) == (720, 1280, 3)
