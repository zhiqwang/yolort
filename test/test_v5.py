# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from pathlib import Path

import torch
from torch import Tensor
from yolort.v5 import load_yolov5_model


def test_load_yolov5_model():
    img_path = "test/assets/zidane.jpg"

    yolov5s_r40_path = Path("yolov5s.pt")

    if not yolov5s_r40_path.exists():
        torch.hub.download_url_to_file(
            "https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt",
            yolov5s_r40_path,
            hash_prefix="9ca9a642",
        )

    model = load_yolov5_model(str(yolov5s_r40_path), autoshape=True, verbose=False)
    results = model(img_path)

    assert isinstance(results.pred, list)
    assert len(results.pred) == 1
    assert isinstance(results.pred[0], Tensor)
    assert results.pred[0].shape == (3, 6)
