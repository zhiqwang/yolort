# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import hashlib

from torch import Tensor
from yolort.v5 import load_yolov5_model, attempt_download


def test_attempt_download():

    model_url = "https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt"
    checkpoint_path = attempt_download(model_url, hash_prefix="9ca9a642")
    with open(checkpoint_path, "rb") as f:
        bytes = f.read()  # read entire file as bytes
        readable_hash = hashlib.sha256(bytes).hexdigest()
    assert readable_hash[:8] == "9ca9a642"


def test_load_yolov5_model():
    img_path = "test/assets/zidane.jpg"

    model_url = "https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt"
    checkpoint_path = attempt_download(model_url, hash_prefix="9ca9a642")

    model = load_yolov5_model(checkpoint_path, autoshape=True, verbose=False)
    results = model(img_path)

    assert isinstance(results.pred, list)
    assert len(results.pred) == 1
    assert isinstance(results.pred[0], Tensor)
    assert results.pred[0].shape == (3, 6)
