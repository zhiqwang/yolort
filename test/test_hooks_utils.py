import pytest
import torch

from yolort.utils import FeatureExtractor
from yolort.models import yolov5s


@pytest.mark.parametrize('b, h, w', [(8, 640, 640), (4, 416, 320), (8, 320, 416)])
def test_feature_extractor(b, h, w):
    c = 3
    in_channels = [128, 256, 512]
    strides = [8, 16, 32]
    num_outputs = 85
    expected_features = [(b, inc, h // s, w // s) for inc, s in zip(in_channels, strides)]
    expected_head_outputs = [(b, c, h // s, w // s, num_outputs) for s in strides]

    model = yolov5s()
    model = model.train()
    yolo_features = FeatureExtractor(model.model, return_layers=['backbone', 'head'])
    images = torch.randn(b, c, h, w)
    targets = torch.randn(61, 6)
    intermediate_features = yolo_features(images, targets)
    features = intermediate_features['backbone']
    head_outputs = intermediate_features['head']
    assert isinstance(features, list)
    assert [f.shape for f in features] == expected_features
    assert isinstance(head_outputs, list)
    assert [h.shape for h in head_outputs] == expected_head_outputs


if __name__ == '__main__':
    pytest.main([__file__])
