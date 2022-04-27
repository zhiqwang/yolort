import pytest
import torch
from yolort.v5 import focus_transform, space_to_depth


@pytest.mark.parametrize("n, b, h, w", [(1, 3, 480, 640), (4, 3, 416, 320), (4, 3, 320, 416)])
def test_space_to_depth(n, b, h, w):
    tensor_input = torch.rand((n, b, h, w))
    out1 = focus_transform(tensor_input)
    out2 = space_to_depth(tensor_input)
    torch.testing.assert_close(out2, out1)
