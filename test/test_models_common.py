
import torch
from yolort.models.common import focus_transform, space_to_depth

def test_space_to_depth():
    tensor_input = torch.randn((1,3,480,640)) 
    out1 = focus_transform(tensor_input)
    out2 = space_to_depth(tensor_input)
    torch.testing.assert_allclose(out1, out2)