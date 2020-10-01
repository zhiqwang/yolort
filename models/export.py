from torch import nn, Tensor

from utils.activations import Hardswish

from .common import Conv


class WrappedYOLO(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Update model
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
            if isinstance(m, Conv) and isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()  # assign activation

        self.model = model

    def forward(self, inputs: Tensor):
        return self.model(inputs)
