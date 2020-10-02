from torch import nn, Tensor

from utils.activations import Hardswish

from .common import Conv


class WrappedYOLO(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        # Update backbone
        for k, m in backbone.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
            if isinstance(m, Conv) and isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()  # assign activation

        self.backbone = backbone

    def forward(self, inputs: Tensor):
        return self.backbone(inputs)
