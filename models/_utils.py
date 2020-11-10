import torch
from torch import Tensor
from torch.jit.annotations import Tuple
from torchvision.ops import box_convert


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """
    def decode_single(
        self,
        rel_codes: Tensor,
        anchors_tuple: Tuple[Tensor, Tensor, Tensor],
    ):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            anchors_tupe (Tensor, Tensor, Tensor): reference boxes.
        """

        pred_wh = (rel_codes[..., 0:2] * 2. + anchors_tuple[0]) * anchors_tuple[1]  # wh
        pred_xy = (rel_codes[..., 2:4] * 2) ** 2 * anchors_tuple[2]  # xy
        pred_boxes = torch.cat([pred_wh, pred_xy], dim=1)
        pred_boxes = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")

        return pred_boxes
