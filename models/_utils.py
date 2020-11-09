from torchvision.ops import box_convert


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """
    def decode_single(self, rel_codes, boxes, wh_weights, xy_weights):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        rel_codes[..., 0:2] = (rel_codes[..., 0:2] * 2. + boxes) * wh_weights  # wh
        rel_codes[..., 2:4] = (rel_codes[..., 2:4] * 2) ** 2 * xy_weights  # xy
        pred_boxes = box_convert(rel_codes, in_fmt="cxcywh", out_fmt="xyxy")

        return pred_boxes
