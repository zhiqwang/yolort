# Copyright (c) 2022, yolort team. All rights reserved.

import random

import torch
from torch import nn, Tensor
from torchvision.ops import box_convert

def _decode_pred_logits(pred_logits: Tensor):
    """
    Decode the prediction logit from the PostPrecess.
    """
    # Compute conf
    # box_conf x class_conf, w/ shape: num_anchors x num_classes
    scores = pred_logits[..., 5:] * pred_logits[..., 4:5]
    boxes = box_convert(pred_logits[..., :4], in_fmt="cxcywh", out_fmt="xyxy")

    return boxes, scores


class FakeYOLO(nn.Module):
    """
    Fake YOLO used to export an ONNX models for ONNX Runtime and OpenVINO.
    """

    def __init__(
        self,
        model: nn.Module,
        iou_thresh: float = 0.45,
        score_thresh: float = 0.35,
        detections_per_img: int = 100,
    ):
        super().__init__()

        self.model = model
        self.post_process = FakePostProcess(
            iou_thresh=iou_thresh,
            score_thresh=score_thresh,
            detections_per_img=detections_per_img,
        )

    def forward(self, x):
        x = self.model(x)
        out = self.post_process(x)
        return out


class FakePostProcess(nn.Module):
    """
    Fake PostProcess used to export an ONNX models containing NMS for ONNX Runtime and OpenVINO.

    Args:
        iou_thresh (float, optional): NMS threshold used for postprocessing the detections.
            Default to 0.45
        score_thresh (float, optional): Score threshold used for postprocessing the detections.
            Default to 0.35
        detections_per_img (int, optional): Number of best detections to keep after NMS.
            Default to 100
        export_type (str, optional): Export onnx backend support onnxruntime and openvino
    """

    def __init__(
        self,
        iou_thresh: float = 0.45,
        score_thresh: float = 0.35,
        detections_per_img: int = 100,
        export_type="onnxruntime",
    ):
        super().__init__()
        self.detections_per_img = detections_per_img
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.export_type = export_type
        self.nms_func = NonMaxSupressionOp.apply

    def forward(self, x: Tensor):
        device = x.device
        boxes, scores = _decode_pred_logits(x)
        scores, classes = scores.max(2, keepdim=True)
        scores_t = scores.transpose(1, 2).contiguous()

        # Prepare parameters of NMS for exporting ONNX
        detections_per_img = torch.tensor([self.detections_per_img]).to(device)
        iou_thresh = torch.tensor([self.iou_thresh]).to(device)
        score_thresh = torch.tensor([self.score_thresh]).to(device)
        selected_indices = self.nms_func(boxes, scores_t, detections_per_img, iou_thresh, score_thresh)

        i, k = selected_indices[:, 0], selected_indices[:, 2]
        if self.export_type == "openvino":
            i, k = i[i >= 0], k = k[k >= 0]
        boxes_keep = boxes[i, k, :]
        classes_keep = classes[i, k, :]
        scores_keep = scores[i, k, :]
        i = i.unsqueeze(1)
        i = i.float()
        classes_keep = classes_keep.float()
        out = torch.concat([i, boxes_keep, classes_keep, scores_keep], 1)
        return out


class NonMaxSupressionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, boxes, scores, detections_per_class, iou_thresh, score_thresh):
        """
        Symbolic method to export an NonMaxSupression ONNX models.

        Args:
            boxes (Tensor): An input tensor with shape [num_batches, spatial_dimension, 4].
                have been multiplied original size here.
            scores (Tensor): An input tensor with shape [num_batches, num_classes, spatial_dimension].
                only one class score here.
            detections_per_class (Tensor, optional): Integer representing the maximum number of
                boxes to be selected per batch per class. It is a scalar.
            iou_thresh (Tensor, optional): Float representing the threshold for deciding whether
                boxes overlap too much with respect to IOU. It is scalar. Value range [0, 1].
            score_thresh (Tensor, optional): Float representing the threshold for deciding when to
                remove boxes based on score. It is a scalar.

        Returns:
            Tensor(int64): selected indices from the boxes tensor. [num_selected_indices, 3],
                the selected index format is [batch_index, class_index, box_index].
        """
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0]
        idxs = torch.arange(100, 100 + num_det)
        zeros = torch.zeros((num_det,), dtype=torch.int64)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(g, boxes, scores, detections_per_class, iou_thresh, score_thresh):
        return g.op("NonMaxSuppression", boxes, scores, detections_per_class, iou_thresh, score_thresh)
