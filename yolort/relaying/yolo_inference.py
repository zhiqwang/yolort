# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from typing import Dict, List, Tuple

import torch
from torch import nn, Tensor
from yolort.models import YOLO
from yolort.models._utils import decode_single
from yolort.models.box_head import _concat_pred_logits

__all__ = ["YOLOInference"]


class YOLOInference(nn.Module):
    """
    A deployment friendly wrapper of YOLO.

    Remove the ``torchvision::nms`` in this warpper, due to the fact that some third-party
    inference frameworks currently do not support this operator very well.
    """

    def __init__(
        self,
        checkpoint_path: str,
        score_thresh: float = 0.25,
        version: str = "r6.0",
    ):
        super().__init__()
        post_process = PostProcess(score_thresh)

        self.model = YOLO.load_from_yolov5(
            checkpoint_path,
            version=version,
            post_process=post_process,
        )

    @torch.no_grad()
    def forward(self, inputs: Tensor):
        """
        Args:
            inputs (Tensor): batched images, of shape [batch_size x 3 x H x W]
        """
        # Compute the detections
        outputs = self.model(inputs)

        return outputs


class PostProcess(nn.Module):
    """
    This is a simplified version of PostProcess to remove the ``torchvision::nms`` module.

    Args:
        score_thresh (float): Score threshold used for postprocessing the detections.
    """

    def __init__(self, score_thresh: float) -> None:
        super().__init__()
        self.score_thresh = score_thresh

    def forward(
        self,
        head_outputs: List[Tensor],
        anchors_tuple: Tuple[Tensor, Tensor, Tensor],
    ) -> List[Dict[str, Tensor]]:
        """
        Just concat the predict logits, ignore the original ``torchvision::nms`` module
        from original ``yolort.models.box_head.PostProcess``.

        Args:
            head_outputs (List[Tensor]): The predicted locations and class/object confidence,
                shape of the element is (N, A, H, W, K).
            anchors_tuple (Tuple[Tensor, Tensor, Tensor]):
        """
        batch_size = len(head_outputs[0])

        all_pred_logits = _concat_pred_logits(head_outputs)

        detections: List[Dict[str, Tensor]] = []

        for idx in range(batch_size):  # image idx, image inference
            pred_logits = torch.sigmoid(all_pred_logits[idx])

            # Compute conf
            # box_conf x class_conf, w/ shape: num_anchors x num_classes
            scores = pred_logits[:, 5:] * pred_logits[:, 4:5]

            boxes = decode_single(pred_logits[:, :4], anchors_tuple)

            # remove low scoring boxes
            inds, labels = torch.where(scores > self.score_thresh)
            boxes, scores = boxes[inds], scores[inds, labels]

            detections.append({"scores": scores, "labels": labels, "boxes": boxes})

        return detections
