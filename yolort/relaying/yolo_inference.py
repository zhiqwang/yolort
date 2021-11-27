# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import torch
from torch import nn, Tensor
from yolort.models import YOLO
from yolort.models.box_head import LogitsDecoder

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
        post_process = LogitsDecoder(score_thresh)

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
