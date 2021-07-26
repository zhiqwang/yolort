# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import torch
from torch import nn, Tensor

from torchvision.models.utils import load_state_dict_from_url

from yolort.models.backbone_utils import darknet_pan_backbone
from yolort.models.anchor_utils import AnchorGenerator
from yolort.models.box_head import YOLOHead

from typing import Any, List, Optional


def yolov5_deploy_friendly(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
):
    r"""yolov5 small release 4.0 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = 'darknet_s_r4_0'
    depth_multiple = 0.33
    width_multiple = 0.5
    version = 'r4.0'
    backbone = darknet_pan_backbone(backbone_name, depth_multiple, width_multiple, version=version)

    model = YOLODeployFriendly(backbone, num_classes, **kwargs)

    if pretrained:
        model_urls_root = 'https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0'
        model_url = f'{model_urls_root}/yolov5_darknet_pan_s_r40_coco-e3fd213d.pt'
        state_dict = load_state_dict_from_url(model_url, progress=progress)
        model.load_state_dict(state_dict)

    return model


class YOLODeployFriendly(nn.Module):
    """
    Deployment Friendly Wrapper of YOLO.
    """
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        # Anchor parameters
        anchor_grids: Optional[List[List[float]]] = None,
        anchor_generator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ):
        super().__init__()
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")
        self.backbone = backbone

        strides = [8, 16, 32]

        if anchor_grids is None:
            anchor_grids = [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ]

        if anchor_generator is None:
            anchor_generator = AnchorGenerator(strides, anchor_grids)
        self.anchor_generator = anchor_generator

        if head is None:
            head = YOLOHead(
                backbone.out_channels,
                anchor_generator.num_anchors,
                anchor_generator.strides,
                num_classes,
            )
        self.head = head

    def forward(self, samples: Tensor):
        """
        Arguments:
            samples (Tensor): batched images, of shape [batch_size x 3 x H x W]
        """
        # get the features from the backbone
        features = self.backbone(samples)

        # compute the yolo heads outputs using the features
        head_outputs = self.head(features)

        all_pred_logits = []
        batch_size, _, _, _, K = head_outputs[0].shape

        for pred_logits in head_outputs:
            pred_logits = pred_logits.reshape(batch_size, -1, K)  # Size=(NN, HWA, K)
            all_pred_logits.append(pred_logits)

        all_pred_logits = torch.cat(all_pred_logits, dim=1)
        return all_pred_logits
