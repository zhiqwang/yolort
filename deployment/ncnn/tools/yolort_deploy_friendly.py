# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from typing import Any, List, Optional

from torch import nn, Tensor
from torchvision.models.utils import load_state_dict_from_url
from yolort.models.backbone_utils import darknet_pan_backbone
from yolort.models.yolo import YOLO, model_urls


class YOLODeployFriendly(YOLO):
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
        super().__init__(
            backbone,
            num_classes,
            anchor_grids=anchor_grids,
            anchor_generator=anchor_generator,
            head=head,
        )

    def forward(self, samples: Tensor):
        """
        Arguments:
            samples (Tensor): batched images, of shape [batch_size x 3 x H x W]
        """
        # get the features from the backbone
        features = self.backbone(samples)

        # compute the yolo heads outputs using the features
        head_outputs = self.head(features)
        return head_outputs


def yolov5s_r40_deploy_ncnn(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLODeployFriendly:
    """
    Deployment friendly Wrapper of yolov5s for ncnn.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_s_r4_0"
    weights_name = "yolov5_darknet_pan_s_r40_coco"
    depth_multiple = 0.33
    width_multiple = 0.5
    version = "r4.0"

    backbone = darknet_pan_backbone(backbone_name, depth_multiple, width_multiple, version=version)

    model = YOLODeployFriendly(backbone, num_classes, **kwargs)
    if pretrained:
        if model_urls.get(weights_name, None) is None:
            raise ValueError(f"No checkpoint is available for model {weights_name}")
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        model.load_state_dict(state_dict)

    del model.anchor_generator
    del model.post_process

    return model
