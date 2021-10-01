# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.

import warnings
from typing import Callable, Tuple, Any, List, Dict, Optional

import torch
from torch import nn, Tensor
from torchvision.models.utils import load_state_dict_from_url

from .anchor_utils import AnchorGenerator
from .backbone_utils import darknet_pan_backbone
from .box_head import YOLOHead, SetCriterion, PostProcess
from .transformer import darknet_tan_backbone

__all__ = [
    "YOLO",
    "yolov5_darknet_pan_s_r31",
    "yolov5_darknet_pan_m_r31",
    "yolov5_darknet_pan_l_r31",
    "yolov5_darknet_pan_s_r40",
    "yolov5_darknet_pan_m_r40",
    "yolov5_darknet_pan_l_r40",
    "yolov5_darknet_tan_s_r40",
    "_yolov5_darknet_pan",
]


class YOLO(nn.Module):
    """
    Implements YOLO series model.

    The input to the model is expected to be a batched tensors, of shape ``[N, C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values
          between ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values
          between ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
    """

    __annotations__ = {
        "compute_loss": SetCriterion,
    }

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        # Anchor parameters
        anchor_grids: Optional[List[List[float]]] = None,
        anchor_generator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        # Training parameter
        criterion: Optional[Callable[..., Dict[str, Tensor]]] = None,
        # Post Process parameter
        score_thresh: float = 0.005,
        nms_thresh: float = 0.45,
        detections_per_img: int = 300,
        post_process: Optional[nn.Module] = None,
    ):
        super().__init__()
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        self.backbone = backbone

        strides: List[int] = [8, 16, 32]

        if anchor_grids is None:
            anchor_grids: List[List[float]] = [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ]

        if anchor_generator is None:
            anchor_generator = AnchorGenerator(strides, anchor_grids)
        self.anchor_generator = anchor_generator

        if criterion is None:
            criterion = SetCriterion(
                anchor_generator.num_anchors,
                anchor_generator.strides,
                anchor_generator.anchor_grids,
                num_classes,
            )
        self.compute_loss = criterion

        if head is None:
            head = YOLOHead(
                backbone.out_channels,
                anchor_generator.num_anchors,
                anchor_generator.strides,
                num_classes,
            )
        self.head = head

        if post_process is None:
            post_process = PostProcess(score_thresh, nms_thresh, detections_per_img)
        self.post_process = post_process

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(
        self,
        losses: Dict[str, Tensor],
        detections: List[Dict[str, Tensor]],
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            return losses

        return detections

    def forward(
        self,
        samples: Tensor,
        targets: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Args:
            samples (NestedTensor): Expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        # get the features from the backbone
        features = self.backbone(samples)

        # compute the yolo heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors_tuple = self.anchor_generator(features)
        losses = {}
        detections: List[Dict[str, Tensor]] = []

        if self.training:
            assert targets is not None
            # compute the losses
            losses = self.compute_loss(targets, head_outputs)
        else:
            # compute the detections
            detections = self.post_process(head_outputs, anchors_tuple)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn(
                    "YOLO always returns a (Losses, Detections) tuple in scripting."
                )
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)


model_urls_root = "https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0"

model_urls = {
    # Path Aggregation Network
    "yolov5_darknet_pan_s_r31_coco": f"{model_urls_root}/yolov5_darknet_pan_s_r31_coco-eb728698.pt",
    "yolov5_darknet_pan_m_r31_coco": f"{model_urls_root}/yolov5_darknet_pan_m_r31_coco-670dc553.pt",
    "yolov5_darknet_pan_l_r31_coco": f"{model_urls_root}/yolov5_darknet_pan_l_r31_coco-4dcc8209.pt",
    "yolov5_darknet_pan_s_r40_coco": f"{model_urls_root}/yolov5_darknet_pan_s_r40_coco-e3fd213d.pt",
    "yolov5_darknet_pan_m_r40_coco": f"{model_urls_root}/yolov5_darknet_pan_m_r40_coco-d295cb02.pt",
    "yolov5_darknet_pan_l_r40_coco": f"{model_urls_root}/yolov5_darknet_pan_l_r40_coco-4416841f.pt",
    # Tranformer Attention Network
    "yolov5_darknet_tan_s_r40_coco": f"{model_urls_root}/yolov5_darknet_tan_s_r40_coco-fe1069ce.pt",
}


def _yolov5_darknet_pan(
    backbone_name: str,
    depth_multiple: float,
    width_multiple: float,
    version: str,
    weights_name: str,
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    """
    Constructs a YOLO model.

    Example::

        >>> model = yolov5(pretrained=True)
        >>> model.eval()
        >>> x = torch.rand(4, 3, 416, 320)
        >>> predictions = model(x)

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = darknet_pan_backbone(
        backbone_name, depth_multiple, width_multiple, version=version
    )

    model = YOLO(backbone, num_classes, **kwargs)
    if pretrained:
        if model_urls.get(weights_name, None) is None:
            raise ValueError(f"No checkpoint is available for model {weights_name}")
        state_dict = load_state_dict_from_url(
            model_urls[weights_name], progress=progress
        )
        model.load_state_dict(state_dict)

    return model


def yolov5_darknet_pan_s_r31(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""yolov5 small release 3.1 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_s_r3_1"
    weights_name = "yolov5_darknet_pan_s_r31_coco"
    depth_multiple = 0.33
    width_multiple = 0.5
    version = "r3.1"
    return _yolov5_darknet_pan(
        backbone_name,
        depth_multiple,
        width_multiple,
        version,
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        **kwargs,
    )


def yolov5_darknet_pan_m_r31(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""yolov5 medium release 3.1 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_m_r3_1"
    weights_name = "yolov5_darknet_pan_m_r31_coco"
    depth_multiple = 0.67
    width_multiple = 0.75
    version = "r3.1"
    return _yolov5_darknet_pan(
        backbone_name,
        depth_multiple,
        width_multiple,
        version,
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        **kwargs,
    )


def yolov5_darknet_pan_l_r31(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""yolov5 large release 3.1 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_l_r3_1"
    weights_name = "yolov5_darknet_pan_l_r31_coco"
    depth_multiple = 1.0
    width_multiple = 1.0
    version = "r3.1"
    return _yolov5_darknet_pan(
        backbone_name,
        depth_multiple,
        width_multiple,
        version,
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        **kwargs,
    )


def yolov5_darknet_pan_s_r40(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""yolov5 small release 4.0 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_s_r4_0"
    weights_name = "yolov5_darknet_pan_s_r40_coco"
    depth_multiple = 0.33
    width_multiple = 0.5
    version = "r4.0"
    return _yolov5_darknet_pan(
        backbone_name,
        depth_multiple,
        width_multiple,
        version,
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        **kwargs,
    )


def yolov5_darknet_pan_m_r40(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""yolov5 medium release 4.0 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_m_r4_0"
    weights_name = "yolov5_darknet_pan_m_r40_coco"
    depth_multiple = 0.67
    width_multiple = 0.75
    version = "r4.0"
    return _yolov5_darknet_pan(
        backbone_name,
        depth_multiple,
        width_multiple,
        version,
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        **kwargs,
    )


def yolov5_darknet_pan_l_r40(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""yolov5 large release 4.0 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_l_r4_0"
    weights_name = "yolov5_darknet_pan_l_r40_coco"
    depth_multiple = 1.0
    width_multiple = 1.0
    version = "r4.0"
    return _yolov5_darknet_pan(
        backbone_name,
        depth_multiple,
        width_multiple,
        version,
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        **kwargs,
    )


def yolov5_darknet_tan_s_r40(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""yolov5 small with a transformer block model from
    `"dingyiwei/yolov5" <https://github.com/ultralytics/yolov5/pull/2333>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_s_r4_0"
    weights_name = "yolov5_darknet_tan_s_r40_coco"
    depth_multiple = 0.33
    width_multiple = 0.5
    version = "r4.0"

    backbone = darknet_tan_backbone(
        backbone_name, depth_multiple, width_multiple, version=version
    )

    model = YOLO(backbone, num_classes, **kwargs)
    if pretrained:
        if model_urls.get(weights_name, None) is None:
            raise ValueError(f"No checkpoint is available for model {weights_name}")
        state_dict = load_state_dict_from_url(
            model_urls[weights_name], progress=progress
        )
        model.load_state_dict(state_dict)

    return model
