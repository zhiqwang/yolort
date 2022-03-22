# Copyright (c) 2020, yolort team. All rights reserved.

import warnings
from typing import Any, Dict, List, Callable, Optional, Tuple

import torch
from torch import nn, Tensor
from yolort.utils import load_state_dict_from_url

from ._checkpoint import load_from_ultralytics
from ._utils import _ovewrite_value_param
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
    "yolov5_darknet_pan_n_r60",
    "yolov5_darknet_pan_n6_r60",
    "yolov5_darknet_pan_s_r60",
    "yolov5_darknet_pan_s6_r60",
    "yolov5_darknet_pan_m_r60",
    "yolov5_darknet_pan_m6_r60",
    "yolov5_darknet_pan_l_r60",
    "yolov5_darknet_pan_l6_r60",
    "yolov5_darknet_pan_x_r60",
    "yolov5_darknet_pan_x6_r60",
    "yolov5_darknet_tan_s_r40",
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

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        # Anchor parameters
        strides: Optional[List[int]] = None,
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

        if strides is None:
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
            criterion = SetCriterion(strides, anchor_grids, num_classes)
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
            post_process = PostProcess(
                anchor_generator.strides,
                score_thresh,
                nms_thresh,
                detections_per_img,
            )
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
        grids, shifts = self.anchor_generator(features)
        losses = {}
        detections: List[Dict[str, Tensor]] = []

        if self.training:
            assert targets is not None
            # compute the losses
            losses = self.compute_loss(targets, head_outputs)
        else:
            # compute the detections
            detections = self.post_process(head_outputs, grids, shifts)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("YOLO always returns a (Losses, Detections) tuple in scripting.")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

    @classmethod
    def load_from_yolov5(
        cls,
        checkpoint_path: str,
        score_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        version: str = "r6.0",
        post_process: Optional[nn.Module] = None,
    ):
        """
        Load model state from the checkpoint trained by YOLOv5.

        Args:
            checkpoint_path (str): Path of the YOLOv5 checkpoint model.
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            version (str): upstream version released by the ultralytics/yolov5, Possible
                values are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
        """
        model_info = load_from_ultralytics(checkpoint_path, version=version)
        backbone_name = f"darknet_{model_info['size']}_{version.replace('.', '_')}"
        depth_multiple = model_info["depth_multiple"]
        width_multiple = model_info["width_multiple"]
        use_p6 = model_info["use_p6"]
        backbone = darknet_pan_backbone(
            backbone_name, depth_multiple, width_multiple, version=version, use_p6=use_p6
        )
        model = cls(
            backbone,
            model_info["num_classes"],
            strides=model_info["strides"],
            anchor_grids=model_info["anchor_grids"],
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            post_process=post_process,
        )

        model.load_state_dict(model_info["state_dict"])
        return model


def _create_yolo(
    *,
    backbone_name: str,
    depth_multiple: float,
    width_multiple: float,
    version: str,
    progress: bool,
    num_classes: int,
    use_p6: bool,
    weights,
    **kwargs: Any,
) -> YOLO:
    if weights is not None:
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 80
    backbone = darknet_pan_backbone(
        backbone_name, depth_multiple, width_multiple, version=version, use_p6=use_p6
    )

    model = YOLO(backbone, num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


model_urls_root_r40 = "https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0"
model_urls_root_r60 = "https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.5.2-alpha"

model_urls = {
    # Path Aggregation Network 3.1 and 4.0
    "yolov5_darknet_pan_s_r31_coco": f"{model_urls_root_r40}/yolov5_darknet_pan_s_r31_coco-eb728698.pt",
    "yolov5_darknet_pan_m_r31_coco": f"{model_urls_root_r40}/yolov5_darknet_pan_m_r31_coco-670dc553.pt",
    "yolov5_darknet_pan_l_r31_coco": f"{model_urls_root_r40}/yolov5_darknet_pan_l_r31_coco-4dcc8209.pt",
    "yolov5_darknet_pan_s_r40_coco": f"{model_urls_root_r40}/yolov5_darknet_pan_s_r40_coco-e3fd213d.pt",
    "yolov5_darknet_pan_m_r40_coco": f"{model_urls_root_r40}/yolov5_darknet_pan_m_r40_coco-d295cb02.pt",
    "yolov5_darknet_pan_l_r40_coco": f"{model_urls_root_r40}/yolov5_darknet_pan_l_r40_coco-4416841f.pt",
    # Path Aggregation Network 6.0
    "yolov5_darknet_pan_n_r60_coco": f"{model_urls_root_r60}/yolov5_darknet_pan_n_r60_coco-bc15659e.pt",
    "yolov5_darknet_pan_n6_r60_coco": f"{model_urls_root_r60}/yolov5_darknet_pan_n6_r60_coco-4e823e0f.pt",
    "yolov5_darknet_pan_s_r60_coco": f"{model_urls_root_r60}/yolov5_darknet_pan_s_r60_coco-9f44bf3f.pt",
    "yolov5_darknet_pan_s6_r60_coco": f"{model_urls_root_r60}/yolov5_darknet_pan_s6_r60_coco-b4ff1fc2.pt",
    "yolov5_darknet_pan_m_r60_coco": f"{model_urls_root_r60}/yolov5_darknet_pan_m_r60_coco-58d32352.pt",
    "yolov5_darknet_pan_m6_r60_coco": f"{model_urls_root_r60}/yolov5_darknet_pan_m6_r60_coco-cc010533.pt",
    "yolov5_darknet_pan_l_r60_coco": f"{model_urls_root_r60}/yolov5_darknet_pan_l_r60_coco-321d8dcd.pt",
    # Tranformer Attention Network
    "yolov5_darknet_tan_s_r40_coco": f"{model_urls_root_r40}/yolov5_darknet_tan_s_r40_coco-fe1069ce.pt",
}


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
    return _create_yolo(
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
    return _create_yolo(
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
    return _create_yolo(
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
    return _create_yolo(
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
    return _create_yolo(
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
    return _create_yolo(
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


def yolov5_darknet_pan_n_r60(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""yolov5 nano release 6.0 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_n_r6_0"
    weights_name = "yolov5_darknet_pan_n_r60_coco"
    depth_multiple = 0.33
    width_multiple = 0.25
    version = "r6.0"
    return _create_yolo(
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


def yolov5_darknet_pan_s_r60(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""yolov5 small release 6.0 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_s_r6_0"
    weights_name = "yolov5_darknet_pan_s_r60_coco"
    depth_multiple = 0.33
    width_multiple = 0.5
    version = "r6.0"
    return _create_yolo(
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


def yolov5_darknet_pan_m_r60(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""yolov5 medium release 6.0 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_m_r6_0"
    weights_name = "yolov5_darknet_pan_m_r60_coco"
    depth_multiple = 0.67
    width_multiple = 0.75
    version = "r6.0"
    return _create_yolo(
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


def yolov5_darknet_pan_l_r60(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""yolov5 large release 6.0 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_l_r6_0"
    weights_name = "yolov5_darknet_pan_l_r60_coco"
    depth_multiple = 1.0
    width_multiple = 1.0
    version = "r6.0"
    return _create_yolo(
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


def yolov5_darknet_pan_x_r60(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""yolov5 X large release 6.0 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_x_r6_0"
    weights_name = "yolov5_darknet_pan_x_r60_coco"
    depth_multiple = 1.33
    width_multiple = 1.25
    version = "r6.0"
    return _create_yolo(
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


def yolov5_darknet_pan_n6_r60(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""
    YOLOv5 P6 nano release v6.0 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_n_r6_0"
    weights_name = "yolov5_darknet_pan_n6_r60_coco"
    depth_multiple = 0.33
    width_multiple = 0.25
    version = "r6.0"
    use_p6 = True
    strides = [8, 16, 32, 64]
    anchor_grids = [
        [19, 27, 44, 40, 38, 94],
        [96, 68, 86, 152, 180, 137],
        [140, 301, 303, 264, 238, 542],
        [436, 615, 739, 380, 925, 792],
    ]

    return _create_yolo(
        backbone_name,
        depth_multiple,
        width_multiple,
        version,
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        use_p6=use_p6,
        strides=strides,
        anchor_grids=anchor_grids,
        **kwargs,
    )


def yolov5_darknet_pan_s6_r60(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""
    YOLOv5 P6 small release v6.0 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_s_r6_0"
    weights_name = "yolov5_darknet_pan_s6_r60_coco"
    depth_multiple = 0.33
    width_multiple = 0.5
    version = "r6.0"
    use_p6 = True
    strides = [8, 16, 32, 64]
    anchor_grids = [
        [19, 27, 44, 40, 38, 94],
        [96, 68, 86, 152, 180, 137],
        [140, 301, 303, 264, 238, 542],
        [436, 615, 739, 380, 925, 792],
    ]

    return _create_yolo(
        backbone_name,
        depth_multiple,
        width_multiple,
        version,
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        use_p6=use_p6,
        strides=strides,
        anchor_grids=anchor_grids,
        **kwargs,
    )


def yolov5_darknet_pan_m6_r60(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""
    YOLOv5 P6 medium release v6.0 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_m_r6_0"
    weights_name = "yolov5_darknet_pan_m6_r60_coco"
    depth_multiple = 0.67
    width_multiple = 0.75
    version = "r6.0"
    use_p6 = True
    strides = [8, 16, 32, 64]
    anchor_grids = [
        [19, 27, 44, 40, 38, 94],
        [96, 68, 86, 152, 180, 137],
        [140, 301, 303, 264, 238, 542],
        [436, 615, 739, 380, 925, 792],
    ]

    return _create_yolo(
        backbone_name,
        depth_multiple,
        width_multiple,
        version,
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        use_p6=use_p6,
        strides=strides,
        anchor_grids=anchor_grids,
        **kwargs,
    )


def yolov5_darknet_pan_l6_r60(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""
    YOLOv5 P6 X large release v6.0 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_l_r6_0"
    weights_name = "yolov5_darknet_pan_l6_r60_coco"
    depth_multiple = 1.0
    width_multiple = 1.0
    version = "r6.0"
    use_p6 = True
    strides = [8, 16, 32, 64]
    anchor_grids = [
        [19, 27, 44, 40, 38, 94],
        [96, 68, 86, 152, 180, 137],
        [140, 301, 303, 264, 238, 542],
        [436, 615, 739, 380, 925, 792],
    ]

    return _create_yolo(
        backbone_name,
        depth_multiple,
        width_multiple,
        version,
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        use_p6=use_p6,
        strides=strides,
        anchor_grids=anchor_grids,
        **kwargs,
    )


def yolov5_darknet_pan_x6_r60(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLO:
    r"""
    YOLOv5 P6 X large release v6.0 model from
    `"ultralytics/yolov5" <https://zenodo.org/badge/latestdoi/264818686>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone_name = "darknet_x_r6_0"
    weights_name = "yolov5_darknet_pan_x6_r60_coco"
    depth_multiple = 1.33
    width_multiple = 1.25
    version = "r6.0"
    use_p6 = True
    strides = [8, 16, 32, 64]
    anchor_grids = [
        [19, 27, 44, 40, 38, 94],
        [96, 68, 86, 152, 180, 137],
        [140, 301, 303, 264, 238, 542],
        [436, 615, 739, 380, 925, 792],
    ]

    return _create_yolo(
        backbone_name,
        depth_multiple,
        width_multiple,
        version,
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        use_p6=use_p6,
        strides=strides,
        anchor_grids=anchor_grids,
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

    backbone = darknet_tan_backbone(backbone_name, depth_multiple, width_multiple, version=version)

    model = YOLO(backbone, num_classes, **kwargs)
    if pretrained:
        if model_urls.get(weights_name, None) is None:
            raise ValueError(f"No checkpoint is available for model {weights_name}")
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        model.load_state_dict(state_dict)

    return model
