# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import warnings

import torch
from torch import nn, Tensor

from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from torch.jit.annotations import Tuple, List, Dict, Optional

from .backbone import darknet
from .box_head import YoloHead, PostProcess
from .anchor_utils import AnchorGenerator


class YOLO(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        # transform parameters
        min_size: int = 320,
        max_size: int = 416,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        # Anchor parameters
        anchor_generator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        # Post Process parameter
        postprocess_detections: Optional[nn.Module] = None,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 300,
    ):
        super().__init__()
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")
        self.backbone = backbone

        if anchor_generator is None:
            anchor_sizes = tuple((x,) for x in [128, 256, 512])
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.anchor_generator = anchor_generator

        if head is None:
            head = YoloHead(
                backbone.out_channels,
                anchor_generator.num_anchors_per_location()[0],
                num_classes,
            )
        self.head = head

        if image_mean is None:
            image_mean = [0., 0., 0.]
        if image_std is None:
            image_std = [1., 1., 1.]

        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        if postprocess_detections is None:
            postprocess_detections = PostProcess(score_thresh, nms_thresh, detections_per_img)
        self.postprocess_detections = postprocess_detections

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
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During Training, it returns a dict[Tensor] which contains the losses
                TODO, currently this repo doesn't support training.
                During Testing, it returns list[BoxList] contains additional fields
                like `scores` and `labels`.
        """
        # get the original image sizes
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        features = self.backbone(images.tensors)

        # compute the yolo heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections = torch.jit.annotate(List[Dict[str, Tensor]], [])

        if self.training:
            assert targets is not None
        else:
            # compute the detections
            detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("YOLO always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)


model_urls = {
    'yolov5s':
        'https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.2.0/yolov5s.pt',
}


def yolov5s(pretrained=False, progress=True,
            num_classes=80, pretrained_backbone=True, **kwargs):
    """
    Constructs a YOLO model.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
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
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values between
          ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction

    Example::

        >>> model = yolov5s(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 416, 320), torch.rand(3, 480, 352)]
        >>> predictions = model(x)

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = darknet(cfg_path='./models/yolov5s.yaml', pretrained=pretrained_backbone)
    model = YOLO(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['yolov5s'], progress=progress)
        model.load_state_dict(state_dict)
    return model
