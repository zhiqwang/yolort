# Copyright (c) 2021, yolort team. All rights reserved.

import warnings
from typing import Any, Dict, List, Callable, Optional, Tuple

import torch
import torchvision
from torch import nn, Tensor
from torchvision.io import read_image
from yolort.data import contains_any_tensor

from . import yolo
from .transform import YOLOTransform, _get_shape_onnx
from .yolo import YOLO

__all__ = ["YOLOv5"]


class YOLOv5(nn.Module):
    """
    Wrapping the pre-processing (`LetterBox`) into the YOLO models.

    Example:

        Demo pipeline for YOLOv5 Inference.

        .. code-block:: python
            from yolort.models import YOLOv5

            # Load model
            model = YOLOv5(arch="yolov5_darknet_pan_n_r60", pretrained=True, score_thresh=0.35)
            model = model.eval()

            # Perform inference on an image file
            predictions = model.predict("bus.jpg")
            # Perform inference on a list of image files
            predictions2 = model.predict(["bus.jpg", "zidane.jpg"])

    Args:
        arch (string): YOLO model architecture. Default: None
        model (nn.Module): YOLO model. Default: None
        num_classes (int): number of output classes of the model (doesn't including
            background). Default: 80
        pretrained (bool): If true, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        size: (Tuple[int, int]): the minimum and maximum size of the image to be rescaled.
            Default: (640, 640)
        size_divisible (int): stride of the models. Default: 32
        fixed_shape (Tuple[int, int], optional): Padding mode for letterboxing. If set to `True`,
            the image will be padded to shape `fixed_shape` if specified. Instead the image will
            be padded to a minimum rectangle to match `min_size / max_size` and each of its edges
            is divisible by `size_divisible` if it is not specified. Default: None
        fill_color (int): fill value for padding. Default: 114
    """

    def __init__(
        self,
        arch: Optional[str] = None,
        model: Optional[nn.Module] = None,
        num_classes: int = 80,
        pretrained: bool = False,
        progress: bool = True,
        size: Tuple[int, int] = (640, 640),
        size_divisible: int = 32,
        fixed_shape: Optional[Tuple[int, int]] = None,
        fill_color: int = 114,
        **kwargs: Any,
    ) -> None:

        super().__init__()

        self.arch = arch
        self.num_classes = num_classes

        if model is None:
            model = yolo.__dict__[arch](
                pretrained=pretrained,
                progress=progress,
                num_classes=num_classes,
                **kwargs,
            )
        self.model = model

        self.transform = YOLOTransform(
            size[0],
            size[1],
            size_divisible=size_divisible,
            fixed_shape=fixed_shape,
            fill_color=fill_color,
        )

        # used only on torchscript mode
        self._has_warned = False

    def _forward_impl(
        self,
        inputs: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Args:
            inputs (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `boxes`.
        """
        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []

        if not self.training:
            for img in inputs:
                val = img.shape[-2:]
                assert len(val) == 2
                original_image_sizes.append((val[0], val[1]))

        # Transform the input
        samples, targets = self.transform(inputs, targets)
        # Compute the detections
        outputs = self.model(samples.tensors, targets=targets)

        losses = {}
        detections: List[Dict[str, Tensor]] = []

        if self.training:
            # compute the losses
            if torch.jit.is_scripting():
                losses = outputs[0]
            else:
                losses = outputs
        else:
            # Rescale coordinate
            if torch.jit.is_scripting():
                result = outputs[1]
            else:
                result = outputs

            if torchvision._is_tracing():
                im_shape = _get_shape_onnx(samples.tensors)
            else:
                im_shape = torch.tensor(samples.tensors.shape[-2:])

            detections = self.transform.postprocess(result, im_shape, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("YOLOv5 always returns a (Losses, Detections) tuple in scripting.")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

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
        inputs: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        This exists since PyTorchLightning forward are used for inference only (separate from
        ``training_step``). We keep ``targets`` here for Backward Compatible.
        """
        return self._forward_impl(inputs, targets)

    @torch.no_grad()
    def predict(self, x: Any, image_loader: Optional[Callable] = None) -> List[Dict[str, Tensor]]:
        """
        Predict function for raw data or processed data
        Args:
            x: Input to predict. Can be raw data or processed data.
            image_loader: Utility function to convert raw data to Tensor.

        Returns:
            The post-processed model predictions.
        """
        image_loader = image_loader or self.default_loader
        images = self.collate_images(x, image_loader)
        return self.forward(images)

    def default_loader(self, img_path: str) -> Tensor:
        """
        Default loader of read a image path.

        Args:
            img_path (str): a image path

        Returns:
            Tensor, processed tensor for prediction.
        """
        return read_image(img_path) / 255.0

    def collate_images(self, samples: Any, image_loader: Callable) -> List[Tensor]:
        """
        Prepare source samples for inference.

        Args:
            samples (Any): samples source, support the following various types:
                - str or List[str]: a image path or list of image paths.
                - Tensor or List[Tensor]: a tensor or list of tensors.

        Returns:
            List[Tensor], The processed image samples.
        """
        p = next(self.parameters())  # for device and type
        if isinstance(samples, Tensor):
            return [samples.to(p.device).type_as(p)]

        if contains_any_tensor(samples):
            return [sample.to(p.device).type_as(p) for sample in samples]

        if isinstance(samples, str):
            samples = [samples]

        if isinstance(samples, (list, tuple)) and all(isinstance(p, str) for p in samples):
            outputs = []
            for sample in samples:
                output = image_loader(sample).to(p.device).type_as(p)
                outputs.append(output)
            return outputs

        raise NotImplementedError(
            f"The type of the sample is {type(samples)}, we currently don't support it now, the "
            "samples should be either a tensor, list of tensors, a image path or list of image paths."
        )

    @classmethod
    def load_from_yolov5(
        cls,
        checkpoint_path: str,
        *,
        lr: float = 0.01,
        size: Tuple[int, int] = (640, 640),
        size_divisible: int = 32,
        fixed_shape: Optional[Tuple[int, int]] = None,
        fill_color: int = 114,
        **kwargs: Any,
    ):
        """
        Load model state from the checkpoint trained by YOLOv5.

        Args:
            checkpoint_path (str): Path of the YOLOv5 checkpoint model.
            lr (float): The initial learning rate
            size: (Tuple[int, int]): the minimum and maximum size of the image to be rescaled.
                Default: (640, 640)
            size_divisible (int): stride of the models. Default: 32
            fixed_shape (Tuple[int, int], optional): Padding mode for letterboxing. If set to `True`,
                the image will be padded to shape `fixed_shape` if specified. Instead the image will
                be padded to a minimum rectangle to match `min_size / max_size` and each of its edges
                is divisible by `size_divisible` if it is not specified. Default: None
            fill_color (int): fill value for padding. Default: 114
        """
        model = YOLO.load_from_yolov5(checkpoint_path, **kwargs)
        yolov5 = cls(
            lr=lr,
            model=model,
            size=size,
            size_divisible=size_divisible,
            fixed_shape=fixed_shape,
            fill_color=fill_color,
        )
        return yolov5
