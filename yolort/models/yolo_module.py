# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import argparse
import warnings
from pathlib import PosixPath
from typing import Any, List, Dict, Tuple, Optional, Union, Callable

import torch
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torchvision.io import read_image
from yolort.data import COCOEvaluator, contains_any_tensor

from . import yolo
from ._utils import _evaluate_iou
from .transform import YOLOTransform
from .yolo import YOLO

__all__ = ["YOLOv5"]


class YOLOv5(LightningModule):
    """
    PyTorch Lightning wrapper of `YOLO`
    """

    def __init__(
        self,
        lr: float = 0.01,
        arch: Optional[str] = None,
        model: Optional[nn.Module] = None,
        pretrained: bool = False,
        progress: bool = True,
        size: Tuple[int, int] = (640, 640),
        num_classes: int = 80,
        annotation_path: Optional[Union[str, PosixPath]] = None,
        **kwargs: Any,
    ):
        """
        Args:
            lr (float): The initial learning rate
            arch (str): YOLO model architecture. Default: None
            model (nn.Module): YOLO model. Default: None
            pretrained (bool): If true, returns a model pre-trained on COCO train2017
            progress (bool): If True, displays a progress bar of the download to stderr
            size: (Tuple[int, int]): the width and height to which images will be rescaled
                before feeding them to the backbone. Default: (640, 640).
            num_classes (int): number of output classes of the model (doesn't including
                background). Default: 80.
            annotation_path (Optional[Union[str, PosixPath]]): Path of the COCO annotation file
                Default: None.
        """
        super().__init__()

        self.lr = lr
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

        self.transform = YOLOTransform(min(size), max(size), fixed_size=size)

        # metrics
        self.evaluator = None
        if annotation_path is not None:
            self.evaluator = COCOEvaluator(annotation_path, iou_type="bbox")

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
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
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

            detections = self.transform.postprocess(result, samples.image_sizes, original_image_sizes)

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

    def training_step(self, batch, batch_idx):
        """
        The training step.
        """
        loss_dict = self._forward_impl(*batch)
        loss = sum(loss_dict.values())
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        preds = self._forward_impl(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, preds)]).mean()
        outs = {"val_iou": iou}
        self.log_dict(outs, on_step=True, on_epoch=True, prog_bar=True)
        return outs

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        self.log("avg_val_iou", avg_iou)

    def test_step(self, batch, batch_idx):
        """
        The test step.
        """
        images, targets = batch
        images = list(image.to(next(self.parameters()).device) for image in images)
        preds = self._forward_impl(images)
        results = self.evaluator(preds, targets)
        # log step metric
        self.log("eval_step", results, prog_bar=True, on_step=True)

    def test_epoch_end(self, outputs):
        return self.log("coco_eval", self.evaluator.compute())

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

    @torch.no_grad()
    def predict(
        self,
        x: Any,
        image_loader: Optional[Callable] = None,
    ) -> List[Dict[str, Tensor]]:
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
        outputs = self.forward(images)
        return outputs

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--arch", default="yolov5_darknet_pan_s_r40", help="model architecture")
        parser.add_argument(
            "--pretrained",
            action="store_true",
            help="Use pre-trained models from the modelzoo",
        )
        parser.add_argument(
            "--lr",
            default=0.01,
            type=float,
            help="initial learning rate, 0.01 is the default value for training "
            "on 8 gpus and 2 images_per_gpu",
        )
        parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
        parser.add_argument(
            "--weight-decay",
            default=5e-4,
            type=float,
            metavar="W",
            help="weight decay (default: 5e-4)",
        )
        return parser

    @classmethod
    def load_from_yolov5(
        cls,
        checkpoint_path: str,
        lr: float = 0.01,
        size: Tuple[int, int] = (640, 640),
        **kwargs: Any,
    ):
        """
        Load model state from the checkpoint trained by YOLOv5.

        Args:
            checkpoint_path (str): Path of the YOLOv5 checkpoint model.
            lr (float): The initial learning rate
            size: (Tuple[int, int]): the width and height to which images will be rescaled
                before feeding them to the backbone. Default: (640, 640).
        """
        model = YOLO.load_from_yolov5(checkpoint_path, **kwargs)
        yolov5 = cls(lr=lr, model=model, size=size)
        return yolov5
