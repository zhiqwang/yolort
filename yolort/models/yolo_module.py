# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import argparse

import torch
from torch import Tensor

from pytorch_lightning import LightningModule

from . import yolo
from .transform import GeneralizedYOLOTransform

from ..datasets import DetectionDataModule, DataPipeline

from typing import Any, List, Dict, Tuple, Optional

__all__ = ['YOLOModule']


class YOLOModule(LightningModule):
    """
    PyTorch Lightning wrapper of `YOLO`
    """
    def __init__(
        self,
        lr: float = 0.01,
        arch: str = 'yolov5_darknet_pan_s_r31',
        pretrained: bool = False,
        progress: bool = True,
        num_classes: int = 80,
        min_size: int = 320,
        max_size: int = 416,
        **kwargs: Any,
    ):
        """
        Args:
            arch: architecture
            lr: the learning rate
            pretrained: if true, returns a model pre-trained on COCO train2017
            num_classes: number of detection classes (including background)
        """
        super().__init__()

        self.lr = lr
        self.num_classes = num_classes

        self.model = yolo.__dict__[arch](
            pretrained=pretrained, progress=progress, num_classes=num_classes, **kwargs)

        self.transform = GeneralizedYOLOTransform(min_size, max_size)

        self._data_pipeline = None

    def forward(
        self,
        inputs: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> List[Dict[str, Tensor]]:
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in inputs:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # Transform the input
        samples, targets = self.transform(inputs, targets)
        # Compute the detections
        detections = self.model(samples.tensors, targets=targets)
        # Rescale coordinate
        detections = self.transform.postprocess(detections, samples.image_sizes, original_image_sizes)

        return detections

    def training_step(self, batch, batch_idx):
        """
        The training step.
        """
        # Transform the input
        samples, targets = self.transform(*batch)
        # yolov5 takes both images and targets for training, returns
        loss_dict = self.model(samples.tensors, targets)
        loss = sum(loss for loss in loss_dict.values())
        return {"loss": loss, "log": loss_dict}

    @torch.no_grad()
    def predict(
        self,
        x: Any,
        batch_idx: Optional[int] = None,
        skip_collate_fn: bool = False,
        dataloader_idx: Optional[int] = None,
        data_pipeline: Optional[DataPipeline] = None,
    ) -> Any:
        """
        Predict function for raw data or processed data

        Args:

            x: Input to predict. Can be raw data or processed data.

            batch_idx: Batch index

            dataloader_idx: Dataloader index

            skip_collate_fn: Whether to skip the collate step.
                this is required when passing data already processed
                for the model, for example, data from a dataloader

            data_pipeline: Use this to override the current data pipeline

        Returns:
            The post-processed model predictions

        """
        data_pipeline = data_pipeline or self.data_pipeline
        batch = x if skip_collate_fn else data_pipeline.collate_fn(x)
        images, _ = batch if len(batch) == 2 and isinstance(batch, (list, tuple)) else (batch, None)
        images = [img.to(self.device) for img in images]
        predictions = self.forward(images)
        output = data_pipeline.uncollate_fn(predictions)  # TODO: pass batch and x
        return output

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=0.005,
        )

    @torch.jit.unused
    @property
    def data_pipeline(self) -> DataPipeline:
        # we need to save the pipeline in case this class
        # is loaded from checkpoint and used to predict
        if not self._data_pipeline:
            self._data_pipeline = self.default_pipeline()
        return self._data_pipeline

    @data_pipeline.setter
    def data_pipeline(self, data_pipeline: DataPipeline) -> None:
        self._data_pipeline = data_pipeline

    @staticmethod
    def default_pipeline() -> DataPipeline:
        """Pipeline to use when there is no datamodule or it has not defined its pipeline"""
        return DetectionDataModule.default_pipeline()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--arch', default='yolov5_darknet_pan_s_r31',
                            help='model architecture')
        parser.add_argument('--num_classes', default=80, type=int,
                            help='number classes of datasets')
        parser.add_argument('--pretrained', action='store_true',
                            help='Use pre-trained models from the modelzoo')
        parser.add_argument('--lr', default=0.02, type=float,
                            help='initial learning rate, 0.02 is the default value for training '
                            'on 8 gpus and 2 images_per_gpu')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--weight-decay', default=5e-4, type=float,
                            metavar='W', help='weight decay (default: 5e-4)')
        return parser
