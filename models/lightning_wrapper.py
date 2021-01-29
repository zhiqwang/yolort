# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import argparse

import torch
from torch import nn, Tensor
from torchvision.models.utils import load_state_dict_from_url

import pytorch_lightning as pl

from . import yolo
from .transform import nested_tensor_from_tensor_list

from typing import Tuple, Any, List, Dict, Optional


class YOLOLitWrapper(pl.LightningModule):
    """
    PyTorch Lightning implementation of `YOLO`
    """
    def __init__(
        self,
        lr: float = 0.01,
        arch: str = 'yolov5_darknet_pan_s_r31',
        pretrained: bool = False,
        progress: bool = True,
        num_classes: int = 80,
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

    def forward(self, inputs: List[Tensor], targets: Optional[Tensor] = None):
        sample = nested_tensor_from_tensor_list(inputs)
        return self.model(sample.tensors, targets=targets)

    def training_step(self, batch, batch_idx):

        samples, targets = batch

        # yolov5 takes both images and targets for training, returns
        loss_dict = self.model(samples.tensors, targets)
        loss = sum(loss for loss in loss_dict.values())
        return {"loss": loss, "log": loss_dict}

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=0.005,
        )

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
