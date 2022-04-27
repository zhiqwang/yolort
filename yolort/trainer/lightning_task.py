# Copyright (c) 2021, yolort team. All rights reserved.

import argparse
from pathlib import PosixPath
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yolort.models as models
from pytorch_lightning import LightningModule
from torch import Tensor
from torchvision.ops import box_iou
from yolort.data.coco_eval import COCOEvaluator


__all__ = ["DefaultTask"]


def _evaluate_iou(target, pred):
    """
    Evaluate intersection over union (IOU) for target from dataset and
    output prediction from model
    """
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()


class DefaultTask(LightningModule):
    """
    Wrapping the trainer into the YOLOv5 Module.

    Args:
        arch (string): YOLOv5 model architecture. Default: 'yolov5s'
        version (str): model released by the upstream YOLOv5. Possible values
            are ['r6.0']. Default: 'r6.0'.
        lr (float): The initial learning rate
        annotation_path (Optional[Union[string, PosixPath]]): Path of the COCO annotation file
            Default: None.
    """

    def __init__(
        self,
        arch: str = "yolov5s",
        version: str = "r6.0",
        lr: float = 0.01,
        annotation_path: Optional[Union[str, PosixPath]] = None,
        **kwargs: Any,
    ) -> None:

        super().__init__()

        self.model = models.__dict__[arch](upstream_version=version, **kwargs)
        self.lr = lr

        # evaluators for validation datasets
        self.evaluator = None
        if annotation_path is not None:
            self.evaluator = COCOEvaluator(annotation_path, iou_type="bbox")

        # used only on torchscript mode
        self._has_warned = False

    def forward(
        self,
        inputs: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        This exists since PyTorchLightning forward are used for inference only (separate from
        ``training_step``). We keep ``targets`` here for Backward Compatible.
        """
        return self.model(inputs, targets)

    def training_step(self, batch, batch_idx):
        """
        The training step.
        """
        loss_dict = self.model(*batch)
        loss = sum(loss_dict.values())
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        preds = self.model(images)
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
        preds = self.model(images)
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
