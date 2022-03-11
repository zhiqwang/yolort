# Copyright (c) 2021, yolort team. All rights reserved.

from pathlib import Path

import pytest
import pytorch_lightning as pl
from yolort.data import _helper as data_helper
from yolort.data.data_module import DetectionDataModule
from yolort.trainer import DefaultTask


def test_training_step():
    # Setup the DataModule
    data_path = "data-bin"
    train_dataset = data_helper.get_dataset(data_root=data_path, mode="train")
    val_dataset = data_helper.get_dataset(data_root=data_path, mode="val")
    data_module = DetectionDataModule(train_dataset, val_dataset, batch_size=8)
    # Load model
    model = DefaultTask(arch="yolov5n")
    model = model.train()
    # Trainer
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, data_module)


@pytest.mark.parametrize("arch, version, map5095, map50", [("yolov5s", "r4.0", 42.5, 65.3)])
def test_test_epoch_end(arch, version, map5095, map50):
    # Acquire the annotation file
    data_path = Path("data-bin")
    coco128_dirname = "coco128"
    data_helper.prepare_coco128(data_path, dirname=coco128_dirname)
    annotation_file = data_path / coco128_dirname / "annotations" / "instances_train2017.json"

    # Get dataloader to test
    val_dataloader = data_helper.get_dataloader(data_root=data_path, mode="val")

    # Load model
    model = DefaultTask(arch=arch, version=version, pretrained=True, annotation_path=annotation_file)

    # test step
    trainer = pl.Trainer(max_epochs=1)
    trainer.test(model, test_dataloaders=val_dataloader)
    # test epoch end
    results = model.evaluator.compute()
    assert results["AP"] > map5095
    assert results["AP50"] > map50
