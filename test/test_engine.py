# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import Tensor
from torchvision.io import read_image
from yolort.data import COCOEvaluator, DetectionDataModule, _helper as data_helper
from yolort.models import yolov5s
from yolort.models.transform import nested_tensor_from_tensor_list
from yolort.models.yolo import yolov5_darknet_pan_s_r31


def default_loader(img_name, is_half=False):
    """
    Read Image using TorchVision.io Here
    """
    img = read_image(img_name)
    img = img.half() if is_half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    return img


def test_train_with_vanilla_model():
    # Do forward over image
    img_name = "test/assets/zidane.jpg"
    img_tensor = default_loader(img_name)
    assert img_tensor.ndim == 3
    # Add a dummy image to train
    img_dummy = torch.rand((3, 416, 360), dtype=torch.float32)

    images = nested_tensor_from_tensor_list([img_tensor, img_dummy])
    targets = torch.tensor(
        [
            [0, 7, 0.3790, 0.5487, 0.3220, 0.2047],
            [0, 2, 0.2680, 0.5386, 0.2200, 0.1779],
            [0, 3, 0.1720, 0.5403, 0.1960, 0.1409],
            [0, 4, 0.2240, 0.4547, 0.1520, 0.0705],
        ],
        dtype=torch.float,
    )

    model = yolov5_darknet_pan_s_r31(num_classes=12)
    model.train()
    out = model(images, targets)
    assert isinstance(out, dict)
    assert isinstance(out["cls_logits"], Tensor)
    assert isinstance(out["bbox_regression"], Tensor)
    assert isinstance(out["objectness"], Tensor)


def test_train_with_vanilla_module():
    """
    For issue #86: <https://github.com/zhiqwang/yolov5-rt-stack/issues/86>
    """
    # Define the device
    device = torch.device("cpu")

    train_dataloader = data_helper.get_dataloader(data_root="data-bin", mode="train")
    # Sample a pair of images/targets
    images, targets = next(iter(train_dataloader))
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # Define the model
    model = yolov5s(num_classes=80)
    model.train()

    out = model(images, targets)
    assert isinstance(out, dict)
    assert isinstance(out["cls_logits"], Tensor)
    assert isinstance(out["bbox_regression"], Tensor)
    assert isinstance(out["objectness"], Tensor)


def test_training_step():
    # Setup the DataModule
    data_path = "data-bin"
    train_dataset = data_helper.get_dataset(data_root=data_path, mode="train")
    val_dataset = data_helper.get_dataset(data_root=data_path, mode="val")
    data_module = DetectionDataModule(train_dataset, val_dataset, batch_size=16)
    # Load model
    model = yolov5s()
    model.train()
    # Trainer
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, data_module)


def test_vanilla_coco_evaluator():
    # Acquire the images and labels from the coco128 dataset
    val_dataloader = data_helper.get_dataloader(data_root="data-bin", mode="val")
    coco = data_helper.get_coco_api_from_dataset(val_dataloader.dataset)
    coco_evaluator = COCOEvaluator(coco)
    # Load model
    model = yolov5s(upstream_version="r4.0", pretrained=True)
    model.eval()
    for images, targets in val_dataloader:
        preds = model(images)
        coco_evaluator.update(preds, targets)

    results = coco_evaluator.compute()
    assert results["AP"] > 37.8
    assert results["AP50"] > 59.6


def test_test_epoch_end():
    # Acquire the annotation file
    data_path = Path("data-bin")
    coco128_dirname = "coco128"
    data_helper.prepare_coco128(data_path, dirname=coco128_dirname)
    annotation_file = data_path / coco128_dirname / "annotations" / "instances_train2017.json"

    # Get dataloader to test
    val_dataloader = data_helper.get_dataloader(data_root=data_path, mode="val")

    # Load model
    model = yolov5s(upstream_version="r4.0", pretrained=True, annotation_path=annotation_file)

    # test step
    trainer = pl.Trainer(max_epochs=1)
    trainer.test(model, test_dataloaders=val_dataloader)
    # test epoch end
    results = model.evaluator.compute()
    assert results["AP"] > 37.8
    assert results["AP50"] > 59.6


def test_predict_with_vanilla_model():
    # Set image inputs
    img_name = "test/assets/zidane.jpg"
    img_input = default_loader(img_name)
    assert img_input.ndim == 3
    # Load model
    model = yolov5s(pretrained=True)
    model.eval()
    # Perform inference on a list of tensors
    out = model([img_input])
    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], dict)
    assert isinstance(out[0]["boxes"], Tensor)
    assert isinstance(out[0]["labels"], Tensor)
    assert isinstance(out[0]["scores"], Tensor)


def test_predict_with_tensor():
    # Set image inputs
    img_name = "test/assets/zidane.jpg"
    img_tensor = default_loader(img_name)
    assert img_tensor.ndim == 3
    # Load model
    model = yolov5s(pretrained=True)
    model.eval()
    # Perform inference on a list of image files
    predictions = model.predict(img_tensor)
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert isinstance(predictions[0], dict)
    assert isinstance(predictions[0]["boxes"], Tensor)
    assert isinstance(predictions[0]["labels"], Tensor)
    assert isinstance(predictions[0]["scores"], Tensor)


def test_predict_with_tensors():
    # Set image inputs
    img_tensor1 = default_loader("test/assets/zidane.jpg")
    assert img_tensor1.ndim == 3
    img_tensor2 = default_loader("test/assets/bus.jpg")
    assert img_tensor2.ndim == 3
    img_tensors = [img_tensor1, img_tensor2]
    # Load model
    model = yolov5s(pretrained=True)
    model.eval()
    # Perform inference on a list of image files
    predictions = model.predict(img_tensors)
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert isinstance(predictions[0], dict)
    assert isinstance(predictions[0]["boxes"], Tensor)
    assert isinstance(predictions[0]["labels"], Tensor)
    assert isinstance(predictions[0]["scores"], Tensor)


def test_predict_with_image_file():
    # Set image inputs
    img_name = "test/assets/zidane.jpg"
    # Load model
    model = yolov5s(pretrained=True)
    model.eval()
    # Perform inference on an image file
    predictions = model.predict(img_name)
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert isinstance(predictions[0], dict)
    assert isinstance(predictions[0]["boxes"], Tensor)
    assert isinstance(predictions[0]["labels"], Tensor)
    assert isinstance(predictions[0]["scores"], Tensor)


def test_predict_with_image_files():
    # Set image inputs
    img_name1 = "test/assets/zidane.jpg"
    img_name2 = "test/assets/bus.jpg"
    img_names = [img_name1, img_name2]
    # Load model
    model = yolov5s(pretrained=True)
    model.eval()
    # Perform inference on a list of image files
    predictions = model.predict(img_names)
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert isinstance(predictions[0], dict)
    assert isinstance(predictions[0]["boxes"], Tensor)
    assert isinstance(predictions[0]["labels"], Tensor)
    assert isinstance(predictions[0]["scores"], Tensor)
