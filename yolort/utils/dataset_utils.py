# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import random
from pathlib import Path, PosixPath
from zipfile import ZipFile

import torch
from torchvision import ops

from ..data.coco import CocoDetection
from ..data.transforms import (
    collate_fn,
    default_train_transforms,
    default_val_transforms,
)


def prepare_coco128(
    data_path: PosixPath,
    dirname: str = 'coco128',
) -> None:
    """
    Prepare coco128 dataset to test.

    Args:
        data_path (PosixPath): root path of coco128 dataset.
        dirname (str): the directory name of coco128 dataset. Default: 'coco128'.
    """
    if not data_path.is_dir():
        print(f'Create a new directory: {data_path}')
        data_path.mkdir(parents=True, exist_ok=True)

    zip_path = data_path / 'coco128.zip'
    coco128_url = 'https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip'
    if not zip_path.is_file():
        print(f'Downloading coco128 datasets form {coco128_url}')
        torch.hub.download_url_to_file(coco128_url, zip_path, hash_prefix='a67d2887')

    coco128_path = data_path / dirname
    if not coco128_path.is_dir():
        print(f'Unzipping dataset to {coco128_path}')
        with ZipFile(zip_path, 'r') as zip_obj:
            zip_obj.extractall(data_path)


def get_data_loader(mode: str = 'train', batch_size: int = 4):
    # Prepare the datasets for training
    # Acquire the images and labels from the coco128 dataset
    data_path = Path('data-bin')
    coco128_dirname = 'coco128'
    coco128_path = data_path / coco128_dirname
    image_root = coco128_path / 'images' / 'train2017'
    annotation_file = coco128_path / 'annotations' / 'instances_train2017.json'

    if not annotation_file.is_file():
        prepare_coco128(data_path, dirname=coco128_dirname)

    if mode == 'train':
        dataset = CocoDetection(image_root, annotation_file, default_train_transforms())
    elif mode == 'val':
        dataset = CocoDetection(image_root, annotation_file, default_val_transforms())
    else:
        raise NotImplementedError(f"Currently not support {mode} mode")

    # We adopt the sequential sampler in order to repeat the experiment
    sampler = torch.utils.data.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    return loader


class DummyCOCODetectionDataset(torch.utils.data.Dataset):
    """
    Generate a dummy dataset for detection
    Example::
        >>> ds = DummyDetectionDataset()
        >>> dl = DataLoader(ds, batch_size=16)
    """
    def __init__(
        self,
        im_size_min: int = 320,
        im_size_max: int = 640,
        num_classes: int = 5,
        num_boxes_max: int = 12,
        class_start: int = 0,
        num_samples: int = 1000,
        box_fmt: str = "cxcywh",
        normalize: bool = True,
    ):
        """
        Args:
            im_size_min: Minimum image size
            im_size_max: Maximum image size
            num_classes: Number of classes for image.
            num_boxes_max: Maximum number of boxes per images
            num_samples: how many samples to use in this dataset.
            box_fmt: Format of Bounding boxes, supported : "xyxy", "xywh", "cxcywh"
        """
        super().__init__()
        self.im_size_min = im_size_min
        self.im_size_max = im_size_max
        self.num_classes = num_classes
        self.num_boxes_max = num_boxes_max
        self.num_samples = num_samples
        self.box_fmt = box_fmt
        self.class_start = class_start
        self.class_end = self.class_start + self.num_classes
        self.normalize = normalize

    def __len__(self):
        return self.num_samples

    @staticmethod
    def _random_bbox(img_shape):
        _, h, w = img_shape
        xs = torch.randint(w, (2,), dtype=torch.float32)
        ys = torch.randint(h, (2,), dtype=torch.float32)

        # A small hacky fix to avoid degenerate boxes.
        return [min(xs), min(ys), max(xs) + 1, max(ys) + 1]

    def __getitem__(self, idx: int):
        h = random.randint(self.im_size_min, self.im_size_max)
        w = random.randint(self.im_size_min, self.im_size_max)
        img_shape = (3, h, w)
        img = torch.rand(img_shape)

        num_boxes = random.randint(1, self.num_boxes_max)
        labels = torch.randint(self.class_start, self.class_end, (num_boxes,), dtype=torch.long)

        boxes = torch.tensor([self._random_bbox(img_shape) for _ in range(num_boxes)], dtype=torch.float32)
        boxes = ops.clip_boxes_to_image(boxes, (h, w))
        # No problems if we pass same in_fmt and out_fmt, it is covered by box_convert
        boxes = ops.box_convert(boxes, in_fmt="xyxy", out_fmt=self.box_fmt)
        if self.normalize:
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        image_id = torch.tensor([idx])
        return img, {"image_id": image_id, "boxes": boxes, "labels": labels}
