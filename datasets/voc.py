import torch
import torchvision

from .detection_datamodule import (
    DetectionDataModule,
    default_train_transforms,
    default_val_transforms,
)

from typing import Callable, List, Any, Optional


class VOCDetectionDataModule(DetectionDataModule):
    def __init__(
        self,
        data_path: str,
        years: List[str] = ["2007", "2012"],
        train_transform: Optional[Callable] = default_train_transforms,
        val_transform: Optional[Callable] = default_val_transforms,
        batch_size: int = 1,
        num_workers: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        train_dataset, num_classes = self.build_datasets(
            data_path, image_set='train', years=years, transforms=train_transform)
        val_dataset, _ = self.build_datasets(
            data_path, image_set='val', years=years, transforms=val_transform)

        super().__init__(train_dataset=train_dataset, val_dataset=val_dataset,
                         batch_size=batch_size, num_workers=num_workers, *args, **kwargs)

        self.num_classes = num_classes

    @staticmethod
    def build_datasets(data_path, image_set, years, transforms):
        datasets = []
        for year in years:
            dataset = VOCDetection(
                data_path,
                year=year,
                image_set=image_set,
                transforms=transforms,
            )
            datasets.append(dataset)

        num_classes = len(datasets[0].prepare.CLASSES)

        if len(datasets) == 1:
            return datasets[0], num_classes
        else:
            return torch.utils.data.ConcatDataset(datasets), num_classes


class ConvertVOCtoCOCO(object):

    CLASSES = (
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
    )

    def __call__(self, image, target):
        # return image, target
        anno = target['annotations']
        filename = anno['filename'].split('.')[0]
        image_id = target['image_id']
        image_id = torch.tensor([image_id])

        height, width = anno['size']['height'], anno['size']['width']

        boxes = []
        classes = []
        ishard = []
        objects = anno['object']
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bbox = obj['bndbox']
            bbox = [int(bbox[n]) - 1 for n in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append(bbox)
            classes.append(self.CLASSES.index(obj['name']))
            ishard.append(int(obj['difficult']))

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        classes = torch.as_tensor(classes, dtype=torch.int64)
        ishard = torch.as_tensor(ishard, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = classes
        target['ishard'] = ishard

        target['image_id'] = image_id
        target["orig_size"] = torch.as_tensor([int(height), int(width)])
        target["size"] = torch.as_tensor([int(height), int(width)])
        # convert filename in int8
        target['filename'] = torch.tensor([ord(i) for i in list(filename)], dtype=torch.int8)

        return image, target


class VOCDetection(torchvision.datasets.VOCDetection):
    def __init__(self, img_folder, year, image_set, transforms):
        super().__init__(img_folder, year=year, image_set=image_set)
        self._transforms = transforms
        self.prepare = ConvertVOCtoCOCO()

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = {
            'image_id': index,
            'annotations': target['annotation'],
        }
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target
