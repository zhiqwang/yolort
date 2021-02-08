import random
import torch
from torch.utils.data import Dataset

from torchvision import ops


__all__ = ["DummyDetectionDataset"]


class DummyCOCODetectionDataset(Dataset):
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
