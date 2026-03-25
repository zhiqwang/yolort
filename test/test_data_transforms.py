# Copyright (c) 2021, yolort team. All rights reserved.
import numpy as np
import torch
from PIL import Image

from yolort.data.transforms import (
    collate_fn,
    Compose,
    ConvertImageDtype,
    default_train_transforms,
    default_val_transforms,
    PILToTensor,
    RandomHorizontalFlip,
)


class TestCollateFunction:
    def test_basic_collate(self):
        batch = [
            (torch.randn(3, 32, 32), {"boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]])}),
            (torch.randn(3, 32, 32), {"boxes": torch.tensor([[0.5, 0.5, 1.0, 1.0]])}),
        ]
        images, targets = collate_fn(batch)
        assert len(images) == 2
        assert len(targets) == 2

    def test_single_element(self):
        batch = [(torch.randn(3, 32, 32), {"label": 1})]
        images, targets = collate_fn(batch)
        assert len(images) == 1


class TestCompose:
    def test_identity_transform(self):
        class Identity:
            def __call__(self, image, target):
                return image, target

        transform = Compose([Identity()])
        img = torch.randn(3, 32, 32)
        target = {"boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]])}
        result_img, result_target = transform(img, target)
        torch.testing.assert_close(result_img, img)

    def test_multiple_transforms(self):
        call_count = [0]

        class Counter:
            def __call__(self, image, target):
                call_count[0] += 1
                return image, target

        transform = Compose([Counter(), Counter(), Counter()])
        img = torch.randn(3, 32, 32)
        transform(img, {})
        assert call_count[0] == 3


class TestRandomHorizontalFlip:
    def test_always_flip(self):
        """Test with p=1.0 so flip always occurs."""
        transform = RandomHorizontalFlip(p=1.0)
        # Create a simple 3x4x4 image
        img = torch.zeros(3, 4, 8)
        img[:, :, 0] = 1.0  # Mark left column

        target = {
            "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
            "labels": torch.tensor([0]),
        }
        result_img, result_target = transform(img, target)
        # After horizontal flip, left column should become right column
        assert result_img[:, :, -1].sum() > 0

    def test_never_flip(self):
        """Test with p=0.0 so flip never occurs."""
        transform = RandomHorizontalFlip(p=0.0)
        img = torch.randn(3, 32, 32)
        target = {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([0]),
        }
        result_img, result_target = transform(img, target)
        torch.testing.assert_close(result_img, img)
        torch.testing.assert_close(result_target["boxes"], target["boxes"])


class TestPILToTensor:
    def test_pil_conversion(self):
        transform = PILToTensor()
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        result_img, result_target = transform(img, None)
        assert isinstance(result_img, torch.Tensor)
        assert result_img.shape == (3, 32, 32)
        assert result_img.dtype == torch.uint8


class TestConvertImageDtype:
    def test_uint8_to_float(self):
        transform = ConvertImageDtype(torch.float32)
        img = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
        result_img, result_target = transform(img, None)
        assert result_img.dtype == torch.float32
        assert result_img.max() <= 1.0
        assert result_img.min() >= 0.0


class TestDefaultTransforms:
    def test_train_transforms_creation(self):
        transforms = default_train_transforms()
        assert transforms is not None
        assert hasattr(transforms, "transforms")

    def test_val_transforms_creation(self):
        transforms = default_val_transforms()
        assert transforms is not None
