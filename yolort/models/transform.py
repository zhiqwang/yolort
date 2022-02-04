# Copyright (c) 2020, yolort team. All rights reserved.

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import box_convert


class NestedTensor:
    """
    Structure that holds a list of images (of possibly varying sizes) as
    a single tensor.

    This works by padding the images to the same size, and storing in a
    field the original sizes of each image.
    """

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]):
        """
        Args:
            tensors (Tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device) -> "NestedTensor":
        cast_tensor = self.tensors.to(device)
        return NestedTensor(cast_tensor, self.image_sizes)

    def __repr__(self):
        return str(self.tensors)


@torch.jit.unused
def _get_shape_onnx(image: Tensor) -> Tensor:
    from torch.onnx import operators

    return operators.shape_as_tensor(image)[-2:]


@torch.jit.unused
def _tracing_item_onnx(v: Tensor) -> int:
    """
    ONNX requires a tensor type for Tensor.item() in tracing mode, so we cast
    its type to int here.
    """
    from typing import cast

    return cast(int, v)


def _resize_image_and_masks(
    image: Tensor,
    new_shape: Tuple[int, int],
    *,
    target: Optional[Dict[str, Tensor]] = None,
) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    if torchvision._is_tracing():
        im_shape = _get_shape_onnx(image)
    else:
        im_shape = torch.tensor(image.shape[-2:])

    ratio = torch.min(new_shape[0] / im_shape[0], new_shape[1] / im_shape[1])

    ratio_h = torch.round(im_shape[0] * ratio).to(dtype=torch.int32)
    ratio_w = torch.round(im_shape[1] * ratio).to(dtype=torch.int32)

    if torchvision._is_tracing():
        new_unpad = _tracing_item_onnx(ratio_h), _tracing_item_onnx(ratio_w)
    else:
        new_unpad = int(ratio_h.item()), int(ratio_w.item())

    image = F.interpolate(image[None], size=new_unpad, mode="bilinear", align_corners=False)[0]

    if target is None:
        return image, target

    if "masks" in target:
        mask = target["masks"]
        mask = F.interpolate(mask[:, None].float(), size=new_unpad, align_corners=False)[:, 0].byte()
        target["masks"] = mask
    return image, target


class YOLOTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a YOLO model. It plays
    the same role of `LetterBox`, and YOLOv5 adopt (0, 1, RGB) as the default mean, std and
    channel mode. We do not normalize below, the inputs need to be scaled down to float [0-1]
    from int[0-255] and transpose the image channel to RGB before being fed to this transformation.

    The transformations it perform are:
        - input / target resizing to get a rectangle within shape `(height, width)` that
            can be divided by `size_divisible`

    It returns a `NestedTensor` for the inputs, and a List[Dict[Tensor]] for the targets.

    Args:
        height (int) : expected height of the image to be rescaled
        width (int) : expected width of the image to be rescaled
        size_divisible (int): stride of the models. Default: 32
        auto_rectangle (bool): The padding mode. If set to `True`, the image will be
            padded to a minimum rectangle within shape `(height, width)` and each of its
            edges is divisible by `size_divisible`. If set to `False`, the image will
            be padded to shape `(height, width)`. Default: True
        fill_color (int): fill value for padding. Default: 114
    """

    def __init__(
        self,
        height: int,
        width: int,
        *,
        size_divisible: int = 32,
        auto_rectangle: bool = True,
        fill_color: int = 114,
    ) -> None:

        super().__init__()
        self.new_shape = (height, width)
        self.size_divisible = size_divisible
        self.auto_rectangle = auto_rectangle
        self.fill_color = fill_color / 255

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[NestedTensor, Optional[Tensor]]:
        device = images[0].device
        images = [img for img in images]
        if targets is not None:
            # make a copy of targets to avoid modifying it in-place
            # once torchscript supports dict comprehension
            # this can be simplified as as follows
            # targets = [{k: v for k,v in t.items()} for t in targets]
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v.to(device=device)
                targets_copy.append(data)
            targets = targets_copy

        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError(
                    "images is expected to be a list of 3d tensors of "
                    f"shape [C, H, W], but got '{image.shape}'."
                )

            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = NestedTensor(images, image_sizes_list)

        if targets is not None:
            targets_batched = []
            for i, target in enumerate(targets):
                num_objects = len(target["labels"])
                if num_objects > 0:
                    targets_merged = torch.full((num_objects, 6), i, dtype=torch.float32, device=device)
                    targets_merged[:, 1] = target["labels"]
                    targets_merged[:, 2:] = target["boxes"]
                    targets_batched.append(targets_merged)
            targets_batched = torch.cat(targets_batched, dim=0)
        else:
            targets_batched = None

        return image_list, targets_batched

    def torch_choice(self, k: List[int]) -> int:
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0.0, float(len(k))).item())
        return k[index]

    def resize(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:

        h, w = image.shape[-2:]
        image, target = _resize_image_and_masks(image, self.new_shape, target)

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = normalize_boxes(bbox, (h, w))
        target["boxes"] = bbox

        return image, target

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images: List[Tensor]) -> Tensor:
        max_size = []
        for i in range(1, images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32))
            max_size.append(max_size_i.to(torch.int32))
        stride = self.size_divisible
        max_size[0] = (torch.ceil((max_size[0].to(torch.float32)) / stride) * stride).to(torch.int32)
        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int32)

        # work around for
        # batched_imgs[i, :channel, dh : dh + img_h, dw : dw + img_w].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:

            img_h, img_w = img.shape[-2:]

            dh = (max_size[1] - img_w) / 2
            dw = (max_size[0] - img_h) / 2

            padding = (
                _tracing_item_onnx(torch.round(dh - 0.1).to(dtype=torch.int32)),
                _tracing_item_onnx(torch.round(dh + 0.1).to(dtype=torch.int32)),
                _tracing_item_onnx(torch.round(dw - 0.1).to(dtype=torch.int32)),
                _tracing_item_onnx(torch.round(dw + 0.1).to(dtype=torch.int32)),
            )
            padded_img = F.pad(img, padding, value=self.fill_color)

            padded_imgs.append(padded_img)

        return torch.stack(padded_imgs)

    def max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images: List[Tensor]) -> Tensor:
        """
        Nest a list of tensors. It plays the same role of the lettebox function.
        """

        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images)

        stride = float(self.size_divisible)
        max_size = self.max_by_axis([list(img.shape) for img in images])
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, self.fill_color)
        for i in range(batched_imgs.shape[0]):
            img = images[i]
            channel, img_h, img_w = img.shape
            # divide padding into 2 sides below
            dh = (max_size[1] - img_h) / 2
            dh = int(round(dh - 0.1))

            dw = (max_size[2] - img_w) / 2
            dw = int(round(dw - 0.1))

            batched_imgs[i, :channel, dh : dh + img_h, dw : dw + img_w].copy_(img)

        return batched_imgs

    def postprocess(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: Tensor,
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:

        for i, (pred, o_im_s) in enumerate(zip(result, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = scale_coords(boxes, image_shapes, o_im_s)
            result[i]["boxes"] = boxes

        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        _indent = "\n    "
        format_string += f"{_indent}Resize(height={self.new_shape[0]}, width={self.new_shape[1]})"
        format_string += "\n)"
        return format_string


def scale_coords(boxes: Tensor, new_size: Tensor, original_size: Tuple[int, int]) -> Tensor:
    """
    Rescale boxes (xyxy) from new_size to original_size
    """
    gain = torch.min(new_size[0] / original_size[0], new_size[1] / original_size[1])
    pad = (new_size[1] - original_size[1] * gain) / 2, (new_size[0] - original_size[0] * gain) / 2
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = (xmin - pad[0]) / gain
    xmax = (xmax - pad[0]) / gain
    ymin = (ymin - pad[1]) / gain
    ymax = (ymax - pad[1]) / gain

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def normalize_boxes(boxes: Tensor, original_size: List[int]) -> Tensor:
    height = torch.tensor(original_size[0], dtype=torch.float32, device=boxes.device)
    width = torch.tensor(original_size[1], dtype=torch.float32, device=boxes.device)
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin / width
    xmax = xmax / width
    ymin = ymin / height
    ymax = ymax / height
    boxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)
    # Convert xyxy to cxcywh
    return box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
