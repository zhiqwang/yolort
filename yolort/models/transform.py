# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import math
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import box_convert


class NestedTensor:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
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


class YOLOTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization
            YOLOv5 use (0, 1) as the default mean and std, so we just need to rescale
            the image manually from uint8_t [0, 255] to float [0, 1] here.
        - input / target resizing to match min_size / max_size
            When fixed_size is set, Different images can have different sizes but
            they will be resized to a fixed size before passing it to the backbone.

    It returns a NestedTensor for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(
        self,
        min_size: int,
        max_size: int,
        fixed_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            min_size (int) : minimum size of the image to be rescaled.
            max_size (int) : maximum size of the image to be rescaled.
            fixed_size (Optional[Tuple[int, int]]): Whether to specify and use the input size.
        """
        super().__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.fixed_size = fixed_size

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
                    data[k] = v.to(device)
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
        images = nested_tensor_from_tensor_list(images)
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
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])

        image, target = _resize_image_and_masks(image, size, float(self.max_size), self.fixed_size, target)

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = normalize_boxes(bbox, (h, w))
        target["boxes"] = bbox

        return image, target

    def postprocess(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:

        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes

        return result


def nested_tensor_from_tensor_list(tensor_list: List[Tensor], size_divisible: int = 32):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list, size_divisible)

        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(tensor_list)] + max_size
        tensor_batched = tensor_list[0].new_full(batch_shape, 0)
        for img, pad_img in zip(tensor_list, tensor_batched):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    else:
        raise ValueError("not supported")
    return tensor_batched


def _max_by_axis(the_list: List[List[int]]) -> List[int]:
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor], size_divisible: int = 32) -> Tensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(
            torch.int64
        )
        max_size.append(max_size_i)
    stride = size_divisible
    max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
    max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []

    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = F.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

    tensor = torch.stack(padded_imgs)

    return tensor


@torch.jit.unused
def _get_shape_onnx(image: Tensor) -> Tensor:
    from torch.onnx import operators

    return operators.shape_as_tensor(image)[-2:]


@torch.jit.unused
def _fake_cast_onnx(v: Tensor) -> float:
    # ONNX requires a tensor but here we fake its type for JIT.
    return v


def _resize_image_and_masks(
    image: Tensor,
    self_min_size: float,
    self_max_size: float,
    fixed_size: Optional[Tuple[int, int]] = None,
    target: Optional[Dict[str, Tensor]] = None,
) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    """
    Resize the image and its targets
    """
    if torchvision._is_tracing():
        im_shape = _get_shape_onnx(image)
    else:
        im_shape = torch.tensor(image.shape[-2:])

    size: Optional[List[int]] = None
    scale_factor: Optional[float] = None
    recompute_scale_factor: Optional[bool] = None
    if fixed_size is not None:
        size = [fixed_size[1], fixed_size[0]]
    else:
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(self_min_size / min_size, self_max_size / max_size)

        if torchvision._is_tracing():
            scale_factor = _fake_cast_onnx(scale)
        else:
            scale_factor = scale.item()
        recompute_scale_factor = True

    image = F.interpolate(
        image[None],
        size=size,
        scale_factor=scale_factor,
        mode="bilinear",
        recompute_scale_factor=recompute_scale_factor,
        align_corners=False,
    )[0]

    if target is None:
        return image, target

    if "masks" in target:
        mask = target["masks"]
        mask = F.interpolate(
            mask[:, None].float(),
            size=size,
            scale_factor=scale_factor,
            recompute_scale_factor=recompute_scale_factor,
        )[:, 0].byte()
        target["masks"] = mask
    return image, target


def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
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
