# Copyright (c) 2020, yolort team. All rights reserved.

import math
import functools
import inspect
import warnings
from typing import Any, Dict, Optional, TypeVar, Callable, Tuple, Union

import torch
from torch import nn, Tensor
from yolort.utils.factory import sequence_to_str

from ._api import WeightsEnum


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def encode_single(reference_boxes: Tensor, anchors: Tensor) -> Tensor:
    """
    Encode a set of anchors with respect to some reference boxes

    Args:
        reference_boxes (Tensor): reference boxes
        anchors_tuple (Tensor): boxes to be encoded
    """
    reference_boxes = torch.sigmoid(reference_boxes)

    pred_xy = reference_boxes[:, :2] * 2.0 - 0.5
    pred_wh = (reference_boxes[:, 2:4] * 2) ** 2 * anchors
    pred_boxes = torch.cat((pred_xy, pred_wh), 1)

    return pred_boxes


def decode_single(
    rel_codes: Tensor,
    grid: Tensor,
    shift: Tensor,
    stride: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    From a set of original boxes and encoded relative box offsets,
    get the decoded boxes.

    Args:
        rel_codes (Tensor): Encoded boxes
        grid (Tensor): Anchor grids
        shift (Tensor): Anchor shifts
        stride (int): Stride
    """
    pred_xy = (rel_codes[..., 0:2] * 2.0 - 0.5 + grid) * stride
    pred_wh = (rel_codes[..., 2:4] * 2.0) ** 2 * shift

    return pred_xy, pred_wh


def bbox_iou(box1: Tensor, box2: Tensor, x1y1x2y2: bool = True, eps: float = 1e-7):
    """
    Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    """
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    # convex (smallest enclosing box) width
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    # convex height
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    # Complete IoU https://arxiv.org/abs/1911.08287v1
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
    ) / 4  # center distance squared

    # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)  # CIoU


def smooth_binary_cross_entropy(eps: float = 0.1) -> Tuple[float, float]:
    # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing binary cross entropy targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(),
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        # must be nn.BCEWithLogitsLoss()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        # required to apply FL to each element
        self.loss_fcn.reduction = "none"

    def forward(self, pred, logit):
        loss = self.loss_fcn(pred, logit)
        # p_t = torch.exp(-loss)
        # non-zero power for gradient stability
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma

        # TF implementation
        # https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = logit * pred_prob + (1 - logit) * (1 - pred_prob)
        alpha_factor = logit * self.alpha + (1 - logit) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        # 'none'
        return loss


D = TypeVar("D")


def kwonly_to_pos_or_kw(fn: Callable[..., D]) -> Callable[..., D]:
    """
    Decorates a function that uses keyword only parameters to also allow them being
    passed as positionals.

    For example, consider the use case of changing the signature of ``old_fn`` into
    the one from ``new_fn``:

    .. code::

        def old_fn(foo, bar, baz=None):
            ...

        def new_fn(foo, *, bar, baz=None):
            ...

    Calling ``old_fn("foo", "bar, "baz")`` was valid, but the same call is no longer valid
    with ``new_fn``. To keep BC and at the same time warn the user of the deprecation, this
    decorator can be used:

    .. code::

        @kwonly_to_pos_or_kw
        def new_fn(foo, *, bar, baz=None):
            ...

        new_fn("foo", "bar, "baz")
    """
    params = inspect.signature(fn).parameters

    try:
        keyword_only_start_idx = next(
            idx for idx, param in enumerate(params.values()) if param.kind == param.KEYWORD_ONLY
        )
    except StopIteration:
        raise TypeError(f"Found no keyword-only parameter on function '{fn.__name__}'") from None

    keyword_only_params = tuple(inspect.signature(fn).parameters)[keyword_only_start_idx:]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> D:
        args, keyword_only_args = args[:keyword_only_start_idx], args[keyword_only_start_idx:]
        if keyword_only_args:
            keyword_only_kwargs = dict(zip(keyword_only_params, keyword_only_args))
            warnings.warn(
                f"Using {sequence_to_str(tuple(keyword_only_kwargs.keys()), separate_last='and ')} "
                "as positional parameter(s) is deprecated. Please use keyword parameter(s) instead."
            )
            kwargs.update(keyword_only_kwargs)

        return fn(*args, **kwargs)

    return wrapper


W = TypeVar("W", bound=WeightsEnum)
M = TypeVar("M", bound=nn.Module)
V = TypeVar("V")


def handle_legacy_interface(
    **weights: Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]],
):
    """Decorates a model builder with the new interface to make it compatible with the old.

    In particular this handles two things:

    1. Allows positional parameters again, but emits a deprecation warning in case they
        are used. See :func:`torchvision.prototype.utils._internal.kwonly_to_pos_or_kw` for details.
    2. Handles the default value change from ``pretrained=False`` to ``weights=None``
        and ``pretrained=True`` to ``weights=Weights`` and emits a deprecation warning
        with instructions for the new interface.

    Args:
        **weights (Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]): Deprecated
            parameter name and default value for the legacy ``pretrained=True``. The default value can
            be a callable in which case it will be called with a dictionary of the keyword arguments. The
            only key that is guaranteed to be in the dictionary is the deprecated parameter name passed
            as first element in the tuple. All other parameters should be accessed with :meth:`~dict.get`.
    """

    def outer_wrapper(builder: Callable[..., M]) -> Callable[..., M]:
        @kwonly_to_pos_or_kw
        @functools.wraps(builder)
        def inner_wrapper(*args: Any, **kwargs: Any) -> M:
            for weights_param, (pretrained_param, default) in weights.items():  # type: ignore[union-attr]
                # If neither the weights nor the pretrained parameter as passed, or the weights argument
                # already use the new style arguments, there is nothing to do. Note that we cannot use
                # `None` as sentinel for the weight argument, since it is a valid value.
                sentinel = object()
                weights_arg = kwargs.get(weights_param, sentinel)
                if (
                    (weights_param not in kwargs and pretrained_param not in kwargs)
                    or isinstance(weights_arg, WeightsEnum)
                    or (isinstance(weights_arg, str) and weights_arg != "legacy")
                    or weights_arg is None
                ):
                    continue

                # If the pretrained parameter was passed as positional argument, it is now mapped to
                # `kwargs[weights_param]`. This happens because the @kwonly_to_pos_or_kw decorator uses
                # the current signature to infer the names of positionally passed arguments and thus
                # has no knowledge that there used to be a pretrained parameter.
                pretrained_positional = weights_arg is not sentinel
                if pretrained_positional:
                    # We put the pretrained argument under its legacy name in the keyword argument
                    # dictionary to have a unified access to the value if the default value is a callable.
                    kwargs[pretrained_param] = pretrained_arg = kwargs.pop(weights_param)
                else:
                    pretrained_arg = kwargs[pretrained_param]

                if pretrained_arg:
                    default_weights_arg = default(kwargs) if callable(default) else default
                    if not isinstance(default_weights_arg, WeightsEnum):
                        raise ValueError(f"No weights available for model {builder.__name__}")
                else:
                    default_weights_arg = None

                if not pretrained_positional:
                    warnings.warn(
                        f"The parameter '{pretrained_param}' is deprecated, please use "
                        f"'{weights_param}' instead."
                    )

                msg = (
                    f"Arguments other than a weight enum or `None` for '{weights_param}' are "
                    "deprecated. The current behavior is equivalent to "
                    f"passing `{weights_param}={default_weights_arg}`."
                )
                if pretrained_arg:
                    msg = (
                        f"{msg} You can also use `{weights_param}={type(default_weights_arg).__name__}."
                        "DEFAULT` to get the most up-to-date weights."
                    )
                warnings.warn(msg)

                del kwargs[pretrained_param]
                kwargs[weights_param] = default_weights_arg

            return builder(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(
                f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead."
            )
    else:
        kwargs[param] = new_value


def _ovewrite_value_param(param: Optional[V], new_value: V) -> V:
    if param is not None:
        if param != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {param} instead.")
    return new_value
