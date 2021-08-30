# Copyright 2020 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Module for creating L0 XLayer objects

L0: Other, mostly input and graph utility operations like Tuple, TupleGetItem
"""

from typing import Dict, List, Any, Optional

import math
import logging

import numpy as np

from yolort.shapes import TensorShape

from ..layer.xlayer import XLayer, BatchData, ConvData
from ..layer.xlayer_factory import xop_register_factory, xop_register
from ..xop_registry import xop_register_op_layout_transform, xop_register_op_transpose_transform

logger = logging.getLogger("pyxir")


###########
# Flatten #
###########


@xop_register("Flatten")
def batch_flatten(
    attrs: Dict[str, Any], in_xlayers: List[XLayer]
) -> Dict[str, List[int]]:
    """
    Return Batch Flatten registration information (shape)
    """
    assert len(in_xlayers) == 1, "Batch Flatten expects one input layer"
    flattened_shape = TensorShape(
        [list(in_xlayers[0].shapes)[0]] + [int(np.prod(list(in_xlayers[0].shapes)[1:]))]
    )
    return {"shape": flattened_shape}


#############
# BatchNorm #
#############


@xop_register_factory("BatchNorm")
def batch_norm(
    op_name: str,
    input_layer: XLayer,
    mean_layer: XLayer,
    variance_layer: XLayer,
    gamma_layer: XLayer,
    beta_layer: XLayer,
    axis: int,
    epsilon: float = 1e-5,
    **kwargs
) -> XLayer:
    """
    Create a batch normalization parameters layer

    Args:
        op_name (str): The name of this batch flatten layer operation
        input_layer (XLayer): The input layer to this batch flatten layer
    """

    bottoms = [input_layer.name]
    attrs = kwargs
    attrs.update(
        {
            "epsilon": epsilon,
            "axis": axis,
        }
    )
    mean, variance = mean_layer.data[0], variance_layer.data[0]
    gamma, beta = gamma_layer.data[0], beta_layer.data[0]
    assert mean.shape == variance.shape

    bn_data = BatchData(mu=mean, sigma_square=variance, gamma=gamma, beta=beta)

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=["BatchNorm"],
        shapes=input_layer.shapes[:],
        sizes=input_layer.sizes[:],
        data=bn_data,
        layer=[op_name],
        tops=[],
        bottoms=bottoms,
        attrs=attrs,
        targets=[],
    )

    return X


@xop_register_op_transpose_transform("BatchNorm")
def batchnorm_transpose_transform(X: XLayer, axes: List[int]) -> None:
    """
    Transform batch normalization layer with transpose according to
    provided axes
    """

    new_shape = TensorShape([X.shapes[i] for i in axes])

    X.shapes = new_shape
    X.attrs["axis"] = axes.index(X.attrs["axis"])


##########
# Conv2D #
##########


@xop_register_factory("Convolution")
def conv2d(
    op_name: str,
    input_layer: XLayer,
    weights_layer: XLayer,
    kernel_size: List[int],
    strides: List[int] = [1, 1],
    padding_hw: List[int] = [0, 0, 0, 0],
    dilation: List[int] = [1, 1],
    groups: int = 1,
    channels: Optional[int] = None,
    data_layout: str = "NCHW",
    kernel_layout: str = "OIHW",
    target_kernel_layout: str = "OIHW",
    **kwargs,
) -> XLayer:
    """
    Create a conv2d XLayer

    Args:
        op_name (str): The name of this conv2d layer operation
        input_layer (XLayer): The input layer to this conv2d layer
        weights_layer (XLayer): The weights input layer to this conv2d layer
        kernel_size (List[int]): The size of the kernel windows
        strides (List[int]): The convolution operation strides
        padding_hw (List[int]): The padding to be added before convolution
            operation, can be length 2 or 4: [pad_h, pad_w] or
            [pad_h_top, pad_h_bottom, pad_w_left, pad_w_right]
        dilation (List[int]): The dilation to be used for this convolution operation
        groups (int): Number of groups for grouped convolution.
        channels (Optional[int]): Number of output channels for this convolution.
        data_layout (str): The layout of the conv2d layer input (`NCHW` or `NHWC`)
        kernel_layout (str): The layout of the conv2d layer kernel (`OIHW`, `HWIO` or `OHWI`)
        target_kernel_layout (str): The target layout of the conv2d
            layer kernel (`OIHW`, `HWIO` or `OHWI`)
    """

    assert "Constant" in weights_layer.type

    assert len(kernel_size) == 2
    assert len(dilation) == 2
    assert len(strides) == 2
    assert len(padding_hw) in [2, 4]

    layout_idx = tuple([data_layout.index(e) for e in "NCHW"])
    layout_idx_transpose = tuple(["NCHW".index(e) for e in data_layout])
    B_idx, C_idx, H_idx, W_idx = layout_idx

    bottoms = [input_layer.name]

    logger.debug("-- Conv2D Kernel layout: {}".format(kernel_layout))
    logger.debug("-- Conv2D W shape: {}".format(weights_layer.data[0].shape))

    if len(kernel_layout) != 4 or sorted(kernel_layout) != ["H", "I", "O", "W"]:
        raise NotImplementedError(
            f"Unsupported kernel layout: {kernel_layout} for "
            f"convolution: {op_name}, should be a permutation of `OIHW`",
        )
    transpose_axes = tuple([kernel_layout.index(e) for e in target_kernel_layout])
    W = np.transpose(weights_layer.data[0], transpose_axes)
    kernel_layout_idx = tuple([target_kernel_layout.index(e) for e in "OIHW"])
    kO_idx, kI_idx, kH_idx, kW_idx = kernel_layout_idx

    if len(padding_hw) == 4:
        pad_ht, pad_hb, pad_wl, pad_wr = padding_hw
    elif len(padding_hw) == 2:
        pad_ht, pad_wl = padding_hw
        pad_hb, pad_wr = padding_hw
    else:
        raise ValueError("'padding_hw' argument should be a list of length 2 "
                         f"but got: {len(padding_hw)}")

    in_ch, out_ch = W.shape[kI_idx] * groups, W.shape[kO_idx]
    logger.debug("-- in_ch: {}, out_ch: {}".format(in_ch, out_ch))
    logger.debug("-- channels: {}".format(channels))

    assert channels is None or out_ch == channels
    channels = out_ch if channels is None else channels

    B = np.zeros([out_ch], dtype=np.float32)
    data = ConvData(W, B)

    # input layer is always in NCHW by design
    insize = [input_layer.shapes[H_idx], input_layer.shapes[W_idx]]
    batches = input_layer.shapes[0]
    logger.debug("-- in shape: {}".format(input_layer.shapes))
    logger.debug("-- padding (t,b,l,r): {}".format((pad_ht, pad_hb, pad_wl, pad_wr)))

    out_h = int(
        (insize[0] + pad_ht + pad_hb - dilation[0] * (kernel_size[0] - 1) - 1) / strides[0] + 1
    )
    out_w = int(
        (insize[1] + pad_wl + pad_wr - dilation[1] * (kernel_size[1] - 1) - 1) / strides[1] + 1
    )

    out_shape = TensorShape(
        [[batches, out_ch, out_h, out_w][i] for i in layout_idx_transpose]
    )

    padding_hh = [pad_ht, pad_hb]
    padding_ww = [pad_wl, pad_wr]

    if data_layout == "NCHW":
        granular_padding = [[0, 0], [0, 0], padding_hh, padding_ww]
    else:
        granular_padding = [[0, 0], padding_hh, padding_ww, [0, 0]]

    logger.debug("-- out shape: {}".format(out_shape))

    attrs = kwargs
    attrs.update(
        {
            "padding": granular_padding,
            "data_layout": data_layout,
            "kernel_layout": target_kernel_layout,
            "shape": out_shape.tolist(),
            "kernel_size": kernel_size,
            "strides": strides,
            "groups": groups,
            "dilation": dilation,
            "channels": [in_ch, out_ch],
        }
    )

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=["Convolution"],
        shapes=out_shape,
        sizes=out_shape.get_size(),
        data=data,
        layer=[op_name],
        tops=[],
        bottoms=bottoms,
        attrs=attrs,
        targets=[],
    )

    return X


@xop_register_op_layout_transform("Convolution")
def conv2d_layout_transform(X: XLayer, target_layout: str) -> None:
    """ Transform layout of provided XLayer to target layout """

    layout = X.attrs["data_layout"]
    axes_transpose = [layout.index(e) for e in target_layout]

    # TODO: strides, dilations

    X.attrs["padding"] = [X.attrs["padding"][i] for i in axes_transpose]
    X.attrs["data_layout"] = target_layout
    X.shapes[:] = TensorShape([X.shapes[i] for i in axes_transpose])


###################
# Conv2DTranspose #
###################

@xop_register_factory("Conv2DTranspose")
def conv2d_transpose(
    op_name: str,
    input_layer: XLayer,
    weights_layer: XLayer,
    kernel_size: List[int],
    strides: List[int] = [1, 1],
    padding_hw: List[int] = [0, 0, 0, 0],
    dilation: List[int] = [1, 1],
    groups: int = 1,
    channels: Optional[int] = None,
    data_layout: str = "NCHW",
    kernel_layout: str = "OIHW",
    target_kernel_layout: str = "OIHW",
    **kwargs
) -> XLayer:
    """
    Create a Conv2DTranspose XLayer

    Args:
        op_name (str): The name of this conv2d layer operation
        input_layer (XLayer): The input layer to this conv2d layer
        weights_layer (XLayer): The weights input layer to this conv2d layer
        kernel_size (List[int]): The size of the kernel windows
        strides (List[int]): The convolution operation strides
        padding (List[int]): The padding to be added before convolution operation,
            can be length 2 or 4: [pad_h, pad_w] or
            [pad_h_top, pad_h_bottom, pad_w_left, pad_w_right]
        dilation (List[int]): The dilation to be used for this convolution operation
        groups (int): Number of groups for grouped convolution.
        channels (Optional[int]): Number of output channels for this convolution.
        data_layout (str): The layout of the conv2d layer
            input (`NCHW` or `NHWC`)
        kernel_layout (str): The layout of the conv2d layer
            kernel (`OIHW`, `HWIO` or `OHWI`)
        target_kernel_layout (str): The target layout of the conv2d
            layer kernel (`OIHW`, `HWIO` or `OHWI`)
    """
    bottoms = [input_layer.name]

    layout_idx = tuple([data_layout.index(e) for e in "NCHW"])
    layout_idx_transpose = tuple(["NCHW".index(e) for e in data_layout])
    B_idx, C_idx, H_idx, W_idx = layout_idx

    logger.debug("-- Conv2DTranspose Kernel layout: {}".format(kernel_layout))
    logger.debug("-- Conv2DTranspose W shape: {}".format(weights_layer.data[0].shape))

    if len(kernel_layout) != 4 or sorted(kernel_layout) != ["H", "I", "O", "W"]:
        raise NotImplementedError(
            "Unsupported kernel layout: {} for"
            " convolution: {}, should be a permutation"
            " of `OIHW`".format(kernel_layout, op_name)
        )
    transpose_axes = tuple([kernel_layout.index(e) for e in target_kernel_layout])
    W = np.transpose(weights_layer.data[0], transpose_axes)
    kernel_layout_idx = tuple([target_kernel_layout.index(e) for e in "OIHW"])
    kO_idx, kI_idx, kH_idx, kW_idx = kernel_layout_idx

    assert len(padding_hw) in [2, 4]
    if len(padding_hw) == 4:
        pad_ht, pad_hb, pad_wl, pad_wr = padding_hw
    elif len(padding_hw) == 2:
        pad_ht, pad_wl = padding_hw
        pad_hb, pad_wr = padding_hw
    else:
        raise ValueError("'padding_hw' argument should be a list of "
                         f"length 2 but got: {len(padding_hw)}")

    in_ch, out_ch = W.shape[kI_idx] * groups, W.shape[kO_idx]
    logger.debug("-- in_ch: {}, out_ch: {}".format(in_ch, out_ch))
    logger.debug("-- channels: {}".format(channels))

    assert channels is None or out_ch == channels
    channels = out_ch if channels is None else channels

    B = np.zeros([out_ch], dtype=np.float32)
    data = ConvData(W, B)

    # Shape
    # Input layer is always in NCHW by design
    insize = [input_layer.shapes[2], input_layer.shapes[3]]
    batches = input_layer.shapes[0]
    logger.debug("{} {}".format(input_layer.shapes, in_ch))
    assert input_layer.shapes[C_idx] == in_ch

    if (
        (pad_ht + pad_hb) == (kernel_size[0] - strides[0])
        and abs(pad_ht - pad_hb) <= 1
        and (pad_wl + pad_wr) == (kernel_size[1] - strides[1])
        and abs(pad_wl - pad_wr) <= 1
    ):
        padding_type = "SAME"
    elif pad_ht == 0 and pad_wl == 0:
        padding_type = "VALID"
    else:
        raise NotImplementedError(
            "Unsupported padding for Conv2DTranspose Only Tensorflow padding 'SAME' and 'VALID' "
            f"are supported but got: {pad_ht, pad_hb, pad_wl, pad_wr} which does not"
            f" translate to 'SAME' == [pad_ht + pad_hb = {kernel_size[0] - strides[0]}, "
            f"pad_wl + pad_wr = {kernel_size[1] - strides[1]}] or 'VALID' == [0, 0]")

    if padding_type == "SAME":
        out_h = insize[0] * strides[0]
        out_w = insize[1] * strides[1]
    elif padding_type == "VALID":
        out_h = (insize[0] - 1) * strides[0] + kernel_size[0]
        out_w = (insize[1] - 1) * strides[1] + kernel_size[1]

    out_shape = TensorShape(
        [[batches, out_ch, out_h, out_w][i] for i in layout_idx_transpose]
    )

    padding = [[0, 0], [0, 0], [pad_ht, pad_hb], [pad_wl, pad_wr]]
    padding = [padding["NCHW".index(i)] for i in data_layout]

    attrs = kwargs
    attrs.update(
        {
            "padding": padding,
            "data_layout": data_layout,
            "kernel_layout": "OIHW",
            "shape": out_shape.tolist(),
            "kernel_size": kernel_size,
            "strides": strides,
            "groups": groups,
            "dilation": dilation,
            "channels": [in_ch, out_ch],
        }
    )

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=["Conv2DTranspose"],
        shapes=out_shape,
        sizes=out_shape.get_size(),
        data=data,
        layer=[op_name],
        tops=[],
        bottoms=bottoms,
        attrs=attrs,
        targets=[],
    )
    return X


@xop_register_op_layout_transform("Conv2DTranspose")
def conv2d_transpose_layout_transform(X: XLayer, target_layout: str) -> None:
    """
    Transform layout of provided XLayer to target layout
    """

    layout = X.attrs["data_layout"]
    axes_transpose = [layout.index(e) for e in target_layout]

    # TODO: strides, dilations

    X.attrs["padding"] = [X.attrs["padding"][i] for i in axes_transpose]
    X.attrs["data_layout"] = target_layout
    X.shapes = TensorShape([X.shapes[i] for i in axes_transpose])


##################
# Global Pooling #
##################


@xop_register_factory("GlobalPooling")
def global_pool2d(
    op_name: str,
    input_layer: XLayer,
    pool_type: str,
    layout: str,
    **kwargs: Any,
) -> XLayer:
    """
    Create a global pooling XLayer

    Args:
        op_name (str): The name of this pooling layer operation
        pool_type (str): Indicates which pooling operation to use (Max or Avg)
        layout (str): The layout of the pooling layer input (`NCHW` or `NHWC`)
        input_layer (XLayer): The input layer to this pooling layer
    """

    if pool_type not in ["Max", "Avg"]:
        raise NotImplementedError(
            "Invalid pooling type: {}, can either be"
            " `Max` or `Avg`.".format(pool_type)
        )

    # NCHW by design
    insize = [input_layer.shapes[2], input_layer.shapes[3]]
    batches, channels = input_layer.shapes[0], input_layer.shapes[1]

    strides = [1, 1]
    padding = [0, 0]
    pool_size = insize

    out_h, out_w = 1, 1

    attrs = kwargs
    attrs.update(
        {
            "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
            "insize": insize,
            "outsize": [out_h, out_w],
            "data_layout": layout,
            "strides": strides,
            "kernel_size": pool_size,
            "pool_type": pool_type,
        }
    )
    out_shape = TensorShape(
        [batches, channels, out_h, out_w]
        if layout == "NCHW"
        else [batches, out_h, out_w, channels]
    )

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=["Pooling"],
        shapes=out_shape,
        sizes=out_shape.get_size(),
        attrs=attrs,
        layer=[op_name],
        tops=[],
        bottoms=[input_layer.name],
        targets=[],
    )
    return X


#######
# Pad #
#######


@xop_register_factory("Pad")
def pad(
    op_name: str,
    input_layer: XLayer,
    padding: List[int],
    pad_value: float,
    **kwargs: Any,
) -> XLayer:
    """
    Create a padding XLayer

    Args:
        op_name (str): The name of this padding layer operation
        padding (List[List[int]]): The padding width to the edges of each axis
        pad_value (float): The padding value. Unsupported for now and always zero
        layout (str): The layout of the padding layer input (`NCHW` or `NHWC` supported)
        input_layer (XLayer): The input layer to this pooling layer
    """
    if pad_value != 0:
        raise NotImplementedError(f"Unsupported padding value: {pad_value}, "
                                  "only 0 is supported for now.")

    if not len(input_layer.shapes) == 4:
        raise NotImplementedError(
            "Padding layer only supported after layer in `NCHW` or `NHWC` format, "
            f"but found layer with {len(input_layer.shapes)} dims",
        )

    unpadded_dims = [[0, 0]] * len(input_layer.shapes[len(padding) :])
    padding = unpadded_dims + [list(pad) for pad in padding]

    shape = TensorShape([s + p[0] + p[1] for s, p in zip(input_layer.shapes, padding)])
    logger.debug("-- Pad shape: {}".format(shape))

    attrs = kwargs
    attrs.update({"padding": padding})

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=["Pad"],
        shapes=shape,
        sizes=shape.get_size(),
        attrs=attrs,
        layer=[op_name],
        tops=[],
        bottoms=[input_layer.name],
        targets=[],
    )
    return X


@xop_register_op_transpose_transform("Pad")
def padding_transpose_transform(X: XLayer, axes: List[int]) -> None:
    """
    Transform padding layer with transpose according to provided axes
    """

    new_shape = [X.shapes[i] for i in axes]

    X.shapes = TensorShape(new_shape)
    new_padding = [X.attrs["padding"][i] for i in axes]
    # X.data[:] = new_padding
    X.attrs["padding"] = new_padding


###########
# Pooling #
###########


@xop_register_factory("Pooling")
def pool2d(
    op_name: str,
    input_layer: XLayer,
    pool_type: str,
    pool_size: List[int],
    strides: List[int] = [1, 1],
    padding: List[int] = [0, 0, 0, 0],
    layout: str = "NCHW",
    ceil_mode: bool = False,
    count_include_pad: bool = False,
    **kwargs
) -> XLayer:
    """
    Create a pooling XLayer

    Args:
        op_name (str): The name of this pooling layer operation
        pool_type (str): Indicates which pooling operation to use (Max or Avg)
        pool_size (List[int]): The size of the pooling window
        strides (List[int]): The pooling operation strides
        padding (List[int]): The padding to be added before pooling
        layout (str): The layout of the pooling layer input (`NCHW` or `NHWC`)
        ceil_mode (bool): Whether to use ceiling or floor rounding while pooling
        count_include_pad (boolean): Whether to include padding to compute average
            (only for average pooling)
        input_layer (XLayer): The input layer to this pooling layer
    """
    if layout not in ["NCHW", "NHWC"]:
        raise ValueError(f"Unsupported layout: {layout}, supported layouts are NCHW and NHWC")

    if pool_type not in ["Max", "Avg"]:
        raise NotImplementedError(f"Invalid pooling type: {pool_type}, can either be `Max` or `Avg`.")

    def valid(x, k, p1, p2, s):
        return math.floor((x + p1 + p2 - k) / s) + 1

    def full(x, k, p1, p2, s):
        return math.ceil((x + p1 + p2 - k) / s) + 1

    # TODO: this is very similar as for NNVM operators -> merge
    if len(padding) == 4:
        # top bottom left right = h_before h_after w_before w_after
        full_paddings = [
            [0, 0],
            [0, 0],
            [padding[0], padding[2]],
            [padding[1], padding[3]],
        ]
    elif len(padding) == 2:
        full_paddings = [
            [0, 0],
            [0, 0],
            [padding[0], padding[0]],
            [padding[1], padding[1]],
        ]
    elif len(padding) == 1:
        full_paddings = [[0, 0], [0, 0], [padding, padding], [padding, padding]]
    else:
        raise ValueError(
            "Invalid padding size passed by Relay operator, "
            f" Sizes of 1, 2 and 4 are supported but not {len(padding)}"
        )

    padding = [
        min(full_paddings[2][0], full_paddings[2][1]),
        min(full_paddings[3][0], full_paddings[3][1]),
    ]

    if layout == "NCHW":
        insize = [input_layer.shapes[2], input_layer.shapes[3]]
        batches, channels = input_layer.shapes[0], input_layer.shapes[1]
    else:
        # NHWC
        insize = [input_layer.shapes[1], input_layer.shapes[2]]
        batches, channels = input_layer.shapes[0], input_layer.shapes[3]
        full_paddings = [full_paddings[i] for i in [0, 2, 3, 1]]

    outsize = []
    calc_func = full if ceil_mode else valid

    outsize = [
        calc_func(
            insize[1],
            pool_size[1],
            full_paddings[3][0],
            full_paddings[3][1],
            strides[1],
        ),
        calc_func(
            insize[0],
            pool_size[0],
            full_paddings[2][0],
            full_paddings[2][1],
            strides[0],
        ),
    ]

    attrs = kwargs
    attrs.update(
        {
            "type": pool_type,
            "padding": full_paddings,
            "strides": strides,  # HW
            "kernel_size": pool_size,  # HW
            "insize": insize,  # HW
            "outsize": [outsize[1], outsize[0]],  # HW
            "data_layout": layout,
            "pool_type": pool_type,
        }
    )
    if pool_type == "Avg":
        attrs["count_include_pad"] = count_include_pad

    out_h, out_w = outsize[1], outsize[0]
    out_shape = TensorShape(
        [batches, channels, out_h, out_w]
        if layout == "NCHW"
        else [batches, out_h, out_w, channels]
    )

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=["Pooling"],
        shapes=out_shape,
        sizes=out_shape.get_size(),
        attrs=attrs,
        layer=[op_name],
        tops=[],
        bottoms=[input_layer.name],
        targets=[],
    )

    return X


@xop_register_op_layout_transform("Pooling")
def pooling_layout_transform(X: XLayer, target_layout: str) -> None:
    """ Transform layout of provided XLayer to target layout """

    layout = X.attrs["data_layout"]
    axes_transpose = [layout.index(e) for e in target_layout]

    # TODO: strides, dilations
    X.attrs["padding"] = [X.attrs["padding"][i] for i in axes_transpose]
    X.attrs["data_layout"] = target_layout
    X.shapes = TensorShape([X.shapes[i] for i in axes_transpose])


################
# Upsampling2D #
################


@xop_register("Upsampling2D")
def upsampling2d(
    attrs: Dict[str, Any], in_xlayers: List[XLayer]
) -> Dict[str, List[int]]:
    """
    Create 2D Upsampling XLayer

    Scale input tensor along the height and weight axes with provided
    scaling factor

    Attributes:
        scale_h (float): The scaling factor for the height dimension
        scale_w (float): The scaling factor for the weight dimension
        data_layout (str): The input tensor layout (combination of NCHW)
        method (str): The scale method to be used (nearest_neighbor, bilinear)
        align_corners (bool): Whether to keep corners in proper place
    """

    assert len(in_xlayers) == 1
    assert "scale_h" in attrs
    assert "scale_w" in attrs
    assert "data_layout" in attrs
    assert "method" in attrs
    if "align_corners" not in attrs:
        attrs["align_corners"] = False

    scale_h = attrs["scale_h"]
    scale_w = attrs["scale_w"]

    layout = attrs["data_layout"]
    assert sorted(layout) == ["C", "H", "N", "W"]

    h_idx = layout.index("H")
    w_idx = layout.index("W")

    shape = in_xlayers[0].shapes[:]
    shape[h_idx] = int(shape[h_idx] * scale_h)
    shape[w_idx] = int(shape[w_idx] * scale_w)

    return {"shape": shape}


@xop_register_op_layout_transform("Upsampling2D")
def upsampling2d_layout_transform(X: XLayer, target_layout: str) -> None:
    """
    Transform layout of provided Upsampling2D XLayer to target layout
    """

    layout = X.attrs["data_layout"]
    axes_transpose = [layout.index(e) for e in target_layout]

    X.attrs["data_layout"] = target_layout
    X.shapes[:] = TensorShape([X.shapes[i] for i in axes_transpose])
