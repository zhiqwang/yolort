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

from typing import Dict, List, Any

import logging

from yolort.shapes import TensorShape, TupleShape

from ..layer.xlayer import XLayer
from ..layer.xlayer_factory import xop_register_factory, xop_register
from ..xop_registry import xop_register_op_transpose_transform

logger = logging.getLogger("pyxir")


########
# Cast #
########


@xop_register('Cast')
def cast(attrs: Dict[str, Any], in_xlayers: List[XLayer]) -> XLayer:
    """
    Cast input tensor to the provided data type
    """
    assert len(in_xlayers) == 1
    assert 'dtype' in attrs

    shape = in_xlayers[0].shapes[:]
    return {'shape': shape}


@xop_register_op_transpose_transform('Cast')
def cast_transpose_transform(X: XLayer, axes: List[int]) -> None:
    """ Transform Cast layer with transpose according to provided axes """
    new_shape = TensorShape([X.shapes[i] for i in axes])
    X.shapes = new_shape


########
# Clip #
########

@xop_register_factory('Clip')
def clip(
    op_name: str,
    input_layer: XLayer,
    a_min: float,
    a_max: float,
    **kwargs,
) -> XLayer:
    """
    Clip the outputs of the previous layer between a_min and a_max

    Arguments
    ---------
    op_name: str
        The name of this elementwise addition operation
    input_layer: XLayer
        The input layer
    a_min: float
        The minimum for the clipping operation
    a_max: float
        The maximum for the clipping operation
    """

    attrs = kwargs

    shape = input_layer.shapes[:]
    bottoms = [input_layer.name]

    # TODO: check here for ReLU6 or after?

    if a_max == 6.0 and a_min == 0.0:
        X = XLayer()
        X = X._replace(
            name=op_name,
            type=['ReLU6'],
            shapes=shape,
            sizes=shape.get_size(),
            layer=[op_name],
            tops=[],
            bottoms=bottoms,
            attrs=attrs,
            targets=[]
        )
    else:
        attrs.update({
            'a_min': a_min,
            'a_max': a_max
        })

        X = XLayer()
        X = X._replace(
            name=op_name,
            type=['Clip'],
            shapes=shape,
            sizes=shape.get_size(),
            layer=[op_name],  # was [node]
            tops=[],
            bottoms=bottoms,
            attrs=attrs,
            targets=[]
        )

    return X


@xop_register_op_transpose_transform('Clip')
def clip_transpose_transform(X: XLayer, axes: List[int]) -> None:
    """
    Transform clip layer with transpose according to provided axes
    """
    new_shape = TensorShape([X.shapes[i] for i in axes])
    X.shapes = new_shape


@xop_register_op_transpose_transform('ReLU6')
def relu6_transpose_transform(X: XLayer, axes: List[int]) -> None:
    """
    Transform ReLU6 layer with transpose according to provided axes
    """
    new_shape = TensorShape([X.shapes[i] for i in axes])
    X.shapes = new_shape


#############
# LeakyReLU #
#############


@xop_register('LeakyReLU')
def leaky_relu(attrs: Dict[str, Any], in_xlayers: List[XLayer]) -> XLayer:
    """
    Leaky ReLU operation

    Returns Leaky ReLU registration information (shape)
    """
    assert len(in_xlayers) == 1
    assert 'alpha' in attrs

    shape = in_xlayers[0].shapes[:]
    return {'shape': shape}


@xop_register_op_transpose_transform('LeakyReLU')
def leaky_relu_transpose_transform(X: XLayer, axes: List[int]) -> None:
    """
    Transform LeakyReLU layer with transpose according to provided axes
    """
    new_shape = TensorShape([X.shapes[i] for i in axes])
    X.shapes = new_shape


#########
# pReLU #
#########


@xop_register_factory('pReLU')
def prelu(
    op_name: str,
    input_layer: XLayer,
    alpha: float,
    axis: int,
    **kwargs,
) -> XLayer:
    """
    Create a parametric ReLU Xlayer

    Args:
        op_name (str): The name of this relu layer operation
        alpha (float): The slope coefficient for negative input
        axis (int): The axis of the channel
        input_layer (XLayer): The input layer to this relu layer
    """

    bottoms = [input_layer.name]

    attrs = kwargs
    attrs.update({
        'alpha': alpha,
        'axis': axis
    })

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=['pReLU'],
        shapes=input_layer.shapes[:],
        sizes=input_layer.sizes[:],
        layer=[op_name],
        tops=[],
        bottoms=bottoms,
        attrs=attrs,
        targets=[]
    )

    return X


@xop_register_op_transpose_transform('pReLU')
def prelu_transpose_transform(X: XLayer, axes: List[int]) -> None:
    """
    Transform pReLU layer with transpose according to provided axes
    """
    new_shape = TensorShape([X.shapes[i] for i in axes])
    X.shapes = new_shape


###########
# Reshape #
###########

@xop_register_factory('Reshape')
def reshape(
    op_name: str,
    input_layer: XLayer,
    newshape: List[int],
    **kwargs: Any,
) -> XLayer:
    """
    Create a Reshape XLayer

    Args:
        op_name (str): The name of this constant layer
        input_layer (XLayer): The input layer to this scaling layer
        newshape (List[int]): The shape that the input should be reshaped to
    """
    if 'Constant' in input_layer.type:
        X = input_layer._replace(
            shapes=TensorShape(newshape),
            data=[input_layer.data[0].reshape(newshape)]
        )

        return X
    else:
        bottoms = [input_layer.name]

        attrs = kwargs
        attrs.update({
            'shape': newshape
        })

        X = XLayer()
        X = X._replace(
            name=op_name,
            type=['Reshape'],
            shapes=TensorShape(newshape),
            sizes=input_layer.sizes,
            # data=newshape,
            layer=[op_name],
            tops=[],
            bottoms=bottoms,
            attrs=attrs,
            targets=[]
        )

        return X


#########
# Split #
#########

@xop_register('Split')
def split(
    attrs: Dict[str, Any],
    in_xlayers: List[XLayer],
) -> Dict[str, List[int]]:
    """
    Registration of 'split' operator and shape computation.

    Split the input tensor along specified axis by the provided indices

    Attributes:
        Axis (int): The axis along which to do the split
        Indices (int or Tuple[int]): If indices attribute is an integer,
            split the input tensor in tensors of equal size along the given axis
            If indices is a tuple of (sorted) integers, the entries specify
            the indices where the tensor should be split along the given axis

    Returns:
        xinfo (Dict): A dictionary containing necessary XOp information (shape)
    """
    # Some operation checks
    assert len(in_xlayers) == 1
    assert 'axis' in attrs
    assert 'indices' in attrs

    indices = attrs['indices']
    axis = attrs['axis']

    inshape = in_xlayers[0].shapes[:]
    assert isinstance(inshape, TensorShape)
    axes_size = inshape[axis]

    new_shape = TupleShape([])

    if isinstance(indices, int):
        if axes_size % indices != 0:
            raise ValueError("Split operation has integer indices attribute"
                             f"{indices} but this is not a divisor of the dimension "
                             f"of the input tensor with shape {inshape} along axis: {axis}")
        new_dim_size = axes_size // indices
        for _ in range(0, indices):
            shape = inshape[:]
            shape[axis] = new_dim_size
            new_shape.append(shape)
    else:
        prev = 0
        for i in indices:
            shape = inshape[:]
            shape[axis] = i - prev
            new_shape.append(shape)
            prev = i

        shape = inshape[:]
        shape[axis] = axes_size - prev
        new_shape.append(shape)

    return {'shape': new_shape}


###########
# Squeeze #
###########

@xop_register_factory('Squeeze')
def squeeze(op_name: str, input_layer: XLayer, axis: List[int], **kwargs) -> XLayer:
    """
    Create a Squeeze XLayer

    Args:
        op_name (str): The name of this constant layer
        axis (List[int]): The set of axes to squeeze
        input_layer (XLayer): The input layer to this scaling layer
    """
    assert(isinstance(axis, list) or axis is None)

    bottoms = [input_layer.name]

    attrs = kwargs
    attrs.update({
        'axis': axis
    })
    in_shapes = input_layer.shapes[:]
    if axis is None:
        shape = TensorShape([dim for dim in in_shapes if dim != 1])
    else:
        shape = TensorShape([dim for i, dim in enumerate(in_shapes)
                             if i not in axis])

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=['Squeeze'],
        shapes=shape,
        sizes=shape.get_size(),
        layer=[op_name],
        tops=[],
        bottoms=bottoms,
        attrs=attrs,
        targets=[]
    )
    return X


########
# Take #
########

@xop_register('Take')
def take(
    attrs: Dict[str, Any],
    in_xlayers: List[XLayer],
) -> Dict[str, List[int]]:
    """
    Slice input tensor according to specified axis and indices

    Args:
        op_name (str): the name of the operation
        in_xlayers (List[XLayer]): the input layers (input tensor and indices)
        axis (int): the axis across which to slice the input tensor
        mode (str): the slice mode ('clip')

    Returns:
        Take registration information (shape)
    """
    assert len(in_xlayers) == 2, "Take layer expects two input layers"
    assert 'axis' in attrs
    if 'mode' in attrs:
        attrs['mode'] = 'clip'
    axis = attrs['axis']

    in_shape = in_xlayers[0].shapes[:]
    indices_shape = in_xlayers[1].shapes[:]
    assert all([s <= in_shape[axis] for s in indices_shape])

    shape = TensorShape(in_shape[:axis] + indices_shape + in_shape[axis + 1:])
    return {'shape': shape}


#############
# Transpose #
#############

@xop_register_factory('Transpose')
def transpose(
    op_name: str,
    input_layer: XLayer,
    axes: List[int],
    internal=0,
    **kwargs,
):
    """
    Create a Transpose XLayer

    Args:
        op_name (str): The name of this constant layer
        axes (List[int]): The axes defining how to do the transpose
        input_layer (XLayer): The input layer to this scaling layer
    """
    bottoms = [input_layer.name]

    new_shape = TensorShape([input_layer.shapes[i] for i in axes])

    attrs = kwargs
    attrs.update({
        'axes': axes
    })

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=['Transpose'],
        shapes=new_shape,
        sizes=new_shape.get_size(),
        layer=[op_name],
        tops=[],
        bottoms=bottoms,
        attrs=attrs,
        internal=internal,
        targets=[]
    )
    return X
