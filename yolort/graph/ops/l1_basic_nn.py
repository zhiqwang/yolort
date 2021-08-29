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
import numpy as np

from ...shapes import TensorShape, get_numpy_broadcasted_shape
from ..layer.xlayer import XLayer, ConvData, ScaleData
from ..layer.xlayer_factory import xop_register_factory, xop_register
from ..xop_registry import xop_register_op_transpose_transform

logger = logging.getLogger("pyxir")


#######
# Add #
#######

@xop_register('Add')
def add(attrs: Dict[str, Any], in_xlayers: List[XLayer]):
    """
    Return numpy-style addition layer registration information (shape)

    NOTE: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    """

    assert len(in_xlayers) == 2

    lX, rX = in_xlayers
    if len(lX.shapes) >= len(rX.shapes):
        lshape = lX.shapes[:]
        rshape = [None] * (len(lX.shapes) - len(rX.shapes)) + rX.shapes[:]
    else:
        rshape = rX.shapes[:]
        lshape = [None] * (len(rX.shapes) - len(lX.shapes)) + lX.shapes[:]

    assert len(lshape) == len(rshape)

    reversed_shape = []
    for ls, rs in zip(reversed(lshape), reversed(rshape)):
        if ls == rs or ls in [1, None] or rs in [1, None]:
            if ls is None:
                reversed_shape.append(rs)
            elif rs is None:
                reversed_shape.append(ls)
            else:
                reversed_shape.append(max(ls, rs))
        else:
            raise ValueError("Invalid shapes for broadcasted additions: "
                             f"{lX.shapes} and {rX.shapes}")

    shape = TensorShape(list(reversed(reversed_shape)))

    return {'shape': shape}


###########
# BiasAdd #
###########

@xop_register_factory('BiasAdd')
def bias_add(op_name: str, input_layer: XLayer, bias_layer: XLayer, axis: int, **kwargs):
    """
    Create an XLayer for adding a bias to the input layer

    Args:
        op_name (str): The name of this pooling layer operation
        axis (int): The axis on which to add the bias to the input.
            If None, do broadcast add
        input_layer (XLayer): The input data layer to this bias add layer
        bias_layer (XLayer): The input bias layer to this bias add layer
    """

    bottoms = [input_layer.name]
    attrs = kwargs
    attrs.update({
        'axis': axis
    })

    logger.debug("--bias_add shape: {}".format(input_layer.shapes[:]))

    X = XLayer()
    X = X._replace(
        shapes=input_layer.shapes[:],
        sizes=input_layer.sizes[:],
        name=op_name,
        type=['BiasAdd'],
        data=[bias_layer.data[0]],
        layer=[op_name],
        tops=[],
        bottoms=bottoms,
        attrs=attrs,
        targets=[])

    return X


@xop_register_op_transpose_transform('BiasAdd')
def bias_add_transpose_transform(X: XLayer, axes: List[int]):
    """ Transform bias addition layer with transpose according to
        provided axes """

    new_shape = TensorShape([X.shapes[i] for i in axes])

    X.shapes[:] = new_shape
    X.attrs['axis'] = axes.index(X.attrs['axis'])


###############
# Concatenate #
###############

@xop_register_factory('Concat')
def concat(op_name: str, input_layers: List[XLayer], axis: int, **kwargs):
    """
    Create an concatenate parameters layer for concatenating a list of
    input layers

    Args:
        op_name (str): The name of this elementwise addition operation
        axis (int): The axis over which to concatenate the input layers
        input_layers (List[XLayer]): The input layers to be concatenated
    """
    bottoms = [input_layer.name for input_layer in input_layers]
    if axis < 0:
        axis = axis + len(input_layers[0].shapes[:])

    # Check concatenation inputs
    assert len(set([len(il.shapes) for il in input_layers])) == 1
    for i in range(len(list(input_layers[0].shapes))):
        # Either i is the axis over which to concatenate or this dimension
        #   in the shape of each input layer is the same
        check = set([il.shapes[i] for il in input_layers])
        # TODO workaround for concatenating when batch is -1 and some other constant
        if len(check) > 1 and -1 in check:
            check.remove(-1)
        assert i == axis or len(check) == 1, "i: {0}, axis: {1}, check: {2}".format(i, axis, check)

    shape = input_layers[0].shapes[:]
    shape[axis] = sum([il.shapes[axis] for il in input_layers])

    attrs = kwargs
    attrs.update({
        'axis': axis
    })

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=['Concat'],
        shapes=shape,
        sizes=shape.get_size(),
        layer=[op_name],
        tops=[],
        bottoms=bottoms,
        attrs=attrs,
        targets=[]
    )

    return X


@xop_register_op_transpose_transform('Concat')
def concat_transpose_transform(X: XLayer, axes: List[int]):
    """
    Transform concat layer with transpose according to provided axes
    """

    new_shape = TensorShape([X.shapes[i] for i in axes])

    X.shapes = new_shape
    X.attrs['axis'] = axes.index(X.attrs['axis'])


###########
# Eltwise #
###########

@xop_register_factory('Eltwise')
def eltwise(op_name: str, lhs_layer: XLayer, rhs_layer: XLayer, **kwargs):
    """
    Create an elementwise addition parameters layer for adding two input
    layers. The input layers should have the same shape

    Args:
        op_name (str): The name of this elementwise addition operation
        lhs_layer (XLayer): The left hand side input layer
        scale_data_layer (XLayer): The right hand side input layer
    """

    bottoms = [lhs_layer.name, rhs_layer.name]

    # TODO
    attrs = kwargs
    attrs.update({
        'op': 'Add'
    })

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=['Eltwise'],
        shapes=lhs_layer.shapes[:],
        sizes=lhs_layer.sizes[:],
        layer=[op_name],  # was [node]
        tops=[],
        bottoms=bottoms,
        attrs=attrs,
        targets=[]
    )

    return X


@xop_register_op_transpose_transform('Eltwise')
def eltwise_transpose_transform(X: XLayer, axes: List[int]):
    """ Transform elementwise layer with transpose according to
        provided axes """
    new_shape = TensorShape([X.shapes[i] for i in axes])
    X.shapes[:] = new_shape


#########
# Dense #
#########

@xop_register_factory('Dense')
def dense(
    op_name: str,
    input_layer: XLayer,
    weights_layer: XLayer,
    units: int,
    data_layout: str = 'NC',
    kernel_layout: str = 'OI',
    **kwargs,
):
    """
    Create a dense parameters layer

    Args:
        op_name (str): The name of this dense layer operation
        units (int): Number of output units of this dense layer
        input_layer (XLayer): The input layer to this dense layer
        weights_layer (XLayer): The weights input layer to this dense layer
        data_layout (str): the input data layout (NC, CN)
        kernel_layout (str): the layout of the kernel (OI, IO)
    """

    if 'Constant' not in weights_layer.type:
        raise ValueError("Dense layer is expecting a 'Constant' weights layer"
                         f" as weights input but got layer of type: {weights_layer.type[0]}")

    bottoms = [input_layer.name]

    bias = np.zeros([units])

    data = ConvData(weights_layer.data[0], bias)

    attrs = kwargs
    attrs.update({
        'W_shape': weights_layer.shapes.tolist(),
        'units': units,
        'data_layout': data_layout,
        'kernel_layout': kernel_layout
    })

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=['Dense'],
        shapes=TensorShape([input_layer.shapes[0], units]),
        sizes=[units],
        data=data,
        layer=[op_name],
        tops=[],
        bottoms=bottoms,
        attrs=attrs,
        targets=[])

    return X


##########
# Divide #
##########

@xop_register('Divide')
def divide(attrs: Dict, in_xlayers: List[XLayer]):
    """
    Return numpy-style division layer registration information (shape)

    NOTE: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    """

    assert len(in_xlayers) == 2
    lX, rX = in_xlayers
    l_shape = lX.shapes[:]
    r_shape = rX.shapes[:]

    broadcast_shape = get_numpy_broadcasted_shape(l_shape, r_shape)
    shape = TensorShape(broadcast_shape)

    return {'shape': shape}


###########
# Dropout #
###########

@xop_register_factory('Dropout')
def dropout(op_name: str, input_layer: XLayer, rate: float, **kwargs):
    """
    Create a dropout XLayer

    Args:
        op_name (str): The name of this pooling layer operation
        rate (float): The dropout rate
        input_layer (XLayer): The input layer to this pooling layer
    """

    attrs = kwargs
    attrs.update({
        'rate': rate
    })

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=['Dropout'],
        shapes=input_layer.shapes[:],
        sizes=input_layer.shapes.get_size(),
        layer=[op_name],
        tops=[],
        bottoms=[input_layer.name],
        attrs=attrs,
        targets=[])

    return X


#######
# Exp #
#######

@xop_register('Exp')
def exp(attrs: Dict[str, Any], in_xlayers: List[XLayer]) -> Dict:
    """ Return exponent registration information (shape) """

    assert len(in_xlayers) == 1

    shape = in_xlayers[0].shapes[:]

    return {'shape': shape}


##############
# ExpandDims #
##############

@xop_register('ExpandDims')
def expand_dims(attrs: Dict[str, Any], in_xlayers: List[XLayer]) -> Dict[str, List[int]]:
    """ Return ExpandDims registration information (shape) """

    assert len(in_xlayers) == 1
    assert 'axis' in attrs
    assert 'num_newaxis' in attrs

    shape = in_xlayers[0].shapes[:]

    axis = attrs['axis']
    num_newaxis = attrs['num_newaxis']

    assert axis < 0 or axis <= len(shape)
    assert axis > 0 or axis >= -len(shape) - 1

    if axis < 0:
        axis = len(shape) + axis + 1

    new_shape = TensorShape(shape[:axis] + [1] * num_newaxis + shape[axis:])

    return {'shape': new_shape}


#######
# Log #
#######

@xop_register('Log')
def log(attrs: str, in_xlayers: List[XLayer]) -> XLayer:
    """
    Return Log registration information (shape)
    """

    assert len(in_xlayers) == 1

    shape = in_xlayers[0].shapes[:]

    return {'shape': shape}


############
# Multiply #
############

@xop_register('Multiply')
def multiply(attrs: str, in_xlayers: List[XLayer]) -> XLayer:
    """
    Return numpy-style Multiplication layer registration information (shape)
    NOTE: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    """

    assert len(in_xlayers) == 2

    # NOTE: One of the input layer's shape size should be strictly larger
    #   than the other input's shape size or the input's shape should be
    #   larger than or equal to the other input's shape in every dimension

    lX, rX = in_xlayers
    if len(lX.shapes) >= len(rX.shapes):
        lshape = lX.shapes[:]
        rshape = [None] * (len(lX.shapes) - len(rX.shapes)) + rX.shapes[:]
    else:
        rshape = rX.shapes[:]
        lshape = [None] * (len(rX.shapes) - len(lX.shapes)) + lX.shapes[:]

    assert len(lshape) == len(rshape)

    reversed_shape = []
    for ls, rs in zip(reversed(lshape), reversed(rshape)):
        if ls == rs or ls in [1, None] or rs in [1, None]:
            if ls is None:
                reversed_shape.append(rs)
            elif rs is None:
                reversed_shape.append(ls)
            else:
                reversed_shape.append(max(ls, rs))
        else:
            raise ValueError("Invalid shapes for broadcasted multiplication: "
                             f"{lX.shapes} and {rX.shapes}")

    shape = TensorShape(list(reversed(reversed_shape)))

    return {'shape': shape}


########
# ReLU #
########

@xop_register('ReLU')
def relu(
    attrs: Dict[str, Any],
    in_xlayers: List[XLayer],
) -> Dict[str, List[int]]:
    """
    Return ReLU shape information
    """
    assert len(in_xlayers) == 1, "ReLU expects one input layer"
    shape = in_xlayers[0].shapes[:]
    return {'shape': shape}


@xop_register_op_transpose_transform('ReLU')
def relu_transpose_transform(X: XLayer, axes: List[int]) -> None:
    """
    Transform ReLU layer with transpose according to provided axes
    """

    new_shape = TensorShape([X.shapes[i] for i in axes])
    X.shapes = new_shape


#########
# rSqrt #
#########

@xop_register('rSqrt')
def rsqrt(
    attrs: Dict[str, Any],
    in_xlayers: List[XLayer],
) -> Dict[str, List[int]]:
    """
    Return rSqrt (1/Sqrt) registration information (shape)
    """

    assert len(in_xlayers) == 1

    shape = in_xlayers[0].shapes[:]

    return {'shape': shape}


#########
# Scale #
#########

@xop_register_factory('Scale')
def scale(
    op_name: str,
    input_layer: XLayer,
    gamma_layer: XLayer,
    beta_layer: XLayer,
    axis: int,
    **kwargs: Any,
):
    """
    Create a scaling XLayer

    Args:
        op_name (str): the name of this scaling layer operation
        axis (int): the axis over which the scaling is done
        input_layer (XLayer): the input layer to this scaling layer
        gamma_layer (XLayer): the scaling input layer
        beta_layer (XLayer): the beta input layer
    """

    bottoms = [input_layer.name]

    gamma = gamma_layer.data[0]  # numpy.ndarray
    beta = beta_layer.data[0]
    assert gamma.shape == beta.shape

    data = ScaleData(gamma, beta)
    attrs = kwargs
    attrs.update({
        'axis': axis
    })

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=['Scale'],
        shapes=input_layer.shapes[:],
        sizes=input_layer.sizes[:],
        data=data,
        layer=[op_name],
        tops=[],
        bottoms=bottoms,
        attrs=attrs,
        targets=[]
    )
    return X


@xop_register_op_transpose_transform('Scale')
def scale_transpose_transform(X: XLayer, axes: List[int]):
    """
    Transform scaling layer with transpose according to provided axes
    """

    new_shape = TensorShape([X.shapes[i] for i in axes])

    X.shapes = new_shape
    X.attrs['axis'] = axes.index(X.attrs['axis']) if X.attrs['axis'] != -1 else -1


###########
# Sigmoid #
###########

@xop_register('Sigmoid')
def sigmoid(attrs, in_xlayers):
    """ Return sigmoid registration information (shape) """
    assert len(in_xlayers) == 1
    shape = in_xlayers[0].shapes[:]
    return {'shape': shape}


###########
# Softmax #
###########

@xop_register('Softmax')
def softmax(attrs: Dict[str, Any], in_xlayers: List[XLayer]) -> Dict[str, List[int]]:
    """ Return softmax registration information (shape) """
    assert len(in_xlayers) == 1
    shape = in_xlayers[0].shapes[:]
    return {'shape': shape}


########
# Sqrt #
########

@xop_register('Sqrt')
def sqrt(attrs: Dict[str, Any], in_xlayers: List[XLayer]) -> Dict[str, List[int]]:
    """
    Return Sqrt registration information (shape)
    """
    assert len(in_xlayers) == 1, ""
    shape = in_xlayers[0].shapes[:]
    return {'shape': shape}


#######
# Sub #
#######

@xop_register('Sub')
def sub(attrs: Dict[str, Any], in_xlayers: List[XLayer]) -> Dict[str, List[int]]:
    """
    Return numpy-style subtraction layer registration information (shape)

    NOTE: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    """
    assert len(in_xlayers) == 2, "Subtract layer expects two input layers"
    lX, rX = in_xlayers
    shape = TensorShape(get_numpy_broadcasted_shape(lX.shapes[:], rX.shapes[:]))
    return {'shape': shape}


########
# Tanh #
########

@xop_register('Tanh')
def tanh(attrs: Dict[str, Any], in_xlayers: List[XLayer]) -> Dict[str, List[int]]:
    """
    Return Tanh registration information (shape)
    """
    assert len(in_xlayers) == 1, "Tanh expects one input layer"
    shape = in_xlayers[0].shapes[:]
    return {'shape': shape}
