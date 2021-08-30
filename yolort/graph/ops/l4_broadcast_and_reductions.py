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

L4: Broadcast and Reduction Operators
"""

from typing import Dict, List

import logging

from yolort.shapes import TensorShape, get_numpy_broadcasted_shape

from ..layer.xlayer import XLayer
from ..layer.xlayer_factory import xop_register_factory, xop_register
from ..xop_registry import xop_register_op_transpose_transform

logger = logging.getLogger("pyxir")


###########
# Greater #
###########

@xop_register('Greater')
def greater(
    attrs: Dict,
    in_xlayers: List[XLayer],
) -> XLayer:
    """
    Return numpy-style greater layer registration information (shape)

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
            raise ValueError("Invalid shapes for broadcasted additions:"
                             " {} and {}".format(lX.shapes, rX.shapes))

    shape = TensorShape(list(reversed(reversed_shape)))

    return {'shape': shape}


###########
# Maximum #
###########

@xop_register('Maximum')
def maximum(attrs: Dict, in_xlayers: List[XLayer]):
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


@xop_register_op_transpose_transform('Maximum')
def maximum_transpose_transform(X: XLayer, axes: List[int]):
    """
    Transform maximum layer with transpose according to provided axes
    """
    new_shape = TensorShape([X.shapes[i] for i in axes])
    X.shapes[:] = new_shape


########
# Mean #
########

@xop_register_factory('Mean')
def mean(
    op_name: str,
    input_layer: XLayer,
    axes: List[int],
    keepdims: bool,
    exclude: bool,
    **kwargs,
) -> XLayer:
    """
    Compute the mean of the input layer over some axes

    Args:

        op_name (str): The name of this elementwise addition operation
        axes (List[int]): The axes over which to compute the mean
        ... TODO
        input_layer (XLayer): The input layer
    """

    attrs = kwargs

    logger.debug("Attrs: {}".format(attrs))

    bottoms = [input_layer.name]

    in_shape = input_layer.shapes[:]

    if exclude:
        axes = [i for i in range(len(in_shape)) if i not in axes]

    if keepdims:
        newshape = [dim if i not in axes else 1
                    for i, dim in enumerate(in_shape)]
    else:
        newshape = [dim for i, dim in enumerate(in_shape)
                    if i not in axes]

    newshape = TensorShape(newshape)
    logger.debug("Mean axes: {}, in shape: {}, out shape: {}"
                 .format(axes, in_shape, newshape))

    attrs.update({
        'axes': axes,
        'keepdims': keepdims,
        # 'exclude': exclude
        #  TODO: dtype??
    })

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=['Mean'],
        shapes=newshape,
        sizes=newshape.get_size(),
        layer=[op_name],
        tops=[],
        bottoms=bottoms,
        attrs=attrs,
        targets=[]
    )

    return X


@xop_register_op_transpose_transform('Mean')
def mean_transpose_transform(
    X: XLayer,
    axes: List[int],
) -> None:
    """
    Transform Mean layer with transpose according to provided axes
    """
    new_shape = [X.shapes[i] for i in axes]
    X.shapes = new_shape
    X.attrs['axes'] = [axes.index(axis) for axis in X.attrs['axes']]


################
# StridedSlice #
################

@xop_register('StridedSlice')
def strided_slice(attrs, in_xlayers):
    """
    Strided slice of an array
    """

    # assert len(in_xlayers) == 1
    begin = attrs['begin']
    end = attrs['end']
    strides = attrs['strides']
    slice_mode = attrs['slice_mode']

    in_shape = in_xlayers[0].shapes[:]
    assert len(in_shape) == len(begin)
    assert len(in_shape) == len(end)
    assert len(in_shape) == len(strides)
    newshape = []

    if slice_mode == 'end':
        for i in range(len(in_shape)):
            newshape.append(int((end[i] - begin[i]) / strides[i]))
    elif slice_mode == 'size':
        raise ValueError("Slice mode `size` not supported yet in PyXIR")
    else:
        raise ValueError("Slice mode `{}` not supported yet in PyXIR".format(slice_mode))

    logger.debug("-- new shape: {}".format(newshape))

    return {'shape': TensorShape(newshape)}
