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

import logging

from typing import List, Optional

from yolort.shapes import TensorShape

from ..layer.xlayer import defaultXLayer, XLayer
from ..layer.xlayer_factory import xop_register_factory

logger = logging.getLogger("pyxir")


#######
# Cvx #
#######

@xop_register_factory('Cvx')
def cvx(
    op_name: str,
    input_layer: XLayer,
    cvx_key: str,
    shape: List[int],
    dtype: Optional[str] = None,
    **kwargs,
):
    """
    Create a cvx input XLayer

    Args:
        op_name (str): The name of this input layer
        cvx_key (str): The cvx key to be used for preprocessing
        shape (List[int]): The input shape
        layout (str): The data layout (`NCHW` and `NHWC` supported for now)
        dtype (Optional[str]): The input data type
    """

    bottoms = [input_layer.name]

    shape[0] = -1
    shape = TensorShape(shape)

    attrs = kwargs
    attrs.update({
        'dtype': dtype,
        'cvx_key': cvx_key
    })

    X = defaultXLayer()
    X = X._replace(
        name=op_name,
        type=['Cvx'],
        shapes=shape,
        sizes=shape.get_size(),
        layer=[op_name],
        bottoms=bottoms,
        attrs=attrs
    )

    return X


#############
# YoloReorg #
#############

@xop_register_factory('YoloReorg')
def yolo_reorg(op_name: str, input_layer: XLayer, stride: int, layout: str, **kwargs) -> XLayer:
    """
    Shuffle and shape transform input data based on stride

    TODO: example

    Args:
        op_name (str): The name of this elementwise addition operation
        stride (int): The stride to be used for reorganization
        input_layer (XLayer): The input layer
    """

    if layout != 'NCHW':
        raise NotImplementedError("YoloReorg is only supported for NCHW data layout")

    attrs = kwargs
    attrs.update({
        'stride': stride,
        'layout': layout
    })

    in_shape = input_layer.shapes[:]

    if in_shape[2] % stride != 0:
        raise ValueError("Invalid YoloReorg operation: height dimension size: "
                         f"{in_shape[2]} should be divisible by: {stride}")
    if in_shape[3] % stride != 0:
        raise ValueError("Invalid YoloReorg operation: height dimension size: "
                         f"{in_shape[3]} should be divisible by: {stride}")

    shape = TensorShape([
        in_shape[0],
        in_shape[1] * stride * stride,
        int(in_shape[2] / stride),
        int(in_shape[3] / stride),
    ])

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=['YoloReorg'],
        shapes=shape,
        sizes=shape.get_size(),
        layer=[op_name],
        tops=[],
        bottoms=[input_layer.name],
        attrs=attrs,
        targets=[]
    )
    return X
