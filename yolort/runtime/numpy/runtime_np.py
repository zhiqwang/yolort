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
Module for running neural network models on top of numpy
"""

from typing import Dict, List, Callable

import logging
import numpy as np

from ...shared import fancy_logging
from ...graph import XGraph

from ..base_runtime import BaseRuntime
from .. import base

from . import rt_layer_np

logger = logging.getLogger("pyxir")
fancy_logger = fancy_logging.getLogger("pyxir")


X_2_NP = {
    # INPUT/OUTPUT
    'Input': base.get_input_layer(rt_layer_np.InputLayer, rt_layer_np.ConstantLayer),
    'Output': base.get_layer(rt_layer_np.OutputLayer),

    # DATA STRUCTURES
    'Tuple': base.get_layer(rt_layer_np.TupleLayer),
    'TupleGetItem': base.get_layer(rt_layer_np.TupleGetItemLayer),

    # PREPROCESSING
    'StrInput': base.get_layer(rt_layer_np.InputLayer),
    'Cvx': base.get_layer(rt_layer_np.CvxLayer),

    # NN
    'BiasAdd': base.get_bias_add_layer(rt_layer_np.BiasAddLayer, rt_layer_np.ConstantLayer),
    'Dense': base.get_dense_layer(rt_layer_np.DenseLayer, rt_layer_np.ConstantLayer),
    'Eltwise': base.get_elementwise_layer(rt_layer_np.ElementwiseLayer, rt_layer_np.ReluLayer),
    'Mean': base.get_layer(rt_layer_np.MeanLayer),

    # NON-LINEARITIES
    'Softmax': base.get_layer(rt_layer_np.SoftmaxLayer),
    'ReLU': base.get_layer(rt_layer_np.ReluLayer),
    'pReLU': base.get_layer(rt_layer_np.LeakyReluLayer),

    # TRANSFORMATIONS
    'Reshape': base.get_layer(rt_layer_np.ReshapeLayer),
    'Flatten': base.get_layer(rt_layer_np.FlattenLayer),
    'Squeeze': base.get_layer(rt_layer_np.SqueezeLayer),
    'Transpose': base.get_layer(rt_layer_np.TransposeLayer),

    # CONVOLUTION
    'Convolution': base.get_conv2d_layer(rt_layer_np.ConvLayer, rt_layer_np.ConstantLayer),
    'Pooling': base.get_pooling_layer(rt_layer_np.PoolingLayer)
}


class RuntimeNP(BaseRuntime):
    """
    Runtime on top of Numpy for running XGraph models

    Args:
        TODO
    """

    def __init__(
        self,
        name,
        xgraph: XGraph,
        device: str = 'cpu',
        batch_size: str = -1,
        placeholder: bool = False,
        last_layers: List[str] = None,
        **kwargs,
    ):
        super().__init__(name, xgraph, device, batch_size, placeholder, last_layers)

    def _xfdnn_op_to_exec_op(self, op_type: str) -> Callable:
        """
        Overwrites Runtime abstract method.

        Takes a operation type and returns a function of type:
        (XLayer, Dict[str,List[int]], Dict[str,numpy.ndarray],
            Dict[str,Dict]) -> List[rt_layer.RtLayer]
        that takes in a parameters layer object, inputs shapes dict, params
        dict and quantization parameters dict and outputs and returns a list
        of executable RtLayerTF objects
        """
        if op_type not in X_2_NP:
            raise NotImplementedError(f"Operation of type: {op_type} is not supported  on RuntimeNP")
        return X_2_NP[op_type]

    def optimize(self, inputs: Dict[str, np.ndarray], debug: bool = False):
        raise NotImplementedError
