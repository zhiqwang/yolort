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
Module for pyxir execution layers implemented on top of numpy
"""

from typing import List
import logging
import numpy as np

from .. import rt_layer
from . import nn

logger = logging.getLogger('pyxir')

##################
# INPUTS/OUTPUTS #
##################


class InputLayer(rt_layer.InputLayer):

    def init(self):
        logger.debug(f"Initializing InputLayer with shape: {self.shape}")

    def forward_exec(self, inputs: List[np.ndarray]) -> np.ndarray:
        assert(len(inputs) == 1)

        return inputs[0]


class ConstantLayer(rt_layer.ConstantLayer):

    def init(self):
        logger.debug(f"Initializing ConstantLayer with shape: {self.shape}")

    def forward_exec(self, inputs: List[np.ndarray]) -> np.ndarray:
        assert(len(inputs) == 0)

        return self.value


class OutputLayer(rt_layer.OutputLayer):

    def init(self):
        logger.debug(f"Initializing OutputLayer with shape: {self.shape}")

    def forward_exec(self, inputs: List[np.ndarray]) -> np.ndarray:
        assert(len(inputs) == 1)

        return inputs[0]

###################
# DATA STRUCTURES #
###################


class TupleLayer(rt_layer.BaseLayer):

    """
    Tuple layer takes input layers and groups them in a tuple output
    """

    def init(self):
        logger.debug(f"Initializing TupleLayer with shape: {self.shape}")

    def forward_exec(self, inputs: List[np.ndarray]) -> np.ndarray:
        return tuple(inputs)


class TupleGetItemLayer(rt_layer.BaseLayerMultiOutputInput):

    """
    TupleGetItem layer takes in a Tuple an returns the specified
    tuple element
    """

    def init(self):
        logger.debug(f"Initializing TupleGetItemLayer with shape: {self.shape}")

        self.index = self.attrs['index']
        self.transpose = 'transpose' in self.attrs and self.attrs['transpose']
        self.axes = list(self.attrs['axes']) if self.transpose else []

    def forward_exec(self, inputs: List[np.ndarray]) -> np.ndarray:
        assert(isinstance(inputs[0], tuple))
        res = inputs[0][self.index]

        if self.transpose:
            return np.transpose(res, tuple(self.axes))

        return res

###################
# DATA STRUCTURES #
###################


class CvxLayer(rt_layer.BaseLayer):

    """
    Cvx layer which takes in a list of strings representing
    the image paths and subsequently loads the images and performs
    specified preprocessing.
    """

    def init(self):
        from pyxir.io.cvx import ImgLoader, ImgProcessor

        self.ImgLoader = ImgLoader
        self.ImgProcessor = ImgProcessor

        logger.debug(f"Initializing CvxLayer with shape: {self.shape}")

        self.data_layout = self.attrs['data_layout']
        if self.data_layout not in ['NCHW', 'NHWC']:
            raise ValueError(f"Unsupported data layout: {self.data_layout} for CvxLayer. "
                             "Supported layouts are `NCHW` and `NHWC`")
        self.cvx_key = self.attrs['cvx_key']

    def forward_exec(self, inputs):
        # type: (List[str]) -> np.ndarray
        assert(len(inputs) == 1)

        img_loader = self.ImgLoader()
        img_processor = self.ImgProcessor(
            proc_key=self.cvx_key
        )

        data = img_loader.load(inputs)
        res = img_processor.execute(data)

        if self.data_layout == 'NCHW':
            res = np.transpose(res, (0, 3, 1, 2))

        return res


######
# NN #
######

class BiasAddLayer(rt_layer.BaseLayer):

    def init(self):
        logger.debug(f"Initializing NP BiasAddLayer with shape: {self.shape}")

    def forward_exec(self, inputs: List[np.ndarray]) -> np.ndarray:
        assert(len(inputs) == 2)
        inpt, biases = inputs

        return np.add(inpt, biases)


class DenseLayer(rt_layer.DenseLayer):

    def init(self):
        logger.debug(f"Initializing DenseLayer with shape: {self.shape}")

    def forward_exec(self, inputs: List[np.ndarray]) -> np.ndarray:
        assert(len(inputs) == 3)

        inpt, weights, biases = inputs

        input_shape = self.input_shapes[0]

        if self.kernel_layout == 'OI':
            weights = np.transpose(weights, (1, 0))

        res = np.add(np.matmul(inpt, weights), biases)

        if self.use_relu:
            res = res.clip(0, np.inf)

        return res


class ElementwiseLayer(rt_layer.BaseLayer):

    def init(self):
        if len(self.inputs) != 2:
            raise ValueError("Run elementwise operation expects 2 inputs, "
                             f"{len(self.inputs)} given")

        self.op = self.attrs['op']
        if self.op != 'Add':
            raise NotImplementedError("Only elementwise add operation supported "
                                      f"at the moment, not: {self.op}")

        logger.debug(f"Initializing ElementwiseLayer with shape: {self.shape}")

    def forward_exec(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError("")


class MeanLayer(rt_layer.BaseLayer):

    def init(self):
        logger.debug(f"Initializing MeanLayer with shape: {self.shape}")

    def forward_exec(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError("")


###################
# NON-LINEARITIES #
###################


class SoftmaxLayer(rt_layer.BaseLayer):

    def init(self):
        logger.debug(f"Initializing SoftmaxLayer with shape: {self.shape}")

    def forward_exec(self, inputs: List[np.ndarray]) -> np.ndarray:
        assert(len(inputs) == 1)

        inpt = inputs[0]

        e_x = np.exp(inpt - np.max(inpt))
        res = e_x / e_x.sum()

        return res


class ReluLayer(rt_layer.BaseLayer):

    def init(self):
        logger.debug(f"Initializing ReluLayer with shape: {self.shape}")

    def forward_exec(self, inputs: List[np.ndarray]) -> np.ndarray:
        assert(len(inputs) == 1)

        X = inputs[0]

        return nn.relu(X)


class LeakyReluLayer(rt_layer.BaseLayer):

    def init(self):
        """
        y = alpha*x for x < 0
        y = x for x> 0
        """
        raise NotImplementedError("LeakyRelu not implemented for numpy"
                                  " runtime")

    def forward_exec(self, inputs: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError("")


###################
# TRANSFORMATIONS #
###################


class ReshapeLayer(rt_layer.BaseLayer):

    def init(self):
        self.target_shape = self.attrs['shape']

        logger.debug(f"Initializing ReshapeLayer with shape: {self.shape}")

    def forward_exec(self, inputs: List[np.ndarray]) -> np.ndarray:
        assert(len(inputs) == 1)

        input_shape, shape = self.input_shapes[0], self.target_shape

        logger.debug("New shape: {}".format(shape))
        if input_shape[0] in [-1, None] and shape[0] != -1:
            logger.warn("[WARNING]: Manually fixing invalid fixed reshape layer "
                        f"shape: {shape}, input has variable shape in first layer")

            assert len(shape) >= 2
            shape = [-1] + shape[1:]

        res = np.reshape(inputs[0], shape)

        return res


class FlattenLayer(rt_layer.BaseLayer):

    def init(self):
        logger.debug(f"Initializing FlattenLayer with shape: {self.shape}")

    def forward_exec(self, inputs: List[np.ndarray]) -> np.ndarray:
        assert(len(inputs) == 1)

        # TODO
        newshape = [(dim if dim is not None else -1) for dim in self.shape]
        res = np.reshape(inputs[0], newshape)

        return res


class SqueezeLayer(rt_layer.BaseLayer):

    def init(self):
        self.axis = self.attrs['axis']

        logger.debug(f"Initializing SqueezeLayer with shape: {self.shape}")

    def forward_exec(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        assert(len(inputs) == 1)

        res = np.squeeze(inputs[0], axis=tuple(self.axis))

        return res


class TransposeLayer(rt_layer.BaseLayer):

    def init(self):
        self.axes = self.attrs['axes']

        logger.debug(f"Initializing TransposeLayer with shape: {self.shape}")

    def forward_exec(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        assert(len(inputs) == 1)

        res = np.transpose(inputs[0], axes=self.axes)

        return res


################
# CONVOLUTIONS #
################


class ConvLayer(rt_layer.ConvLayer):

    def init(self):
        logger.debug(f"Initializing ConvLayer with shape: {self.shape}")

        self.data_layout = self.attrs['data_layout']

    def forward_exec(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        if self.data_layout != 'NCHW':
            raise NotImplementedError("Only 'NCHW' layout supported for "
                                      "numpy conv2d layer for now")

        X, kernel, bias = inputs
        kernel_layout, paddings, strides, dilations = (
            self.kernel_layout, self.paddings, self.strides, self.dilations)

        # TODO
        assert(dilations == [1, 1, 1, 1])
        assert(strides[0:2] == [1, 1])

        if kernel_layout == 'HWIO':
            kernel = np.transpose(kernel, (3, 2, 0, 1))
        elif kernel_layout == 'OHWI':
            kernel = np.transpose(kernel, (0, 3, 1, 2))

        # zero padding
        X = np.pad(X, paddings, mode='constant')

        X_res, _ = nn.conv2d(X, kernel, strides=strides[2:4])

        # Add bias
        if bias.any():
            X_res += bias.reshape([1, -1, 1, 1])

        # Activations
        if self.use_activation == 'relu':
            X_res = nn.relu(X_res)
        elif self.use_activation == 'leaky_relu':
            raise NotImplementedError("Leaky relu not implemented")

        return X_res


class PoolingLayer(rt_layer.PoolingLayer):

    def init(self):
        logger.debug(f"Initializing PoolingLayer with shape: {self.shape}")

        self.data_layout = self.attrs['data_layout']

    def forward_exec(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        assert(len(inputs) == 1)

        X = inputs[0]

        op, ksize, paddings, strides = self.op, self.ksize, self.paddings, self.strides

        assert(strides[0:2] == [1, 1])

        if self.data_layout != 'NCHW':
            raise ValueError("Numpy runtime does only support pooling"
                             " in 'NCHW' format")

        if op == 'Max':
            pool_func = nn.max_pool
        elif op == 'Avg':
            pool_func = nn.avg_pool
        else:
            raise NotImplementedError("Provided pooling operation is not "
                                      f"supported at this moment: {op}")

        # zero padding
        X = np.pad(X, paddings, mode='constant')

        # padding='VALID',
        X_res = pool_func(X, ksize=ksize, strides=strides[2:4])

        return X_res
