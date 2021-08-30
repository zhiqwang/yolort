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
Module for neural network layers
"""

from typing import List, Dict
import abc
import numpy as np

from yolort.shapes import TupleShape, TensorShape


class RtLayer:

    __metaclass__ = abc.ABCMeta

    """
    A generic layer for an xfDNN neural network graph

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes (List[List[int] or Tuple[int]]): the input shapes
            for all the inputs
        subgraph (str): Indicates the subgraph this layer belongs to
    """

    def __init__(
        self,
        name,
        op_type,
        shape,
        dtype,
        inputs,
        input_shapes,
        subgraph,
    ):

        self.name = name
        self.type = op_type
        self.set_shape(shape)
        self.dtype = dtype
        self.inputs = inputs
        self.set_input_shapes(input_shapes)
        self.subgraph = subgraph

    @abc.abstractmethod
    def init(self):
        """
        TODO: docstring
        """
        raise NotImplementedError("")

    def set_shape(self, shape: List[int]):
        # TODO
        # if not isinstance(shape, list):
        #     raise ValueError("RtLayer shape should be of list type")
        # elif isinstance(shape[0], list):
        #     self.shape = [[(dim if dim is not None and dim >= 0 else None)
        #                    for dim in sh] for sh in shape]
        # else:
        #     self.shape = [(dim if dim is not None and dim >= 0 else None)
        #                   for dim in shape]
        if not isinstance(shape, (TupleShape, TensorShape)):
            raise ValueError("RtLayer shape should be of TensorShape or TupleShape type, "
                             f"but got: {type(shape)}")
        self.shape = shape._replace(-1, None)

    def set_input_shapes(self, input_shapes: List[List[int]]) -> None:
        # self.input_shapes = [[(dim if dim is not None and dim >= 0 else None)
        #                      for dim in shape] for shape in input_shapes]
        if not isinstance(input_shapes, list):
            raise ValueError("RtLayer shape should be of type list, "
                             f"but got: {type(input_shapes)}")
        self.input_shapes = [ishape._replace(-1, None) for ishape in input_shapes]

    @abc.abstractmethod
    def forward_exec(self, inputs: List[np.ndarray]) -> np.ndarray:
        """
        TODO: docstring
        """
        raise NotImplementedError("")

    # GETTERS

    def get_output_for_quantization(self, inputs: List[np.ndarray]) -> np.ndarray:
        """
        TODO: docstring
        """
        return self.forward_exec(inputs)

    def get_params(self) -> Dict:
        return {}

    def get_input_names(self) -> List[str]:
        return self.inputs

    # CHECKS

    def is_input_layer(self):
        return False


class BaseLayer(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic base layer

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes (List[List[int] or Tuple[int]]): the input shapes for all the inputs
        data (List[np.ndarray]): the data stored in the layer
        subgraph (str): Indicates the subgraph this layer belongs to
        attrs: the provided operation specific attributes
    """

    def __init__(
        self,
        name,
        xtype,
        shape,
        dtype,
        inputs,
        input_shapes,
        data,
        subgraph,
        attrs,
    ):
        # TODO: checks
        super().__init__(name, xtype, shape, dtype,
                         inputs, input_shapes, subgraph)

        self.data = data
        self.attrs = attrs

        self.init()


class BaseLayerMultiOutputInput(BaseLayer):

    __metaclass__ = abc.ABCMeta

    def set_input_shapes(self, input_shapes):
        # type: (List[List[List[int]]]) -> None
        """
        Override for specific input shape handling, this layer expects
        input shapes as List[List[List[int]]] because of preceding Tuple
        output layer
        """
        self.input_shapes = [
            [
                [
                    (dim if dim is not None and dim >= 0 else None)
                    for dim in shape
                ]
                for shape in shapes
            ]
            for shapes in input_shapes
        ]


##################
# INPUTS/OUTPUTS #
##################


class InputLayer(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic input layer which can be implemented on various
    backends (e.g. tensorflow)

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes (List[List[int] or Tuple[int]]): the input shapes
            for all the inputs
        subgraph (str): Indicates the subgraph this layer belongs to
    """

    def __init__(
        self,
        name,
        shape,
        dtype,
        inputs,
        input_shapes,
        subgraph,
    ):
        # TODO: checks
        super().__init__(name, 'Input', shape, dtype,
                         inputs, input_shapes, subgraph)

        assert(len(self.input_shapes) == 1)

        self.init()

    def is_input_layer(self):
        return True


class ConstantLayer(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic constant layer which can be implemented on various
    backends (e.g. tensorflow)

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes (List[List[int] or Tuple[int]]): the input shapes
            for all the inputs
        subgraph (str): Indicates the subgraph this layer belongs to
        value (np.ndarray): the constant value
    """

    def __init__(
        self,
        name,
        shape,
        dtype,
        inputs,
        input_shapes,
        subgraph,
        value,
    ):
        super().__init__(name, 'Constant', shape, dtype,
                         inputs, input_shapes, subgraph)

        assert(len(self.input_shapes) == 0)
        assert(tuple(self.shape) == tuple(value.shape))

        self.value = value

        self.init()

    def is_input_layer(self):
        return False


class OutputLayer(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic output layer which can be implemented on various
    backends (e.g. tensorflow)

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes (List[List[int] or Tuple[int]]): the input shapes
            for all the inputs
        subgraph (str): Indicates the subgraph this layer belongs to
    """

    def __init__(
        self,
        name,
        shape,
        dtype,
        inputs,
        input_shapes,
        subgraph,
    ):
        # TODO: checks
        super().__init__(name, 'Output', shape, dtype,
                         inputs, input_shapes, subgraph)

        self.init()


#######################
# BASIC NN OPERATIONS #
#######################


class DenseLayer(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic Dense operation layer which can be implemented on various
    backends (e.g. tensorflow)

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes (List[List[int] or Tuple[int]]): the input shapes
            for all the inputs
        subgraph (str): Indicates the subgraph this layer belongs to
        kernel_layout (str): the layout of the data input
        weights (np.ndarray): the weights for the matrix multiplication
            with the input
        kernel_layout (str): the layout of the weight matrix, should be
            either OI (outchan, inchan) or IO
        biases (np.ndarray): the biases to be added to the result of the
            inputs-weights multiplication
        use_relu (bool): Whether to add a relu operation to this layer
    """

    def __init__(
        self,
        name,
        shape,
        dtype,
        inputs,
        input_shapes,
        subgraph,
        data_layout,
        weights,
        kernel_layout,
        biases,
        use_relu,
    ):
        # TODO: checks
        super().__init__(name, 'Dense', shape, dtype,
                         inputs, input_shapes, subgraph)

        assert len(self.inputs) == 3
        assert len(self.input_shapes) == 3
        assert len(self.shape) == len(self.input_shapes[0])

        if data_layout not in ['NC', 'CN']:
            raise ValueError("Invalid data layout: {} for dense layer: {}"
                             " should be either NC or CN"
                             .format(data_layout, self.name))
        if kernel_layout not in ['OI', 'IO']:
            raise ValueError("Invalid weights layout: {} for dense layer: {}"
                             " should be either OI (outchan, inchan) or IO"
                             .format(kernel_layout, self.name))

        self.data_layout = data_layout
        self.kernel_layout = kernel_layout
        self.weights = weights
        self.biases = biases
        self.use_relu = use_relu

        # INITIALIZE
        self.init()

#########################
# BASIC MATH OPERATIONS #
#########################


class BatchNormLayer(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic batch normalization operation layer which can be implemented on
    various backends (e.g. tensorflow)
    TODO

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes (List[List[int] or Tuple[int]]): the input shapes
            for all the inputs
        subgraph (str): Indicates the subgraph this layer belongs to
        mean (np.ndarray): the mean of the input
        variance (np.ndarray): the variance of the input
        gamma (np.ndarray): the scale value
        beta (np.ndarray): the offset value
        variance_epsilon (float): a small floating point number to add to
            variance to avoid dividing by zero in batchnorm calculation
        scale (np.ndarray (optional, default None)): the scale, if defined
            this scale will be applied to the batchnorm output
        beta (np.ndarray (optional, default None)): the offset to be added
            to the batchnorm ouput if defined
        attrs (dict): the layer attributes
    """

    def __init__(
        self,
        name,
        shape,
        dtype,
        inputs,
        input_shapes,
        subgraph,
        mean,
        variance,
        gamma,
        beta,
        variance_epsilon,
        attrs=None,
    ):
        # TODO: checks
        super().__init__(name, 'BatchNorm', shape, dtype,
                         inputs, input_shapes, subgraph)

        assert(len(self.inputs) <= 5)
        assert(len(self.input_shapes) <= 5)
        assert(self.input_shapes[0] == self.shape)

        if mean is not None and mean.shape[0] != self.shape[1]:
            raise ValueError("BatchNorm layer: mean size should be equal to"
                             " number of input channels, but got {} and {}"
                             .format(mean.shape[0], self.shape[1]))
        if variance is not None and variance.shape[0] != self.shape[1]:
            raise ValueError("BatchNorm layer: variance size should be equal"
                             " to number of input channels, but got {} and {}"
                             .format(variance.shape[0], self.shape[1]))
        if gamma is not None and gamma.shape[0] != self.shape[1]:
            raise ValueError("BatchNorm layer: gamma size should be equal"
                             " to number of input channels, but got {} and {}"
                             .format(gamma.shape[0], self.shape[1]))
        if beta is not None and beta.shape[0] != self.shape[1]:
            raise ValueError("BatchNorm layer: beta size should be equal"
                             " to number of input channels, but got {} and {}"
                             .format(variance.shape[0], self.shape[1]))

        assert 'axis' in attrs

        self.mean = mean
        self.variance = variance
        self.gamma = gamma
        self.beta = beta
        self.variance_epsilon = variance_epsilon
        self.attrs = attrs if attrs is not None else {}

        # INITIALIZE
        self.init()


class ScaleLayer(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic scaling operation layer which can be implemented on various
    backends (e.g. tensorflow)
    TODO

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes (List[List[int] or Tuple[int]]): the input shapes
            for all the inputs
        subgraph (str): Indicates the subgraph this layer belongs to
        gamma (np.ndarray): the scale to be applied to the input
        beta (np.ndarray): the offset to be added to tthe input
    """

    def __init__(
        self,
        name,
        shape,
        dtype,
        inputs,
        input_shapes,
        subgraph,
        attrs,
        gamma,
        beta,
    ):
        # TODO: checks
        super().__init__(name, 'Scale', shape, dtype,
                         inputs, input_shapes, subgraph)

        assert(len(self.inputs) <= 3)
        assert(len(self.input_shapes) <= 3)
        assert(self.input_shapes[0] == self.shape)

        if gamma is not None and gamma.shape[0] != self.shape[1]:
            raise ValueError("Scale layer: scale size should be equal to number of input channels, "
                             f"but got {gamma.shape[0]} and {self.shape[1]}")
        if beta is not None and beta.shape[0] != self.shape[1]:
            raise ValueError("Scale layer: beta size should be equal to number of input channels, "
                             f"but got {beta.shape[0]} and {self.shape[1]}")

        assert 'axis' in attrs

        self.attrs = attrs
        self.gamma = gamma
        self.beta = beta

        # INITIALIZE
        self.init()

    def get_params(self):
        return {'gamma': self.gamma, 'beta': self.beta}


#################
# CONVOLUTIONAL #
#################


class ConvLayer(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic convolutional layer which can be implemented on various
    backends (e.g. tensorflow)

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes (List[List[int] or Tuple[int]]): the input shapes
            for all the inputs
        subgraph (str): Indicates the subgraph this layer belongs to
        attrs (dict): the attributes
        kernel (np.ndarray,): The convolution kernel to be used,
            should be in format OIHW(=outchan, inchan, height, width)
        kernel_layout (str): the layout of the kernel, should be either OIHW, HWIO or OHWI
        kernel_groups (int): controls the number of convolutions to be done on the input channels.
            If parameter is equal to 1 respectively in_channels this operation is 
            the normal conv2d respectively depthwise conv2d. Other values are not 
            permitted at the moment.
        biases (np.ndarray,): The biases to be added to each convolution output channel
        paddings (List[List[int]], dim*2): The paddings to be used for each
            dimension. If layout is NCHW, then padding argument should be 
            [[pad_N_before, pad_N_after], [pad_C_before, pad_C_after], ...]
        strides (List[int]): If layout is NCHW, [stride_N, stride_C, stride_H, stride_W]
        dilations (List[int]): If layout is NCHW, [dilation_N, dilation_C, dilation_H, dilation_W]
        use_activation (str): Add specified activation function after this layer [relu, leaky_relu].
            Default: None
        activation_atrs (Dict): The attributes used by the activation function
            Default: {}
    """

    def __init__(
        self,
        name,
        shape,
        dtype,
        inputs,
        input_shapes,
        subgraph,
        attrs,
        kernel,
        kernel_layout,
        kernel_groups,
        biases,
        paddings,
        strides,
        dilations,
        use_activation=None,
        activation_attrs=None
    ):
        # TODO: checks
        super().__init__(name, 'Convolution', shape, dtype,
                         inputs, input_shapes, subgraph)

        if kernel is None and len(self.input_shapes) < 2:
            raise ValueError("Kernel should either be initialized or a second "
                             f"input should be provided for convolution layer: {self.name}")
        elif biases is None and len(self.input_shapes) < 3:
            raise ValueError("Biases should either be initialized or a third "
                             f"input should be provided for convolution layer: {self.name}")
        elif kernel is not None and biases is not None and len(self.input_shapes) != 1:
            raise ValueError(f"Invalid number of input shapes: {self.name}, there should "
                             "be only one input if kernel and biases are provided")
        elif len(self.input_shapes) != len(self.inputs):
            raise ValueError("Number of input shapes should be equal to the number of inputs but "
                             f"are {len(self.input_shapes)} and {len(self.inputs)}")

        if kernel_layout not in ['OIHW', 'HWIO', 'OHWI']:
            raise ValueError(f"Invalid kernel layout: {kernel_layout} for convolution "
                             f"layer {self.name}, layout should be OIHW, HWIO or OHWI")

        channel_idx = attrs['data_layout'].index('C')
        # if kernel_groups not in [1, self.input_shapes[0][channel_idx]]:
        #     raise NotImplementedError(
        #         "Invalid number of kernel groups: {}. Only 1 and {} (number of"
        #         " channels of input) supported. These correspond to a standard"
        #         " conv2d respectively depthwise conv2d operation."
        #         .format(kernel_groups, self.input_shapes[0][channel_idx]))

        if len(paddings) != 4:
            raise ValueError(f"Invalid number of paddings: {paddings}, paddings should "
                             "have length 4 (`NCHW`)")
        if len(strides) != 4:
            raise ValueError(f"Invalid number of strides: {paddings}, strides should "
                             "have length 4 (`NCHW`)")
        if len(dilations) != 4:
            raise ValueError(f"Invalid number of dilations: {paddings}, dilations should "
                             "have length 4 (`NCHW`)")

        if use_activation not in [None, 'relu', 'leaky_relu']:
            raise ValueError(f"Invalid activation after conv2d operation: {use_activation}. "
                             "Only [None, 'relu', 'leaky_relu'] are supported.")

        self.attrs = attrs

        self.kernel = kernel
        self.kernel_layout = kernel_layout
        self.kernel_groups = kernel_groups
        self.biases = biases

        self.paddings = paddings
        self.strides = strides
        self.dilations = dilations

        self.use_activation = use_activation
        self.activation_attrs = activation_attrs if activation_attrs is not None else {}

        # INITIALIZE
        self.init()

    def get_params(self):
        return {'W': self.kernel, 'B': self.biases}


class Conv2DTransposeLayer(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic traspose convolutional layer which can be implemented on various
    backends (e.g. tensorflow)

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes (List[List[int] or Tuple[int]]): the input shapes
            for all the inputs
        subgraph (str): Indicates the subgraph this layer belongs to
        kernel (np.ndarray,): The convolution kernel to be used, should be in format OIHW
            (=outchan, inchan, height, width)
        kernel_layout (str): the layout of the kernel, should be either OIHW, HWIO or OHWI
        kernel_groups (int): controls the number of convolutions to be done on the input channels.
            If parameter is equal to 1 respectively in_channels this operation is 
            the normal conv2d respectively depthwise conv2d. Other values are not 
            permitted at the moment.
        biases (np.ndarray,): The biases to be added to each convolution output channel
        paddings (List[List[int]], dim*2): The paddings to be used for each dimension.
            If layout is NCHW, then padding argument should be
            [[pad_N_before, pad_N_after], [pad_C_before, pad_C_after], ...]
        strides (List[int]): If layout is NCHW, [stride_N, stride_C, stride_H, stride_W]
        dilations (List[int]): If layout is NCHW, [dilation_N, dilation_C, dilation_H, dilation_W]
        use_activation (str): Add specified activation function after this layer [relu, leaky_relu]
            Default: None.
        activation_atrs (Dict): The attributes used by the activation function
            Default: {}.
        batch_size (int): The batch size used for this runtime layer, necessary for tf 
            conv2d_transpose output_shape argument
    """

    def __init__(
        self,
        name,
        shape,
        dtype,
        inputs,
        input_shapes,
        subgraph,
        attrs,
        kernel,
        kernel_layout,
        kernel_groups,
        biases,
        paddings,
        strides,
        dilations,
        use_activation=None,
        activation_attrs=None,
        batch_size=-1,
        placeholder=False,
    ):
        # TODO: checks
        super().__init__(name, 'Transposed Convolution', shape, dtype,
                         inputs, input_shapes, subgraph)

        if kernel is None and len(self.input_shapes) < 2:
            raise ValueError("Kernel should either be initialized or a second "
                             f"input should be provided for convolution layer: {self.name}")
        elif biases is None and len(self.input_shapes) < 3:
            raise ValueError("Biases should either be initialized or a third "
                             f"input should be provided for convolution layer: {self.name}")
        elif kernel is not None and biases is not None and len(self.input_shapes) != 1:
            raise ValueError(f"Invalid number of input shapes: {self.name}, there should "
                             "be only one input if kernel and biases are provided")
        elif len(self.input_shapes) != len(self.inputs):
            raise ValueError("Number of input shapes should be equal to the number of inputs "
                             f"but are {len(self.input_shapes)} and {len(self.inputs)}")

        if kernel_layout not in ['OIHW', 'HWIO', 'OHWI']:
            raise ValueError(f"Invalid kernel layout: {kernel_layout} for convolution "
                             f"layer {self.name}, layout should be OIHW, HWIO or OHWI")
        if kernel_groups not in [1, self.input_shapes[0][1]]:
            raise NotImplementedError(
                f"Invalid number of kernel groups: {kernel_groups}. Only 1 and "
                f"{self.input_shapes[0][1]} (number of channels of input) supported. "
                "These correspond to a standard conv2d respectively depthwise "
                "conv2d operation."
            )

        if len(paddings) != 4:
            raise ValueError(f"Invalid number of paddings: {paddings}, paddings should "
                             "have length 4 (`NCHW`)")
        if len(strides) != 4:
            raise ValueError(f"Invalid number of strides: {paddings}, strides should "
                             "have length 4 (`NCHW`)")
        if len(dilations) != 4:
            raise ValueError(f"Invalid number of dilations: {paddings}, dilations should "
                             "have length 4 (`NCHW`)")

        if use_activation not in [None, 'relu', 'leaky_relu']:
            raise ValueError(f"Invalid activation after conv2d operation: {use_activation}. "
                             "Only [None, 'relu', 'leaky_relu'] are supported.")

        self.attrs = attrs

        self.kernel = kernel
        self.kernel_layout = kernel_layout
        self.kernel_groups = kernel_groups
        self.biases = biases

        self.paddings = paddings
        self.strides = strides
        self.dilations = dilations

        self.use_activation = use_activation
        self.activation_attrs = activation_attrs if activation_attrs is not None else {}
        self.batch_size = batch_size
        self.placeholder = placeholder

        # INITIALIZE
        self.init()

    def get_params(self):
        return {'W': self.kernel, 'B': self.biases}


class PoolingLayer(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic pooling layer which can be implemented on various
    backends (e.g. tensorflow)

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes (List[List[int] or Tuple[int]]): the input shapes
            for all the inputs
        subgraph (str): Indicates the subgraph this layer belongs to
        op (str): The pooling operation, either Max or Avg
        paddings (List[List[int]]): The paddings to be used for each dimension.
            If layout is NCHW, then padding argument should be 
            [[pad_N_before, pad_N_after], [pad_C_before, pad_C_after], ...]
        ksize (List[int]): If layout is NCHW, [kernel_N, kernel_C, kernel_H, kernel_W]
        strides (List[int]): If layout is NCHW, [stride_N, stride_C, stride_H, stride_W]
    """

    def __init__(
        self,
        name,
        shape,
        dtype,
        inputs,
        input_shapes,
        subgraph,
        attrs,
        op,
        paddings,
        ksize,
        strides
    ):
        # TODO: checks
        super().__init__(name, 'Pooling', shape, dtype,
                         inputs, input_shapes, subgraph)

        assert(len(self.inputs) == 1)
        assert(len(self.input_shapes) == 1)

        self.attrs = attrs
        self.op = op
        self.ksize = ksize
        self.paddings = paddings
        self.strides = strides

        # INITIALIZE
        self.init()

#######################
# QUANTIZATION LAYERS #
#######################


class QuantizeLayer(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic quantization layer which can be implemented on various
    backends (e.g. tensorflow)

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes (List[List[int] or Tuple[int]]): the input shapes
            for all the inputs
        subgraph (str): Indicates the subgraph this layer belongs to
        input_types (List[str]): the data types for all inputs
        threshold (List[float]): the threshold (upper limit) for this quantization layer
        axis (int): specifies the axis on which the threshold should be applied
        bitwidth (int): the bitwidth to be quantized to
        do_rounding (bool): whether to do rounding to integer (otherwise, casting is used)
    """

    def __init__(
        self,
        name,
        shape,
        dtype,
        inputs,
        input_shapes,
        subgraph,
        input_types,
        threshold,
        axis,
        bitwidth,
        do_rounding=False
    ):
        super().__init__(name, 'Quantize', shape, dtype,
                         inputs, input_shapes, subgraph)

        assert(len(input_types) == len(self.input_shapes) == 1)
        if self.dtype not in ['int8']:
            raise ValueError(
                f"Invalid quantize layer (output) dtype: {self.dtype}, should be `int8`")

        self.input_types = input_types
        self.threshold = threshold
        self.axis = axis
        self.bitwidth = bitwidth
        self.do_rounding = do_rounding

        if self.bitwidth != 8:
            raise NotImplementedError("Quantize layer only supports bitwith 8 for now")

        # INITIALIZE
        self.init()


class UnQuantizeLayer(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic quantization layer which can be implemented on various
    backends (e.g. tensorflow)

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes (List[List[int] or Tuple[int]]): the input shapes
            for all the inputs
        subgraph (str): Indicates the subgraph this layer belongs to
        input_types (List[str]): the data types for all inputs
        threshold (float): the threshold (upper limit) for this quantization layer
        axis (int): specifies the axis on which the threshold should be applied
        bitwidth (int): the bitwidth to be quantized to
    """

    def __init__(
        self,
        name,
        shape,
        dtype,
        inputs,
        input_shapes,
        subgraph,
        input_types,
        threshold,
        axis,
        bitwidth
    ):
        super().__init__(name, 'UnQuantize', shape, dtype,
                         inputs, input_shapes, subgraph)

        assert(len(input_types) == len(self.input_shapes) == 1)
        self.input_types = input_types
        self.threshold = threshold
        self.axis = axis
        self.bitwidth = bitwidth

        if self.bitwidth != 8:
            raise NotImplementedError("UnQuantize layer only supports bitwith 8 for now")

        # INITIALIZE
        self.init()


class QuantizeBiasLayer(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic quantization layer for bias inputs which can be implemented on
    various backends (e.g. tensorflow)

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes: List[List[int] or Tuple[int]]
            the shapes for all the inputs
        subgraph (str): Indicates the subgraph this layer belongs to
        input_types (List[str]): the types for all the inputs
        threshold_bias (List[float]): the threshold (upper limit)
            for the bias input of this layer
        threshold_ext (float): the external threshold (upper limit)
            for the input of the layer in which this bias will be added
        bitwidth (int): the bitwidth to be quantized to
        do_rounding (bool): whether to do rounding to integer
            (otherwise, casting is used)
    """

    def __init__(
        self,
        name,
        shape,
        dtype,
        inputs,
        input_shapes,
        subgraph,
        input_types,
        threshold_bias,
        threshold_ext,
        bitwidth,
        do_rounding=False,
    ):
        super().__init__(name, 'QuantizeBias', shape, dtype,
                         inputs, input_shapes, subgraph)

        if self.dtype not in ['int32']:
            raise ValueError(
                f"Invalid quantize bias layer (output) dtype: {self.dtype}, should be `int32`")

        self.input_types = input_types
        self.threshold_bias = threshold_bias
        self.threshold_ext = threshold_ext
        self.bitwidth = bitwidth
        self.do_rounding = do_rounding

        if self.bitwidth != 8:
            raise NotImplementedError("Quantize layer only supports bitwith 8 for now")

        # INITIALIZE
        self.init()


class QuantizeScaleBiasLayer(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic quantization layer for scaling layer bias inputs which can be
    implemented on various backends (e.g. tensorflow)

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes: List[List[int] or Tuple[int]]
        the shapes for all the inputs
        subgraph (str): Indicates the subgraph this layer belongs to
        input_types (List[str]): the types for all the inputs
        th_scale (List[float]): the threshold for the scale to which
            this bias is added
        th_ext (float): the external threshold (upper limit) for the
            input of the layer in which this bias will be added
        bitwidth (int): the bitwidth to be quantized to
        do_rounding (bool): whether to do rounding to integer
            (otherwise, casting is used)
    """

    def __init__(
        self,
        name,
        shape,
        dtype,
        inputs,
        input_shapes,
        input_types,
        subgraph,
        scale,
        postscale_shift,
        th_out,
        bitwidth,
        do_rounding=False
    ):
        super().__init__(name, 'QuantizeScaleBias', shape, dtype,
                         inputs, input_shapes, subgraph)

        if self.dtype not in ['int32', 'int64']:
            raise ValueError(f"Invalid quantize bias layer (output) dtype: {self.dtype}, "
                             "should be `int32` or `int64`")

        self.input_types = input_types
        self.scale = scale
        self.postscale_shift = postscale_shift
        self.th_out = th_out
        self.bitwidth = bitwidth
        self.do_rounding = do_rounding

        if self.bitwidth != 8:
            raise NotImplementedError("Quantize layer only supports bitwith 8 for now")

        # INITIALIZE
        self.init()


class QuantizeInterLayer(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic quantization layer for in between layers which can be implemented
    on various backends (e.g. tensorflow)

    Args:
        name (str): the name of this layer
        shape (List[int] or Tuple[int]): the shape of this layer
        dtype (str): the type of this layer
        inputs (List[str]): the input names of this layer
        input_shapes (List[List[int] or Tuple[int]]): the input shapes
            for all the inputs
        subgraph (str): Indicates the subgraph this layer belongs to
        prescale_shift (int): the right shift parameter before scaling
        scale (int): the scaling parameter
        postscale_shift (int): the right shift parameter for after scaling
        axis (int): specifies the axis on which the scaling should be applied
        bitwidth (int): the bitwidth to be quantized to
    """

    def __init__(
        self,
        name,
        shape,
        dtype,
        inputs,
        input_shapes,
        subgraph,
        prescale_shift,
        scale,
        postscale_shift,
        axis,
        bitwidth,
        relu=False,  # TODO
    ):
        super().__init__(name, 'QuantizeInter', shape, dtype,
                         inputs, input_shapes, subgraph)

        # ! preshift should be zero for 8 bit quantization
        assert(all([val == 0 for val in prescale_shift]))
        # assert(all([val >= 0 for val in scale]))
        assert(all([val >= 0 for val in postscale_shift]))
        assert(axis < len(self.input_shapes[0]))
        assert(len(prescale_shift) == len(scale) == len(postscale_shift))

        if not self.dtype == 'int8':
            raise ValueError("The data type of quantize inter layer should be `int8`, "
                             f"but {self.dtype} was provided")

        self.prescale_shift = prescale_shift
        self.scale = scale
        self.postscale_shift = postscale_shift
        self.axis = axis
        self.bitwidth = bitwidth
        self.relu = relu

        if self.bitwidth != 8:
            raise NotImplementedError("Quantize layer only supports bitwith 8 for now")

        # INITIALIZE
        self.init()
