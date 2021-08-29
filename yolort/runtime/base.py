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
Module for converting XLayer to RtLayer (runtime layer) objects
"""

from typing import List, Dict, Callable

import copy
import logging
import warnings

from ..graph import XLayer
from ..shapes import TensorShape
from ..shared import fancy_logging
from ..shared.vector import FloatVector

from .rt_layer import BaseLayer, RtLayer, InputLayer, ConstantLayer

logger = logging.getLogger("pyxir")
fancy_logger = fancy_logging.getLogger("pyxir")


def get_layer(BaseLayer: BaseLayer) -> Callable:

    def _get_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        **kwargs,
    ) -> List[RtLayer]:
        """
        TODO: docstring
        """
        shapes = X.shapes[:]

        return [BaseLayer(
            name=X.name,
            xtype=X.type[0],
            shape=shapes,
            dtype=X.attrs['dtype'] if 'dtype' in X.attrs else 'float32',
            inputs=X.bottoms[:],
            input_shapes=[input_shapes[bottom] for bottom in X.bottoms],
            data=X.data,
            subgraph=X.subgraph,
            attrs=copy.deepcopy(X.attrs)
        )]

    return _get_layer


##################
# INPUTS/OUTPUTS #
##################

def get_input_layer(
    InputLayer: InputLayer,
    ConstantLayer: ConstantLayer,
) -> Callable:

    def _get_input_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        **kwargs,
    ) -> List[RtLayer]:
        """
        TODO: docstring
        """
        assert(len(X.bottoms) == 0)

        if X.name not in params:

            shapes = X.shapes[:]

            layers = [InputLayer(
                name=X.name,
                shape=shapes,
                dtype='float32',
                inputs=[X.name],
                input_shapes=[shapes],
                subgraph=X.subgraph
            )]
        else:
            layers = [ConstantLayer(
                name=X.name,
                shape=X.shapes[:],
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=X.subgraph,
                value=params[X.name]
            )]

        return layers

    return _get_input_layer


def get_constant_layer(ConstantLayer: ConstantLayer) -> Callable:

    def __get_constant_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        **kwargs,
    ) -> List[RtLayer]:
        """
        TODO: docstring
        """
        assert len(X.bottoms) == 0

        shapes = X.shapes[:]

        layers = [ConstantLayer(
            name=X.name,
            shape=shapes,
            dtype=X.attrs['dtype'] if 'dtype' in X.attrs else 'float32',
            inputs=[X.name],
            input_shapes=[],
            subgraph=X.subgraph,
            value=X.data[0]
        )]

        return layers

    return __get_constant_layer


######
# NN #
######

def get_batchnorm_layer(BatchNormLayer, ConstantLayer, ReluLayer):

    def __get_batchnorm_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        **kwargs,
    ) -> List[RtLayer]:
        """
        TODO: not being used for the moment for quantization,
        bn and scale are transformed to scaling layer
        """
        attrs = X.attrs
        op_name = X.name
        use_relu = 'activation' in X.attrs and X.attrs['activation'] == 'ReLU'

        bottoms = X.bottoms[:]

        # gamma & beta
        layers = []

        # TODO: Do this in decorator instead of here
        if len(X.bottoms) == 1:
            mu_name = f"{op_name}_mu"
            var_name = f"{op_name}_variance"
            gamma_name = f"{op_name}_gamma"
            beta_name = f"{op_name}_beta"
            mu, variance = params[mu_name], params[var_name]
            gamma, beta = params[gamma_name], params[beta_name]

            layers.append(ConstantLayer(
                name=mu_name,
                shape=TensorShape(list(mu.shape)),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=X.subgraph,
                value=mu
            ))

            layers.append(ConstantLayer(
                name=var_name,
                shape=TensorShape(list(variance.shape)),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=X.subgraph,
                value=variance
            ))

            layers.append(ConstantLayer(
                name=gamma_name,
                shape=TensorShape(list(gamma.shape)),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=X.subgraph,
                value=gamma
            ))

            layers.append(ConstantLayer(
                name=beta_name,
                shape=TensorShape(list(beta.shape)),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=X.subgraph,
                value=beta
            ))

            bottoms += [mu_name, var_name, gamma_name, beta_name]
            bottom_shapes = {}
            bottom_shapes[mu_name] = TensorShape(list(mu.shape))
            bottom_shapes[var_name] = TensorShape(list(variance.shape))
            bottom_shapes[gamma_name] = TensorShape(list(gamma.shape))
            bottom_shapes[beta_name] = TensorShape(list(beta.shape))
            input_shapes.update(bottom_shapes)

        shapes = X.shapes[:]

        layers.append(BatchNormLayer(
            name=op_name,
            shape=shapes,
            dtype='float32',
            inputs=bottoms,
            input_shapes=[input_shapes[bottom] for bottom in bottoms],
            subgraph=X.subgraph,
            mean=None,
            variance=None,
            gamma=None,
            beta=None,
            variance_epsilon=float(attrs['epsilon']),
            attrs=attrs
        ))

        if use_relu:
            layers.append(ReluLayer(
                name=op_name,
                xtype=X.type[0],
                shape=X.shapes[:],
                dtype='float32',
                inputs=[op_name],
                input_shapes=[X.shapes[:]],
                data=[],
                subgraph=X.subgraph,
                attrs={}
            ))

        return layers

    return __get_batchnorm_layer


def get_bias_add_layer(BiasAddLayer, ConstantLayer):

    def _get_bias_add_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        **kwargs,
    ) -> List[RtLayer]:
        """
        Construct Runtime BiasAdd layer
        """

        op_name = X.name

        bottoms = X.bottoms[:]

        layers = []

        if len(X.bottoms) == 1:
            bias_name = op_name + "_bias"
            bias = params[bias_name]
            # TODO: hack
            # if max(bias.shape) == np.prod(bias.shape):
            #     bias = bias.reshape((-1,))

            layers.append(ConstantLayer(
                name=bias_name,
                shape=TensorShape(list(bias.shape)),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=X.subgraph,
                value=bias
            ))

            X.bottoms.append(bias_name)
            bottom_shapes = {}
            bottom_shapes[bias_name] = TensorShape(list(bias.shape))
            input_shapes.update(bottom_shapes)

        bias_add_layer = get_layer(BiasAddLayer)(
            X, input_shapes, params, **kwargs
        )[0]
        layers.append(bias_add_layer)

        return layers

    return _get_bias_add_layer


def get_dense_layer(DenseLayer, ConstantLayer):

    def _get_dense_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        **kwargs,
    ) -> List[RtLayer]:
        """
        Run one dense step on input with provided parameters:
            Res = inpt*trans(W) + B
        """
        # TODO
        if len(X.bottoms) != 1:
            print(X)
            print(DenseLayer)
            print(ConstantLayer)

        op_name = X.name

        layers = []
        bottoms = X.bottoms[:]

        # TODO: Move this to a decorator
        if len(X.bottoms) == 1:
            weights_name, bias_name = op_name + "_weights", op_name + "_biases"
            W, B = params[weights_name], params[bias_name]

            layers.append(ConstantLayer(
                name=weights_name,
                shape=TensorShape(list(W.shape)),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=X.subgraph,
                value=W
            ))

            layers.append(ConstantLayer(
                name=bias_name,
                shape=TensorShape(list(B.shape)),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=X.subgraph,
                value=B
            ))

            bottoms += [weights_name, bias_name]
            bottom_shapes = {}
            bottom_shapes[weights_name] = TensorShape(list(W.shape))
            bottom_shapes[bias_name] = TensorShape(list(B.shape))
            input_shapes.update(bottom_shapes)

        layers.append(DenseLayer(
            name=op_name,
            shape=X.shapes[:],
            dtype='float32',
            inputs=bottoms,
            input_shapes=[input_shapes[bottom] for bottom in bottoms],
            subgraph=X.subgraph,
            data_layout=X.attrs['data_layout'],
            weights=W,
            kernel_layout=X.attrs['kernel_layout'],
            biases=B,
            use_relu='activation' in X.attrs and
            X.attrs['activation'] == 'ReLU'
        ))
        return layers

    return _get_dense_layer


def get_elementwise_layer(ElementwiseLayer, ReluLayer):

    def _get_elementwise_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        **kwargs,
    ) -> List[RtLayer]:
        # TODO: We are assuming here that the inputs to the elementwise
        #   operation can't be parameters! This is not the most general assumption!
        #   Add support for elementwise operations which include parameters
        op_name = X.name

        use_relu = 'activation' in X.attrs and X.attrs['activation'] == 'ReLU'

        # TODO
        shape = X.shapes[:]

        layers = [ElementwiseLayer(
                name=op_name,
                xtype=X.type[0],
                shape=shape,
                dtype='float32',
                inputs=X.bottoms,
                input_shapes=[input_shapes[bottom] for bottom in X.bottoms],
                data=[],
                subgraph=X.subgraph,
                attrs={
                    'op': 'Add'
                }
        )]

        if use_relu:
            layers.append(ReluLayer(
                name=op_name,
                xtype='ReLU',
                shape=shape,
                dtype='float32',
                inputs=[op_name],
                input_shapes=[shape],
                data=[],
                subgraph=X.subgraph,
                attrs={}
            ))

        return layers

    return _get_elementwise_layer


def get_scaling_layer(ScaleLayer, ConstantLayer, ReluLayer):

    def __get_scaling_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        **kwargs,
    ) -> List[RtLayer]:
        """
        TODO: docstring
        """
        op_name = X.name
        use_relu = 'activation' in X.attrs and X.attrs['activation'] == 'ReLU'
        attrs = X.attrs

        # gamma, beta = params[op_name + '_gamma'], params[op_name + '_beta']
        bottoms = X.bottoms[:]

        # gamma & beta
        layers = []

        # TODO: Do this in decorator instead of here
        if len(X.bottoms) == 1:
            gamma_name = op_name + "_gamma"
            beta_name = op_name + "_beta"
            gamma, beta = params[gamma_name], params[beta_name]

            layers.append(ConstantLayer(
                name=gamma_name,
                shape=TensorShape(list(gamma.shape)),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=X.subgraph,
                value=gamma
            ))

            layers.append(ConstantLayer(
                name=beta_name,
                shape=TensorShape(list(beta.shape)),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=X.subgraph,
                value=beta
            ))

            bottoms += [gamma_name, beta_name]
            bottom_shapes = {}
            bottom_shapes[gamma_name] = TensorShape(list(gamma.shape))
            bottom_shapes[beta_name] = TensorShape(list(beta.shape))
            input_shapes.update(bottom_shapes)

        shapes = X.shapes[:]

        layers.append(ScaleLayer(
            name=op_name,
            shape=shapes,
            dtype='float32',  # Do scaling computation in float32
            inputs=bottoms,
            input_shapes=[input_shapes[bottom] for bottom in bottoms],
            subgraph=X.subgraph,
            attrs=attrs,
            gamma=None,
            beta=None
        ))

        if use_relu:
            layers.append(ReluLayer(
                name=op_name,
                xtype=X.type[0],
                shape=X.shapes[:],
                dtype='float32',
                inputs=[op_name],
                input_shapes=[X.shapes[:]],
                data=[],
                subgraph=X.subgraph,
                attrs={}
            ))

        return layers

    return __get_scaling_layer


################
# CONVOLUTIONS #
################

def get_conv2d_layer(ConvLayer, ConstantLayer):

    def _get_conv2d_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        **kwargs,
    ) -> List[RtLayer]:
        """
        TODO: docstring
        """
        op_name = X.name
        activation = None
        activation_attrs = {}
        if 'activation' in X.attrs and X.attrs['activation'] == 'pReLU':
            assert('alpha' in X.attrs)
            activation = 'leaky_relu'
            activation_attrs['alpha'] = X.attrs['alpha']
        elif 'activation' in X.attrs and X.attrs['activation'] == 'ReLU':
            activation = 'relu'

        logger.debug(X.bottoms)
        assert(len(X.bottoms) in [1, 3])

        layout = X.attrs['data_layout']
        logger.debug("Conv2d layout: {}".format(layout))

        # Padding
        paddings = [list(pad) for pad in X.attrs['padding']]

        if layout == 'NHWC':
            strides = [1, X.attrs['strides'][0], X.attrs['strides'][1], 1]
            dilations = [1, X.attrs['dilation'][0], X.attrs['dilation'][1], 1]
        else:
            strides = [1, 1, X.attrs['strides'][0], X.attrs['strides'][1]]
            dilations = [1, 1, X.attrs['dilation'][0], X.attrs['dilation'][1]]

        layers = []
        bottoms = X.bottoms[:]

        # KERNEL & BIAS

        # TODO: Do this while building graph instead of here
        if len(X.bottoms) == 1:
            kernel_name = op_name + "_kernel"
            bias_name = op_name + "_biases"
            W, B = params[kernel_name], params[bias_name]

            layers.append(ConstantLayer(
                name=kernel_name,
                shape=TensorShape(list(W.shape)),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=X.subgraph,
                value=W
            ))

            layers.append(ConstantLayer(
                name=bias_name,
                shape=TensorShape(list(B.shape)),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=X.subgraph,
                value=B
            ))

            bottoms += [kernel_name, bias_name]
            bottom_shapes = {}
            bottom_shapes[kernel_name] = TensorShape(list(W.shape))
            bottom_shapes[bias_name] = TensorShape(list(B.shape))
            input_shapes.update(bottom_shapes)

        shapes = X.shapes[:]

        layers.append(ConvLayer(
            name=op_name,
            shape=shapes,
            dtype='float32',  # X.attrs['dtype']
            inputs=bottoms,
            input_shapes=[input_shapes[bottom] for bottom in bottoms],
            subgraph=X.subgraph,
            attrs=X.attrs,
            kernel=None,
            kernel_layout=X.attrs['kernel_layout'],  # 'OIHW'
            kernel_groups=X.attrs['groups'],
            biases=None,
            paddings=paddings,
            strides=strides,
            dilations=dilations,
            use_activation=activation,
            activation_attrs=activation_attrs
        ))
        return layers

    return _get_conv2d_layer


def get_conv2d_transpose_layer(Conv2DTransposeLayer, ConstantLayer):

    def __get_conv2d_transpose_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        batch_size: int = -1,
        placeholder: bool = False,
        **kwargs,
    ) -> List[RtLayer]:
        """
        Create an executable conv2d transpose (deconvolution) layer from the
        provided XLayer
        """
        # TODO: Do some checks on input?

        op_name = X.name
        activation = None
        activation_attrs = {}
        if 'activation' in X.attrs and X.attrs['activation'] == 'pReLU':
            assert 'alpha' in X.attrs
            activation = 'leaky_relu'
            activation_attrs['alpha'] = X.attrs['alpha']
        elif 'activation' in X.attrs and X.attrs['activation'] == 'ReLU':
            activation = 'relu'

        logger.debug(X.bottoms)
        assert(len(X.bottoms) in [1, 3])

        layout = X.attrs['data_layout']
        logger.debug(f"Conv2DTranspose layout: {layout}")

        padding = X.attrs['padding']

        if layout == 'NHWC':
            strides = [1, X.attrs['strides'][0], X.attrs['strides'][1], 1]
            dilations = [1, X.attrs['dilation'][0], X.attrs['dilation'][1], 1]
        else:
            strides = [1, 1, X.attrs['strides'][0], X.attrs['strides'][1]]
            dilations = [1, 1, X.attrs['dilation'][0], X.attrs['dilation'][1]]

        layers = []
        bottoms = X.bottoms[:]

        # KERNEL & BIAS

        # TODO: Do this while building graph instead of here
        if len(X.bottoms) == 1:
            kernel_name = op_name + "_kernel"
            bias_name = op_name + "_biases"
            W, B = params[kernel_name], params[bias_name]

            layers.append(ConstantLayer(
                name=kernel_name,
                shape=TensorShape(list(W.shape)),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=X.subgraph,
                value=W
            ))

            layers.append(ConstantLayer(
                name=bias_name,
                shape=TensorShape(list(B.shape)),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=X.subgraph,
                value=B
            ))

            bottoms += [kernel_name, bias_name]
            bottom_shapes = {}
            bottom_shapes[kernel_name] = TensorShape(list(W.shape))
            bottom_shapes[bias_name] = TensorShape(list(B.shape))
            input_shapes.update(bottom_shapes)

        layers.append(Conv2DTransposeLayer(
            name=op_name,
            shape=X.shapes[:],
            dtype='float32',
            inputs=bottoms,
            input_shapes=[input_shapes[bottom] for bottom in bottoms],
            subgraph=X.subgraph,
            attrs=X.attrs,
            kernel=None,
            kernel_layout=X.attrs['kernel_layout'],
            kernel_groups=X.attrs['groups'],
            biases=None,
            paddings=padding,
            strides=strides,
            dilations=dilations,
            use_activation=activation,
            activation_attrs=activation_attrs,
            batch_size=batch_size,
            placeholder=placeholder
        ))

        return layers

    return __get_conv2d_transpose_layer


def get_pooling_layer(PoolingLayer):

    def _get_pooling_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        **kwargs,
    ) -> List[RtLayer]:
        """
        TODO: docstring
        """
        op_name = X.name
        layout = X.attrs['data_layout']

        logger.debug(f"Pooling layer: {X.attrs['pool_type']}")
        if X.attrs['pool_type'] == 'Max':
            op_type = 'Max'
        elif X.attrs['pool_type'] == 'Avg':
            op_type = 'Avg'
        else:
            raise NotImplementedError("Provided pooling operation is not "
                                      f"supported at this moment: {op}")

        full_paddings = [list(pad) for pad in X.attrs['padding']]

        assert(len(X.bottoms) == 1)

        # !! The following handles wrong padding (Caffe issue??) #TODO
        insize_h, insize_w = X.attrs['insize'][0], X.attrs['insize'][1]
        outsize_h, outsize_w = X.attrs['outsize'][0], X.attrs['outsize'][1]
        strides_h, strides_w = X.attrs['strides'][0], X.attrs['strides'][1]
        kernel_h, kernel_w = X.attrs['kernel_size'][0], X.attrs['kernel_size'][1]

        logger.debug(f"Full paddings: {full_paddings}")
        logger.debug(f"X: {X}")

        if layout == 'NCHW':
            pad_h_b, pad_h_a = full_paddings[2][0], full_paddings[2][1]
            pad_w_b, pad_w_a = full_paddings[3][0], full_paddings[3][1]
        else:
            pad_h_b, pad_h_a = full_paddings[1][0], full_paddings[1][1]
            pad_w_b, pad_w_a = full_paddings[2][0], full_paddings[2][1]

        x_h = (outsize_h - 1) * strides_h + kernel_h - pad_h_b - pad_h_a - insize_h
        x_w = (outsize_w - 1) * strides_w + kernel_w - pad_w_b - pad_w_a - insize_w

        logger.debug(f"x_h: {x_h}, x_w: {x_w}")
        if x_h == 1 or x_w == 1:
            warning = (f"[WARNING] Incorrect padding values in Pooling layer: {op_name}"
                       "but we will adjust for it by adding padding!!")
            warnings.warn(warning)
        else:
            # Do not alter the padding, we assume this is going to work, 
            # otherwise error will follow
            x_h, x_w = 0, 0

        shapes = X.shapes[:]
        if layout == 'NHWC':
            ksize = [1, kernel_h, kernel_w, 1]
            paddings = [
                list(full_paddings[0]), 
                [full_paddings[1][0], full_paddings[1][1] + x_h],
                [full_paddings[2][0], full_paddings[2][1] + x_w],
                list(full_paddings[3]),
            ]
            strides = [1, strides_h, strides_w, 1]
        else:
            ksize = [1, 1, kernel_h, kernel_w]
            paddings = [
                list(full_paddings[0]), 
                list(full_paddings[1]),
                [full_paddings[2][0], full_paddings[2][1] + x_h],
                [full_paddings[3][0], full_paddings[3][1] + x_w],
            ]
            strides = [1, 1, strides_h, strides_w]

        layers = []

        # Normal pooling layer
        layers.append(PoolingLayer(
            name=op_name,
            shape=shapes,
            dtype='float32',
            inputs=X.bottoms,
            input_shapes=[input_shapes[bottom] for bottom in X.bottoms],
            subgraph=X.subgraph,
            attrs=X.attrs,
            op=op_type,
            ksize=ksize,
            paddings=paddings,
            strides=strides
        ))
        return layers

    return _get_pooling_layer


################
# Quantization #
################

def get_quantize_layer(QuantizeLayer):

    def __get_quantize_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        **kwargs,
    ) -> List[RtLayer]:
        """
        TODO formalize checks
        """
        assert len(X.bottoms) == 1
        assert X.attrs['axis'] in [0, 1, 2, 3]
        assert X.attrs['quant_bitwidth'] == 8
        assert isinstance(X.attrs['quant_threshold'], FloatVector)

        return [QuantizeLayer(
            name=X.name,
            shape=X.shapes[:],
            dtype=X.attrs['dtype'],
            inputs=X.bottoms,
            input_shapes=[X.shapes[:]],
            subgraph=X.subgraph,
            input_types=X.attrs['input_types'],
            threshold=X.attrs['quant_threshold'],
            axis=X.attrs['axis'],  # TODO
            do_rounding=True,
            bitwidth=8
        )]

    return __get_quantize_layer


def get_unquantize_layer(UnQuantizeLayer):

    def __get_unquantize_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        **kwargs,
    ) -> List[RtLayer]:
        """
        TODO formalize checks
        """
        assert len(X.bottoms) == 1
        assert X.attrs['quant_bitwidth'] == 8
        assert X.attrs['axis'] in [0, 1, 2, 3]
        assert isinstance(X.attrs['quant_threshold'], FloatVector)

        return [UnQuantizeLayer(
            name=X.name,
            shape=X.shapes[:],
            dtype=X.attrs['dtype'],
            inputs=X.bottoms,
            input_shapes=[input_shapes[bottom] for bottom in X.bottoms],
            subgraph=X.subgraph,
            input_types=X.attrs['input_types'],
            threshold=X.attrs['quant_threshold'],
            axis=X.attrs['axis'],  # TODO
            bitwidth=8
        )]

    return __get_unquantize_layer


def get_quantize_bias_layer(QuantizeBiasLayer):

    def __get_quantize_bias_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        **kwargs,
    ) -> List[RtLayer]:
        """
        TODO formalize checks
        """

        assert X.attrs['quant_bitwidth'] == 8
        assert isinstance(X.attrs['quant_threshold'], float)
        assert isinstance(X.attrs['quant_th_params'], FloatVector)

        return [QuantizeBiasLayer(
            name=X.name,
            shape=X.shapes[:],
            dtype=X.attrs['dtype'],  # 'int32',
            inputs=X.bottoms,
            input_shapes=[input_shapes[bottom] for bottom in X.bottoms],
            subgraph=X.subgraph,
            input_types=X.attrs['input_types'],  # ['float32']
            threshold_bias=X.attrs['quant_th_params'],
            threshold_ext=X.attrs['quant_threshold'],
            bitwidth=8,
            do_rounding=True
        )]

    return __get_quantize_bias_layer


def get_quantize_inter_layer(QuantizeInterLayer):

    def __get_quantize_inter_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        **kwargs,
    ) -> List[RtLayer]:
        """
        TODO: docstring
        """
        assert len(X.bottoms) == 1
        assert X.attrs['axis'] == 1
        assert 'quant_prescale_shift' in X.attrs
        assert all([ps == 0 for ps in X.attrs['quant_prescale_shift']])
        assert X.attrs['quant_bitwidth'] == 8
        assert 'quant_scale' in X.attrs
        assert 'quant_postscale_shift' in X.attrs

        return [QuantizeInterLayer(
            name=X.name,
            shape=X.shapes[:],
            dtype=X.attrs['dtype'],  # 'int64',
            inputs=X.bottoms,
            input_shapes=[X.shapes[:]],
            subgraph=X.subgraph,
            prescale_shift=X.attrs['quant_prescale_shift'],
            scale=X.attrs['quant_scale'],  # [32768],
            postscale_shift=X.attrs['quant_postscale_shift'],  # [15]
            axis=X.attrs['axis'],
            bitwidth=8,
            relu='activation' in X.attrs and X.attrs['activation'] == 'ReLU'
        )]

    return __get_quantize_inter_layer


def get_quantize_scale_bias_layer(QuantizeScaleBiasLayer):

    def __get_quantize_scale_bias_layer(
        X: XLayer,
        input_shapes: Dict,
        params: Dict,
        **kwargs,
    ) -> List[RtLayer]:
        """
        TODO formalize checks
        """

        assert X.attrs['quant_bitwidth'] == 8
        assert 'quant_scale' in X.attrs
        assert 'quant_postscale_shift' in X.attrs
        assert 'quant_th_out' in X.attrs

        return [QuantizeScaleBiasLayer(
            name=X.name,
            shape=X.shapes[:],
            dtype=X.attrs['dtype'],  # 'int32',
            inputs=X.bottoms,
            input_shapes=[input_shapes[bottom] for bottom in X.bottoms],
            subgraph=X.subgraph,
            input_types=X.attrs['input_types'],  # ['float32']
            scale=X.attrs['quant_scale'],
            postscale_shift=X.attrs['quant_postscale_shift'],
            th_out=X.attrs['quant_th_out'],
            bitwidth=8,
            do_rounding=True
        )]

    return __get_quantize_scale_bias_layer
