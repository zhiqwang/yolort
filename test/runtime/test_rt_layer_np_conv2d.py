#!/usr/bin/env python
#
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
Module for testing the pyxir numpy runtime
"""

import numpy as np

from pyxir.shapes import TensorShape

from pyxir.runtime.numpy.rt_layer_np import InputLayer, ConvLayer, ConstantLayer


def test_conv2d_layer_basic():

    W = np.reshape(
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
        (2, 1, 2, 2)
    )
    B = np.array([1, -1], dtype=np.float32)

    layers = [
        InputLayer(
            name='input',
            shape=TensorShape([1, 1, 4, 4]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([1, 1, 4, 4])],
            subgraph=None
        ),
        ConstantLayer(
            name='conv1_kernel',
            shape=TensorShape([2, 1, 2, 2]),
            dtype='float32',
            inputs=[],
            input_shapes=[],
            subgraph=None,
            value=W
        ),
        ConstantLayer(
            name='conv1_bias',
            shape=TensorShape([2]),
            dtype='float32',
            inputs=[],
            input_shapes=[],
            subgraph=None,
            value=B
        ),
        ConvLayer(
            name='conv1',
            shape=TensorShape([2, 1, 3, 3]),
            dtype='float32',
            inputs=['input', 'conv1_kernel', 'conv1_bias'],
            input_shapes=[
                TensorShape([1, 1, 4, 4]),
                TensorShape([2, 1, 2, 2]),
                TensorShape([2]),
            ],
            subgraph=None,
            attrs={
                'data_layout': 'NCHW'
            },
            kernel=None,
            kernel_layout='OIHW',
            kernel_groups=1,
            biases=None,
            paddings=[[0, 0], [0, 0], [0, 0], [0, 0]],
            strides=[1, 1, 1, 1],
            dilations=[1, 1, 1, 1]
        )
    ]

    inputs = {
        'input': np.ones((1, 1, 4, 4), dtype=np.float32)
    }

    for layer in layers:
        inpts = [inputs[name] for name in layer.inputs]
        outpt = layer.forward_exec(inpts)

        inputs[layer.name] = outpt

    expected_outpt = np.array([[
        [[11., 11., 11.], [11., 11., 11.], [11., 11., 11.]],
        [[25., 25., 25.], [25., 25., 25.], [25., 25., 25.]]
    ]])

    np.testing.assert_array_equal(outpt, expected_outpt)


def test_conv2d_layer_sym_padding():

    W = np.reshape(
        np.array([[[1, 2], [0, 3]], [[1, -1], [-1, -1]]],
                    dtype=np.float32),
        (2, 1, 2, 2)
    )
    B = np.array([0, 0], dtype=np.float32)

    layers = [
        InputLayer(
            name='input',
            shape=TensorShape([1, 1, 4, 4]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([1, 1, 4, 4])],
            subgraph=None
        ),
        ConstantLayer(
            name='conv1_kernel',
            shape=TensorShape([2, 1, 2, 2]),
            dtype='float32',
            inputs=[],
            input_shapes=[],
            subgraph=None,
            value=W
        ),
        ConstantLayer(
            name='conv1_bias',
            shape=TensorShape([2]),
            dtype='float32',
            inputs=[],
            input_shapes=[],
            subgraph=None,
            value=B
        ),
        ConvLayer(
            name='conv1',
            shape=TensorShape([2, 1, 3, 3]),
            dtype='float32',
            inputs=['input', 'conv1_kernel', 'conv1_bias'],
            input_shapes=[TensorShape([1, 1, 4, 4]),
                            TensorShape([2, 1, 2, 2]),
                            TensorShape([2])],
            subgraph=None,
            attrs={
                'data_layout': 'NCHW'
            },
            kernel=None,
            kernel_layout='OIHW',
            kernel_groups=1,
            biases=None,
            paddings=[[0, 0], [0, 0], [1, 1], [1, 1]],
            strides=[1, 1, 1, 1],
            dilations=[1, 1, 1, 1]
        )
    ]

    inputs = {
        'input': np.array([[2, -1], [0, 8]], dtype=np.float32).reshape(
            (1, 1, 2, 2))
    }

    for layer in layers:
        inpts = [inputs[name] for name in layer.inputs]
        outpt = layer.forward_exec(inpts)

        inputs[layer.name] = outpt

    expected_outpt = np.array([[
        [[6., -3., 0.], [4., 24., -1.], [0., 16., 8.]],
        [[-2., -1., 1.], [-2., -5., -9.], [0., -8., 8.]]
    ]])

    np.testing.assert_array_equal(outpt, expected_outpt)


def test_conv2d_layer_asym_padding():

    W = np.reshape(
        np.array([[[1, 2], [0, 3]], [[1, -1], [-1, -1]]],
                    dtype=np.float32),
        (2, 1, 2, 2)
    )
    B = np.array([0, 0], dtype=np.float32)

    layers = [
        InputLayer(
            name='input',
            shape=TensorShape([1, 1, 4, 4]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([1, 1, 4, 4])],
            subgraph=None
        ),
        ConstantLayer(
            name='conv1_kernel',
            shape=TensorShape([2, 1, 2, 2]),
            dtype='float32',
            inputs=[],
            input_shapes=[],
            subgraph=None,
            value=W
        ),
        ConstantLayer(
            name='conv1_bias',
            shape=TensorShape([2]),
            dtype='float32',
            inputs=[],
            input_shapes=[],
            subgraph=None,
            value=B
        ),
        ConvLayer(
            name='conv1',
            shape=TensorShape([2, 1, 3, 3]),
            dtype='float32',
            inputs=['input', 'conv1_kernel', 'conv1_bias'],
            input_shapes=[TensorShape([1, 1, 4, 4]),
                            TensorShape([2, 1, 2, 2]),
                            TensorShape([2])],
            subgraph=None,
            attrs={
                'data_layout': 'NCHW'
            },
            kernel=None,
            kernel_layout='OIHW',
            kernel_groups=1,
            biases=None,
            paddings=[[0, 0], [0, 0], [0, 1], [0, 1]],
            strides=[1, 1, 1, 1],
            dilations=[1, 1, 1, 1]
        )
    ]

    inputs = {
        'input': np.array([[2, -1], [0, 8]], dtype=np.float32).reshape(
            (1, 1, 2, 2))
    }

    for layer in layers:
        inpts = [inputs[name] for name in layer.inputs]
        outpt = layer.forward_exec(inpts)

        inputs[layer.name] = outpt

    expected_outpt = np.array([[
        [[24., -1.], [16., 8.]],
        [[-5., -9.], [-8., 8.]]
    ]])

    np.testing.assert_array_equal(outpt, expected_outpt)


def test_conv2d_layer_relu():

    W = np.reshape(
        np.array([[[1, 2], [0, 3]], [[1, -1], [-1, -1]]],
                    dtype=np.float32),
        (2, 1, 2, 2)
    )
    B = np.array([0, 0], dtype=np.float32)

    layers = [
        InputLayer(
            name='input',
            shape=TensorShape([1, 1, 4, 4]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([1, 1, 4, 4])],
            subgraph=None
        ),
        ConstantLayer(
            name='conv1_kernel',
            shape=TensorShape([2, 1, 2, 2]),
            dtype='float32',
            inputs=[],
            input_shapes=[],
            subgraph=None,
            value=W
        ),
        ConstantLayer(
            name='conv1_bias',
            shape=TensorShape([2]),
            dtype='float32',
            inputs=[],
            input_shapes=[],
            subgraph=None,
            value=B
        ),
        ConvLayer(
            name='conv1',
            shape=TensorShape([2, 1, 3, 3]),
            dtype='float32',
            inputs=['input', 'conv1_kernel', 'conv1_bias'],
            input_shapes=[TensorShape([1, 1, 4, 4]),
                            TensorShape([2, 1, 2, 2]),
                            TensorShape([2])],
            subgraph=None,
            attrs={
                'data_layout': 'NCHW'
            },
            kernel=None,
            kernel_layout='OIHW',
            kernel_groups=1,
            biases=None,
            paddings=[[0, 0], [0, 0], [1, 1], [1, 1]],
            strides=[1, 1, 1, 1],
            dilations=[1, 1, 1, 1],
            use_activation='relu'
        )
    ]

    inputs = {
        'input': np.array([[2, -1], [0, 8]], dtype=np.float32).reshape(
            (1, 1, 2, 2))
    }

    for layer in layers:
        inpts = [inputs[name] for name in layer.inputs]
        outpt = layer.forward_exec(inpts)

        inputs[layer.name] = outpt

    expected_outpt = np.array([[
        [[6., 0., 0.], [4., 24., 0.], [0., 16., 8.]],
        [[0., 0., 1.], [0., 0., 0.], [0., 0., 8.]]
    ]])

    np.testing.assert_array_equal(outpt, expected_outpt)


def test_conv2d_layer_stride2():

    W = np.reshape(
        np.array([[[1, 2, 0], [0, 3, 3], [1, 1, 1]],
                    [[1, -1, 1], [-1, -1, 1], [1, 1, 1]]],
                    dtype=np.float32),
        (2, 1, 3, 3)
    )
    B = np.array([0, 0], dtype=np.float32)

    layers = [
        InputLayer(
            name='input',
            shape=TensorShape([1, 1, 4, 4]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([1, 1, 4, 4])],
            subgraph=None
        ),
        ConstantLayer(
            name='conv1_kernel',
            shape=TensorShape([2, 1, 3, 3]),
            dtype='float32',
            inputs=[],
            input_shapes=[],
            subgraph=None,
            value=W
        ),
        ConstantLayer(
            name='conv1_bias',
            shape=TensorShape([2]),
            dtype='float32',
            inputs=[],
            input_shapes=[],
            subgraph=None,
            value=B
        ),
        ConvLayer(
            name='conv1',
            shape=TensorShape([2, 1, 2, 2]),
            dtype='float32',
            inputs=['input', 'conv1_kernel', 'conv1_bias'],
            input_shapes=[TensorShape([1, 1, 4, 4]),
                            TensorShape([2, 1, 3, 3]),
                            TensorShape([2])],
            subgraph=None,
            attrs={
                'data_layout': 'NCHW'
            },
            kernel=None,
            kernel_layout='OIHW',
            kernel_groups=1,
            biases=None,
            paddings=[[0, 0], [0, 0], [0, 1], [0, 1]],
            strides=[1, 1, 2, 2],
            dilations=[1, 1, 1, 1]
        )
    ]

    inputs = {
        'input': np.array([[2, -1, 0, 8], [0, 8, 2, -1],
                            [1, 1, 1, 1], [0, -1, -8, -1]],
                            dtype=np.float32).reshape((1, 1, 4, 4))
    }

    for layer in layers:
        inpts = [inputs[name] for name in layer.inputs]
        outpt = layer.forward_exec(inpts)

        inputs[layer.name] = outpt

    expected_outpt = np.array([[
        [[33., 15.], [-24., 0.]],
        [[0., -7.], [-6., 9.]]
    ]])

    np.testing.assert_array_equal(outpt, expected_outpt)
