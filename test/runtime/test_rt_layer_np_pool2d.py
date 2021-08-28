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

from pyxir.runtime.numpy.rt_layer_np import InputLayer, PoolingLayer


def test_max_pool2d_layer_basic():

    layers = [
        InputLayer(
            name='input',
            shape=TensorShape([1, 2, 3, 3]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([1, 2, 3, 3])],
            subgraph=None
        ),
        PoolingLayer(
            name='pool1',
            shape=TensorShape([1, 2, 2, 2]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([1, 2, 3, 3])],
            subgraph=None,
            attrs={
                'data_layout': 'NCHW',
            },
            op='Max',
            ksize=[2, 2],
            paddings=[[0, 0], [0, 0], [0, 0], [0, 0]],
            strides=[1, 1, 1, 1]
        )
    ]

    inputs = {
        'input': np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[1, -1, 0], [3, 0, -5], [0, 1, 8]],
            ], dtype=np.float32
        ).reshape((1, 2, 3, 3))
    }

    for layer in layers:
        inpts = [inputs[name] for name in layer.inputs]
        outpt = layer.forward_exec(inpts)

        inputs[layer.name] = outpt

    expected_outpt = np.array([[
        [[5., 6.], [8., 9.]],
        [[3., 0.], [3., 8.]]
    ]])

    np.testing.assert_array_equal(outpt, expected_outpt)


def test_max_pool2d_layer_padding():

    layers = [
        InputLayer(
            name='input',
            shape=TensorShape([1, 2, 2, 2]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([1, 2, 2, 2])],
            subgraph=None
        ),
        PoolingLayer(
            name='pool1',
            shape=TensorShape([1, 2, 2, 2]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([1, 2, 2, 2])],
            subgraph=None,
            attrs={
                'data_layout': 'NCHW',
            },
            op='Max',
            ksize=[2, 2],
            paddings=[[0, 0], [0, 0], [1, 0], [1, 0]],
            strides=[1, 1, 1, 1]
        )
    ]

    inputs = {
        'input': np.array(
            [
                [[1, -2], [8, 9]],
                [[1, -1], [0, 1]],
            ], dtype=np.float32
        ).reshape((1, 2, 2, 2))
    }

    for layer in layers:
        inpts = [inputs[name] for name in layer.inputs]
        outpt = layer.forward_exec(inpts)

        inputs[layer.name] = outpt

    expected_outpt = np.array([[
        [[1., 1.], [8., 9.]],
        [[1., 1.], [1., 1.]]
    ]])

    np.testing.assert_array_equal(outpt, expected_outpt)


def test_max_pool2d_layer_stride2():

    layers = [
        InputLayer(
            name='input',
            shape=TensorShape([1, 2, 3, 3]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([1, 2, 3, 3])],
            subgraph=None
        ),
        PoolingLayer(
            name='pool1',
            shape=TensorShape([1, 2, 2, 2]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([1, 2, 3, 3])],
            subgraph=None,
            attrs={
                'data_layout': 'NCHW'
            },
            op='Max',
            ksize=[3, 3],
            paddings=[[0, 0], [0, 0], [1, 1], [1, 1]],
            strides=[1, 1, 2, 2]
        )
    ]

    inputs = {
        'input': np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[1, -1, 0], [3, 0, -5], [0, 1, 8]],
            ], dtype=np.float32
        ).reshape((1, 2, 3, 3))
    }

    for layer in layers:
        inpts = [inputs[name] for name in layer.inputs]
        outpt = layer.forward_exec(inpts)

        inputs[layer.name] = outpt

    expected_outpt = np.array([[
        [[5., 6.], [8., 9.]],
        [[3., 0.], [3., 8.]]
    ]])

    np.testing.assert_array_equal(outpt, expected_outpt)
