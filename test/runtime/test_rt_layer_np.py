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

from pathlib import Path

import pytest
import numpy as np

from yolort.shapes import TensorShape, TupleShape
from yolort.runtime.numpy.rt_layer_np import (
    InputLayer,
    CvxLayer,
    ConstantLayer,
    DenseLayer,
    OutputLayer,
    FlattenLayer,
    ReluLayer,
    SoftmaxLayer,
    ReshapeLayer,
    SqueezeLayer,
    TransposeLayer,
    TupleLayer,
    TupleGetItemLayer,
)

try:
    from pyxir.io.cvx import ImgLoader
    skip_cvx = False
except ModuleNotFoundError:
    skip_cvx = True

IMAGE_PATH = Path(__file__).parents[1].resolve() / "assets"


def softmax(x):
    x_exp = np.exp(x - np.max(x))
    return x_exp / x_exp.sum()


def test_input_layer():
    layers = [
        InputLayer(
            name='input',
            shape=TensorShape([16]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([16])],
            subgraph=None
        )
    ]

    inpt = np.array(
        [1, -1, 0, 4, -5, 1, 0, 8, 3, -5, 1, 0, 1, 9, -3, -4],
        dtype=np.float32,
    )

    inputs = [inpt]
    for layer in layers:
        outpt = layer.forward_exec(inputs)
        inputs = [outpt]

    expected_outpt = inpt

    np.testing.assert_array_almost_equal(outpt, expected_outpt)


@pytest.mark.skipif(skip_cvx, reason="Skipping Cvx related test because cvx is not available")
def test_cvx_layer_nchw():
    layers = [
        CvxLayer(
            name='cvx',
            xtype='CvxInput',
            shape=TensorShape([1, 3, 225, 225]),
            dtype='float32',
            inputs=['cvx'],
            input_shapes=[],
            data=[],
            subgraph=None,
            attrs={
                'cvx_key': 'scale-0.5',
                'data_layout': 'NCHW'
            }
        )
    ]

    test_img = str(IMAGE_PATH / 'v.png')

    inputs = [test_img]
    for layer in layers:
        outpt = layer.forward_exec(inputs)
        inputs = [outpt]

    assert outpt.shape == (1, 3, 225, 225)


@pytest.mark.skipif(skip_cvx, reason="Skipping Cvx related test because cvx is not available")
def test_cvx_layer_nhwc():
    layers = [
        CvxLayer(
            name='cvx',
            xtype='CvxInput',
            shape=TensorShape([1, 225, 225, 3]),
            dtype='float32',
            inputs=['cvx'],
            input_shapes=[],
            data=[],
            subgraph=None,
            attrs={
                'cvx_key': 'scale-0.5',
                'data_layout': 'NHWC'
            }
        )
    ]

    test_img = str(IMAGE_PATH / 'v.png')

    inputs = [test_img]
    for layer in layers:
        outpt = layer.forward_exec(inputs)
        inputs = [outpt]

    assert outpt.shape == (1, 225, 225, 3)


def test_constant_layer():
    layers = [
        ConstantLayer(
            name='constant',
            shape=TensorShape([16]),
            dtype='float32',
            inputs=[],
            input_shapes=[],
            subgraph=None,
            value=np.array(
                [1, -1, 0, 4, -5, 1, 0, 8, 3, -5, 1, 0, 1, 9, -3, -4],
                dtype=np.float32,
            )
        )
    ]

    inputs = []
    for layer in layers:
        outpt = layer.forward_exec(inputs)
        inputs = [outpt]

    expected_outpt = np.array(
        [1, -1, 0, 4, -5, 1, 0, 8, 3, -5, 1, 0, 1, 9, -3, -4],
        dtype=np.float32,
    )

    np.testing.assert_array_almost_equal(outpt, expected_outpt)


def test_dense_layer():

    W = np.array([[1., 3., 0., -7.], [2., -4., 6., 8.]], dtype=np.float32)
    B = np.array([-1., -1.], dtype=np.float32)

    layers = [
        InputLayer(
            name='input',
            shape=TensorShape([1, 4]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([1, 4])],
            subgraph=None
        ),
        ConstantLayer(
            name='dense1_weights',
            shape=TensorShape([2, 4]),
            dtype='float32',
            inputs=[],
            input_shapes=[],
            subgraph=None,
            value=W
        ),
        ConstantLayer(
            name='dense1_biases',
            shape=TensorShape([2]),
            dtype='float32',
            inputs=[],
            input_shapes=[],
            subgraph=None,
            value=B
        ),
        DenseLayer(
            name='dense1',
            shape=TensorShape([1, 2]),
            dtype='float32',
            inputs=['input', 'dense1_weights', 'dense1_biases'],
            input_shapes=[
                TensorShape([1, 4]),
                TensorShape([2, 4]),
                TensorShape([2]),
            ],
            subgraph=None,
            data_layout='NC',
            weights=W,
            kernel_layout='OI',
            biases=B,
            use_relu=False
        ),
        OutputLayer(
            name='output',
            shape=TensorShape([1, 2]),
            dtype='float32',
            inputs=['dense1'],
            input_shapes=[TensorShape([1, 2])],
            subgraph=None,
        ),
    ]

    inputs = {
        'input': np.ones((1, 4), dtype=np.float32)
    }

    for layer in layers:
        inpts = [inputs[name] for name in layer.inputs]
        outpt = layer.forward_exec(inpts)

        inputs[layer.name] = outpt

    expected_outpt = np.array([[-4.0, 11.]], dtype=np.float32)

    np.testing.assert_array_almost_equal(outpt, expected_outpt)


def test_flatten_layer():
    layers = [
        InputLayer(
            name='input1',
            shape=TensorShape([1, 1, 2, 2]),
            dtype='float32',
            inputs=['input1'],
            input_shapes=[TensorShape([1, 1, 2, 2])],
            subgraph=None
        ),
        FlattenLayer(
            name='flatten',
            xtype='Flatten',
            shape=TensorShape([1, 2, 2]),
            dtype='float32',
            inputs=['input1'],
            input_shapes=[TensorShape([1, 1, 2, 2])],
            data=[],
            subgraph=None,
            attrs={}
        )
    ]

    inputs = [np.array([[[[1, 2], [3, 4]]]])]
    inpt1 = layers[0].forward_exec(inputs)
    outpt = layers[1].forward_exec([inpt1])

    expected_outpt = np.array([[[1, 2], [3, 4]]])

    np.testing.assert_array_equal(outpt, expected_outpt)


def test_output_layer():
    layers = [
        InputLayer(
            name='output',
            shape=TensorShape([16]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([16])],
            subgraph=None
        )
    ]

    inpt = np.array(
        [1, -1, 0, 4, -5, 1, 0, 8, 3, -5, 1, 0, 1, 9, -3, -4],
        dtype=np.float32,
    )

    inputs = [inpt]
    for layer in layers:
        outpt = layer.forward_exec(inputs)
        inputs = [outpt]

    expected_outpt = inpt

    np.testing.assert_array_almost_equal(outpt, expected_outpt)



def test_relu():
    layers = [
        InputLayer(
            name='input',
            shape=TensorShape([1, 1, 4, 4]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([1, 1, 4, 4])],
            subgraph=None
        ),
        ReluLayer(
            name='relu1',
            xtype='ReLU',
            shape=TensorShape([1, 1, 4, 4]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([1, 1, 4, 4])],
            data=[],
            subgraph=None,
            attrs={}
        )
    ]

    inputs = [
        np.array(
            [1, -1, 0, 4, -5, 1, 0, 8, 3, -5, 1, 0, 1, 9, -3, -4],
            dtype=np.float32
        ).reshape(1, 1, 4, 4)
    ]
    for layer in layers:
        outpt = layer.forward_exec(inputs)
        inputs = [outpt]

    expected_outpt = np.array([[[[1, 0, 0, 4],
                                    [0, 1, 0, 8],
                                    [3, 0, 1, 0],
                                    [1, 9, 0, 0]]]])

    np.testing.assert_array_equal(outpt, expected_outpt)


def test_reshape_layer():
    layers = [
        InputLayer(
            name='input1',
            shape=TensorShape([1, 1, 2, 2]),
            dtype='float32',
            inputs=['input1'],
            input_shapes=[TensorShape([1, 1, 2, 2])],
            subgraph=None
        ),
        ReshapeLayer(
            name='reshape',
            xtype='Reshape',
            shape=TensorShape([1, 2, 2]),
            dtype='float32',
            inputs=['input1'],
            input_shapes=[TensorShape([1, 1, 2, 2])],
            subgraph=None,
            data=[],
            attrs={
                'shape': [1, 2, 2]
            }
        )
    ]

    inputs = [np.array([[[[1, 2], [3, 4]]]])]
    inpt1 = layers[0].forward_exec(inputs)
    outpt = layers[1].forward_exec([inpt1])

    expected_outpt = np.array([[[1, 2], [3, 4]]])

    np.testing.assert_array_equal(outpt, expected_outpt)


def test_softmax_layer():
    layers = [
        InputLayer(
            name='input',
            shape=TensorShape([16]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([16])],
            subgraph=None
        ),
        SoftmaxLayer(
            name='softmax1',
            xtype='Softmax',
            shape=TensorShape([16]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([16])],
            data=[],
            subgraph=None,
            attrs={}
        )
    ]

    inpt = np.array(
        [1, -1, 0, 4, -5, 1, 0, 8, 3, -5, 1, 0, 1, 9, -3, -4],
        dtype=np.float32,
    )

    inputs = [inpt]
    for layer in layers:
        outpt = layer.forward_exec(inputs)
        inputs = [outpt]

    expected_outpt = softmax(inpt)

    np.testing.assert_array_almost_equal(outpt, expected_outpt)


def test_squeeze_layer():
    layers = [
        InputLayer(
            name='input',
            shape=TensorShape([1, 1, 1, 4, 4]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([16])],
            subgraph=None
        ),
        SqueezeLayer(
            name='squeeze1',
            xtype='Squeeze',
            shape=TensorShape([1, 4, 4]),
            dtype='float32',
            inputs=['input'],
            input_shapes=[TensorShape([1, 1, 1, 4, 4])],
            data=[],
            subgraph=None,
            attrs={
                'axis': [1, 2]
            }
        )
    ]

    inpt = np.array(
        [1, -1, 0, 4, -5, 1, 0, 8, 3, -5, 1, 0, 1, 9, -3, -4],
        dtype=np.float32
    ).reshape(1, 1, 1, 4, 4)

    inputs = [inpt]
    for layer in layers:
        outpt = layer.forward_exec(inputs)
        inputs = [outpt]

    expected_outpt = np.array(
        [1, -1, 0, 4, -5, 1, 0, 8, 3, -5, 1, 0, 1, 9, -3, -4],
        dtype=np.float32
    ).reshape(1, 4, 4)

    np.testing.assert_array_almost_equal(outpt, expected_outpt)


def test_transpose_layer():
    layers = [
        InputLayer(
            name='input1',
            shape=TensorShape([1, 1, 2, 2]),
            dtype='float32',
            inputs=['input1'],
            input_shapes=[TensorShape([1, 1, 2, 2])],
            subgraph=None
        ),
        TransposeLayer(
            name='transpose',
            xtype='Transpose',
            shape=TensorShape([1, 2, 2, 1]),
            dtype='float32',
            inputs=['input1'],
            input_shapes=[TensorShape([1, 1, 2, 2])],
            subgraph=None,
            data=[],
            attrs={
                'axes': [0, 2, 3, 1]
            }
        )
    ]

    inputs = [np.array([[[[1, 2], [3, 4]]]])]
    inpt1 = layers[0].forward_exec(inputs)
    outpt = layers[1].forward_exec([inpt1])

    expected_outpt = np.transpose(inputs[0], (0, 2, 3, 1))

    np.testing.assert_array_equal(outpt, expected_outpt)


def test_tuple_get_item():
    A = np.array([0.1, 0.05], dtype=np.float32)
    B = np.array([0.1, 0.05, 0.1], dtype=np.float32)

    layers = [
        TupleLayer(
            name='tuple',
            xtype='Tuple',
            shape=TupleShape([TensorShape([2]), TensorShape([3])]),
            dtype='float32',
            inputs=['input1', 'input2'],
            input_shapes=[TensorShape([2]), TensorShape([3])],
            data=[],
            attrs={},
            subgraph=None
        ),
        TupleGetItemLayer(
            name='tgi',
            xtype='TupleGetItem',
            shape=TensorShape([3]),
            dtype='float32',
            inputs=['tuple'],
            input_shapes=[TupleShape([TensorShape([2]), TensorShape([3])])],
            subgraph=None,
            data=[],
            attrs={
                'index': 1
            }
        )
    ]

    tupl = layers[0].forward_exec([A, B])
    outpt = layers[1].forward_exec([tupl])

    np.testing.assert_array_equal(outpt, B)


def test_tuple_get_item_transpose():
    A = np.ones((1, 4, 4, 3), dtype=np.float32)
    B = np.ones((1, 4, 4, 3), dtype=np.float32)

    layers = [
        TupleLayer(
            name='tuple',
            xtype='Tuple',
            shape=TupleShape([TensorShape([1, 4, 4, 3]), TensorShape([1, 4, 4, 3])]),
            dtype='float32',
            inputs=['input1', 'input2'],
            input_shapes=[TensorShape([1, 4, 4, 3]), TensorShape([1, 4, 4, 3])],
            data=[],
            attrs={},
            subgraph=None
        ),
        TupleGetItemLayer(
            name='tgi',
            xtype='TupleGetItem',
            shape=TensorShape([1, 3, 4, 4]),
            dtype='float32',
            inputs=['tuple'],
            input_shapes=[TupleShape([TensorShape([1, 4, 4, 3]),
                                        TensorShape([1, 4, 4, 3])])],
            subgraph=None,
            data=[],
            attrs={
                'index': 1,
                'transpose': True,
                'axes': [0, 3, 1, 2]
            }
        )
    ]

    tupl = layers[0].forward_exec([A, B])
    outpt = layers[1].forward_exec([tupl])

    assert outpt.shape == (1, 3, 4, 4)
