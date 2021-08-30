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
Module for testing the runtime factory
"""

import numpy as np

from typing import List, Dict

# ! Important for device registration
import pyxir

from yolort.graph import XGraph
from yolort.graph.layer.xlayer import XLayer, ConvData
from yolort.graph.xgraph_factory import XGraphFactory
from yolort.runtime.base_runtime import BaseRuntime


class BaseRuntimeSubTest(BaseRuntime):

    def __init__(self, name, xgraph: XGraph):
        super().__init__(name, xgraph)

    def _init_net(self, network: List[XLayer], params: Dict[str, np.ndarray]):
        # Do nothing
        pass


class TestBaseRuntime:

    xgraph_factory = XGraphFactory()

    def test_base_runtime_net_params(self):

        xlayers = [
            XLayer(name='in1', type=['Input'], bottoms=[], tops=['conv1'], targets=[]),
            XLayer(name='in2', type=['Input'], bottoms=[], tops=['add1'], targets=[]),
            XLayer(
                name='conv1',
                type=['Convolution'],
                bottoms=['in1'],
                tops=['add1'],
                data=ConvData(
                    weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                    biases=np.array([0., 1.], dtype=np.float32)
                ),
                targets=[],
            ),
            XLayer(
                name='add1',
                type=['Eltwise'],
                bottoms=['conv1', 'in2'],
                tops=['conv2', 'pool1'],
                targets=[],
            ),
            XLayer(
                name='conv2',
                type=['Convolution'],
                bottoms=['add1'],
                tops=['add2'],
                data=ConvData(
                    weights=np.array([[[[4, 5], [6, 7]]]], dtype=np.float32),
                    biases=np.array([0., -1.], dtype=np.float32)
                ),
                targets=[],
            ),
            XLayer(name='pool1', type=['Pooling'], bottoms=['add1'], tops=['add2'], targets=[]),
            XLayer(name='add2', type=['Eltwise'], bottoms=['conv2', 'pool1'], tops=[], targets=[]),
            XLayer(name='pool2', type=['Pooling'], bottoms=['add1'], tops=['pool3'], targets=[]),
            XLayer(name='pool3', type=['Pooling'], bottoms=['pool2'], tops=[], targets=[]),
        ]

        xgraph = self.xgraph_factory.build_from_xlayer(xlayers)
        base_runtime = BaseRuntimeSubTest('test', xgraph)

        # 1.
        net, params = base_runtime._get_net_and_params(xgraph, ['add1'])

        assert len(net) == 4
        np.testing.assert_array_equal(
            params['conv1_kernel'],
            np.array([[[[1, 2], [3, 4]]]],
            dtype=np.float32),
        )
        np.testing.assert_array_equal(
            params['conv1_biases'],
            np.array([0., 1.],
            dtype=np.float32),
        )

        # 2.
        net, params = base_runtime._get_net_and_params(xgraph, ['conv1'])

        assert len(net) == 2
        assert net[0].name == 'in1'
        assert net[1].name == 'conv1'
        np.testing.assert_array_equal(
            params['conv1_kernel'],
            np.array([[[[1, 2], [3, 4]]]],
            dtype=np.float32),
        )
        np.testing.assert_array_equal(
            params['conv1_biases'],
            np.array([0., 1.],
            dtype=np.float32),
        )

        # 3.
        net, params = base_runtime._get_net_and_params(xgraph, ['pool1', 'pool2'])

        assert len(net) == 7
        np.testing.assert_array_equal(
            params['conv1_kernel'],
            np.array([[[[1, 2], [3, 4]]]],
            dtype=np.float32),
        )
        np.testing.assert_array_equal(
            params['conv1_biases'],
            np.array([0., 1.],
            dtype=np.float32),
        )
