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
Module for testing the pyxir runtime manager
"""

from typing import List, Dict, Optional
import numpy as np

from yolort.runtime.rt_manager import RtManager
from yolort.runtime.base_runtime import BaseRuntime
from yolort.runtime.runtime_factory import RuntimeFactory


def get_test_op_1(X, layout, input_shapes, params, quant_params):
    raise NotImplementedError("")


def get_test_op_2(X, layout, input_shapes, params, quant_params):
    raise NotImplementedError("")


X_2_T = {
    'TestOp1': get_test_op_1
}


class TestRuntime(BaseRuntime):

    """
    Test Runtime
    """

    def __init__(
        self,
        network: List[dict],
        params: Dict[str, np.ndarray],
        device: str = 'cpu',
        layout: str = 'NCHW',
        quant_loc: Optional[str] = None,
    ):
        super().__init__(network, params, device, layout, quant_loc)

    def _xfdnn_op_to_exec_op(self, op_type: str):
        """
        Overwrites Runtime abstract method.

        Takes a operation type and returns a function of type: 
        (XLayer, Dict[str,List[int]], Dict[str,numpy.ndarray], 
            Dict[str,Dict]) -> List[xf_layer.XfDNNLayer]
        that takes in a parameters layer object, inputs shapes dict, params dict 
        and quantization parameters dict and outputs and returns a list of executable 
        XfDNNLayerTF objects
        """
        if not op_type in X_2_T:
            raise NotImplementedError(
                f"Operation of type: {op_type} is not supported on TestRuntime")

        return X_2_T[op_type]


class TestRtManager:
    @classmethod
    def setUpClass(cls):
        cls.rt_manager.register_rt('test', TestRuntime, X_2_T)

    def __init__(self):
        self.rt_manager = RtManager()
        self.xf_exec_graph_factory = RuntimeFactory()

    def test_register_rt(self):
        rt_manager = self.rt_manager
        xf_exec_graph_factory = self.xf_exec_graph_factory

        assert('test' in rt_manager.runtimes)
        assert('test' in xf_exec_graph_factory._runtimes)

    def test_register_op(self):
        rt_manager = self.rt_manager
        xf_exec_graph_factory = self.xf_exec_graph_factory

        rt_manager.register_op('test', 'TestOp2', get_test_op_2)
        assert('TestOp2' in X_2_T)
