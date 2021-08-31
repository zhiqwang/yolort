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
Module for testing the Pyxir base functionality
"""

import pytest
import numpy as np
import yolort as yrt

from yolort.opaque_func import OpaqueFunc
from yolort.shared.xbuffer import XBuffer
from yolort.opaque_func_registry import OpaqueFuncRegistry
from yolort.graph.xgraph_factory import XGraphFactory

try:
    import tensorflow as tf
    skip_tf = False
except ModuleNotFoundError:
    skip_tf = True


class TestBase:

    xg_factory = XGraphFactory()

    @pytest.mark.skipif(skip_tf, reason="Because tensorflow is not available")
    def test_build_rt_opaque_func_cpu_tf(self):
        tf.compat.v1.reset_default_graph()

        W = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32).reshape(2, 1, 2, 2)
        # B = np.array([0, 0], dtype=np.float32)

        iX = yrt.ops.input('in', [1, 1, 4, 4])
        wX = yrt.ops.constant('w', W)
        # bX = yrt.ops.constant('b', B)
        cX = yrt.ops.conv2d('conv', iX, wX, kernel_size=[2, 2], strides=[1, 1],
                            padding_hw=[0, 0, 0, 0], dilation=[1, 1], groups=1,
                            channels=2, data_layout='NCHW', kernel_layout='OIHW')

        xlayers = [iX, wX, cX]

        xg = TestBase.xg_factory.build_from_xlayer(xlayers)

        of = OpaqueFuncRegistry.Get('pyxir.build_rt')
        rt_of = OpaqueFunc()

        # call opaque func
        of(xg, 'cpu', 'cpu-tf', ['in'], ['conv'], rt_of)

        # By now, the `rt_of` is initialized with the opaque runtime function
        ins = [XBuffer(np.ones((1, 1, 4, 4), dtype=np.float32))]
        outs = [XBuffer(np.empty((1, 2, 3, 3), dtype=np.float32))]

        rt_of(ins, outs)

        assert len(outs) == 1

        expected_outpt = np.array([[
            [[10., 10., 10.], [10., 10., 10.], [10., 10., 10.]],
            [[26., 26., 26.], [26., 26., 26.], [26., 26., 26.]],
        ]])

        np.testing.assert_array_equal(outs[0], expected_outpt)
