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
Module for testing the OpaqueFunc data structure
"""

import numpy as np

from yolort.type import TypeCode
from yolort.opaque_func import OpaqueFunc
from yolort.graph import XGraph, XLayer
from yolort.shared.xbuffer import XBuffer
from yolort.shared.container import StrContainer


class TestOpaqueFunc:

    def test_str_arg(self):

        def py_func(str_name):
            assert str_name == "test"

        of = OpaqueFunc(py_func, [TypeCode.Str])
        of("test")

    def test_strings_arg(self):

        def py_func(str_names):
            assert str_names == ["test1", "test2"]

        of = OpaqueFunc(py_func, [TypeCode.vStr])
        of(["test1", "test2"])

    def test_str_cont_arg(self):

        def py_func(str_cont):
            assert str_cont == "test"
            str_cont.set_str("2")

        of = OpaqueFunc(py_func, [TypeCode.StrContainer])
        s = StrContainer("test")
        of(s)
        assert s == "2"

    def test_xgraph_arg_name(self):

        def py_func(xg):
            assert xg.get_name() == "test"
            xg.set_name("test2")

        xgraph = XGraph("test")
        of = OpaqueFunc(py_func, [TypeCode.XGraph])
        of(xgraph)
        assert xgraph.get_name() == "test2"

    def test_xgraph_layer_add(self):

        def py_func(xg):
            assert xg.get_name() == "test"
            X = XLayer(name='x1', type=['X1'])
            xg.add(X)

        xgraph = XGraph("test")
        assert len(xgraph) == 0
        of = OpaqueFunc(py_func, [TypeCode.XGraph])
        of(xgraph)

        assert xgraph.get_name() == "test"
        assert len(xgraph) == 1
        assert xgraph.get('x1').type == ['X1']

    def test_xbuffer_arg_name(self):

        def py_func(xb):
            np.testing.assert_equal(xb.to_numpy(), np.array([1., 2.], dtype=np.float32))
            xb.copy_from(xb * 2)

        xb = XBuffer(np.array([1., 2.], dtype=np.float32))
        of = OpaqueFunc(py_func, [TypeCode.XBuffer])
        of(xb)
        np.testing.assert_equal(xb.to_numpy(), np.array([2., 4.], dtype=np.float32))

    def test_vector_xbuffer_arg_name(self):

        def py_func(xbuffers):
            assert len(xbuffers) == 2

            np.testing.assert_equal(xbuffers[0].to_numpy(), np.array([1., 2.], dtype=np.float32))
            xbuffers[0].copy_from(xbuffers[0] * 2)

            np.testing.assert_equal(xbuffers[1].to_numpy(), np.array([-1., 0.], dtype=np.float32))
            xbuffers[1].copy_from(xbuffers[1] * 2)

        xb1 = XBuffer(np.array([1., 2.], dtype=np.float32))
        xb2 = XBuffer(np.array([-1., 0.], dtype=np.float32))
        of = OpaqueFunc(py_func, [TypeCode.vXBuffer])
        of([xb1, xb2])

        np.testing.assert_equal(xb1.to_numpy(), np.array([2., 4.], dtype=np.float32))
        np.testing.assert_equal(xb2.to_numpy(), np.array([-2., 0.], dtype=np.float32))

    def test_opaque_func_arg_name(self):

        def py_func(s):
            assert s == "test"

        of = OpaqueFunc(py_func, [TypeCode.Str])

        def new_py_func(s):
            assert s == "test2"

        def py_func_of(of_):
            assert isinstance(of, OpaqueFunc)
            of_.set_func(new_py_func, [TypeCode.Str])
            of_("test2")

        of2 = OpaqueFunc(py_func_of, [TypeCode.OpaqueFunc])
        of2(of)

    def test_opaque_func_arg_name_empty(self):

        of = OpaqueFunc()

        def py_func_of(of_):

            def new_py_func(s):
                assert s == "test2"

            assert isinstance(of, OpaqueFunc)
            of_.set_func(new_py_func, [TypeCode.Str])
            of_("test2")

        of2 = OpaqueFunc(py_func_of, [TypeCode.OpaqueFunc])
        of2(of)
        of("test2")
