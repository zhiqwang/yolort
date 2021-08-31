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
Module for testing the OpaqueFuncRegistry data structure
"""

import pytest

from yolort.type import TypeCode
from yolort.opaque_func_registry import OpaqueFuncRegistry, register_opaque_func
from yolort.opaque_func import OpaqueFunc
from yolort.graph import XGraph, XLayer


class TestOpaqueFuncregistry:

    def setup(self):
        OpaqueFuncRegistry.Clear()

    def test_basic_flow(self):

        def py_func(str_name):
            assert str_name == "test"

        of = OpaqueFunc(py_func, [TypeCode.Str])

        assert OpaqueFuncRegistry.Size() == 0
        ofr = OpaqueFuncRegistry.Register('py_func')
        ofr.set_func(of)

        assert OpaqueFuncRegistry.Size() == 1
        assert OpaqueFuncRegistry.GetRegisteredFuncs() == ['py_func']

        of2 = OpaqueFuncRegistry.Get("py_func")
        of2("test")

    def test_registration_flow(self):
        assert OpaqueFuncRegistry.Size() == 0

        @register_opaque_func('py_func', [TypeCode.Str, TypeCode.XGraph])
        def py_func(str_name, xgraph):
            assert xgraph.get_name() == "name1"
            xgraph.set_name(str_name)

        assert OpaqueFuncRegistry.Size() == 1
        assert OpaqueFuncRegistry.GetRegisteredFuncs() == ['py_func']

        xg = XGraph("name1")
        of = OpaqueFuncRegistry.Get("py_func")
        of("name2", xg)
        assert xg.get_name() == "name2"

    def test_registration_flow_same_name(self):
        assert OpaqueFuncRegistry.Size() == 0

        @register_opaque_func('py_func', [TypeCode.Str, TypeCode.XGraph])
        def py_func(str_name, xgraph):
            assert xgraph.get_name() == "name1"
            xgraph.set_name(str_name)

        with pytest.raises(ValueError):
            @register_opaque_func('py_func', [TypeCode.Str, TypeCode.XGraph])
            def py_func(str_name, xgraph):
                assert xgraph.get_name() == "name1"
                xgraph.set_name(str_name)

    def test_multiple_registrations(self):
        assert OpaqueFuncRegistry.Size() == 0

        @register_opaque_func('py_func1', [TypeCode.Str, TypeCode.XGraph])
        def py_func(str_name, xgraph):
            assert xgraph.get_name() == "name1"
            xgraph.set_name(str_name)

        @register_opaque_func('py_func2', [TypeCode.Str, TypeCode.XGraph])
        def py_func2(str_name, xgraph):
            xgraph.add(XLayer(name=str_name, type=[str_name]))

        assert OpaqueFuncRegistry.Size() == 2
        assert set(OpaqueFuncRegistry.GetRegisteredFuncs()) == set(['py_func1', 'py_func2'])

        xg = XGraph("name1")
        of1 = OpaqueFuncRegistry.Get("py_func1")
        of1("name2", xg)
        assert xg.get_name() == "name2"
        assert len(xg) == 0

        of2 = OpaqueFuncRegistry.Get("py_func2")
        of2("x1", xg)
        assert xg.get_name() == "name2"
        assert len(xg) == 1
        assert xg.get('x1').name == 'x1'
        assert xg.get('x1').type == ['x1']
