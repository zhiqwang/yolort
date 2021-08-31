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
Module for testing the TargetRegistry class
"""

import pytest

from yolort.target import Target
from yolort.target_registry import TargetRegistry, register_op_support_check


class TestTargetRegistry:

    target_registry = TargetRegistry()

    @classmethod
    def setup_class(cls):

        def xgraph_build_func(xgraph):
            raise NotImplementedError

        def xgraph_optimizer(xgraph):
            raise NotImplementedError

        def xgraph_quantizer(xgraph):
            raise NotImplementedError

        def xgraph_compiler(xgraph):
            raise NotImplementedError

        cls.target_registry.register_target(
            'test',
            xgraph_optimizer,
            xgraph_quantizer,
            xgraph_compiler,
            xgraph_build_func,
        )

        @register_op_support_check('test', 'Convolution')
        def conv_op_support(X, bXs, tXs):
            return True

        @register_op_support_check('test', 'Pooling')
        def pooling_op_support(X, bXs, tXs):
            return True

    @classmethod
    def teardown_class(cls):
        cls.target_registry.unregister_target('test')

    def test_initialization(self):
        assert set(TestTargetRegistry.target_registry.get_target_names()) == set(['cpu', 'test'])

    def test_register_target_twice(self):

        with pytest.raises(ValueError, match="Target: test is already registered."):
            def xgraph_build_func(xgraph):
                raise NotImplementedError

            def xgraph_optimizer(xgraph):
                raise NotImplementedError

            def xgraph_quantizer(xgraph):
                raise NotImplementedError

            def xgraph_compiler(xgraph):
                raise NotImplementedError

            TestTargetRegistry.target_registry.register_target(
                'test',
                xgraph_optimizer,
                xgraph_quantizer,
                xgraph_compiler,
                xgraph_build_func
            )

    def test_target_build_func(self):
        bf = TestTargetRegistry.target_registry.get_target_build_func('test')
        assert callable(bf)

    def test_target_optimizer(self):
        of = TestTargetRegistry.target_registry.get_target_optimizer('test')
        assert callable(of)

    def test_target_quantizer(self):
        qf = TestTargetRegistry.target_registry.get_target_quantizer('test')
        assert callable(qf)

    def test_target_compiler(self):
        cf = TestTargetRegistry.target_registry.get_target_compiler('test')
        assert callable(cf)

    def test_get_target(self):
        t = TestTargetRegistry.target_registry.get_target('test')
        assert isinstance(t, Target)

        with pytest.raises(ValueError, match="Unknown target: notarget"):
            t = TestTargetRegistry.target_registry.get_target('notarget')

    def test_register_op_support(self):

        test_ops = TestTargetRegistry.target_registry.get_supported_op_check_names('test')
        assert set(test_ops) == set(['Convolution', 'Pooling'])

        @register_op_support_check('test', 'Test')
        def test_op_support(X, bXs, tXs):
            raise NotImplementedError

        test_ops = TestTargetRegistry.target_registry.get_supported_op_check_names('test')
        assert set(test_ops) == set(['Convolution', 'Pooling', 'Test'])
