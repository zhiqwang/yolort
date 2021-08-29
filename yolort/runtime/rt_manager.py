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
Module for managing runtimes
"""

from typing import Dict, Callable

from .base_runtime import BaseRuntime
from .runtime_factory import RuntimeFactory


class RtManager:

    class __RtManager:
        """
        Implementation of singleton RtManager
        """

        def __init__(self):
            self.runtimes = {}
            self.runtime_factory = RuntimeFactory()

        def register_rt(
            self,
            rt_name: str,
            rt: BaseRuntime,
            rt_ops: Dict,
        ):
            """
            Registration of runtime with provided name, runtime executable
            graph and runtime operations
            """
            if rt_name in self.runtimes:
                raise ValueError(f"Runtime with name: {rt_name} already registered. "
                                 "Runtime names have to be unique.")

            self.runtimes[rt_name] = rt_ops
            self.runtime_factory.register_exec_graph(rt_name, rt)

        def register_op(
            self,
            rt_name: str,
            op_type: str,
            setup_func: Callable,
        ):
            # type: (str, str, Function) -> None
            """
            Registration of a new runtime operation

            `setup_func` should take in arguments:
                X: XLayer
                layout: str
                input_shapes: dict
                params: dict
                quant_params: QuantParams
            """
            if rt_name not in self.runtimes:
                raise NotImplementedError(
                    f"The provided runtime: {rt_name} is not registered. Please register a runtime "
                    "with the `register_rt` method before adding operations."
                )

            rt_ops = self.runtimes[rt_name]
            if op_type in rt_ops:
                raise ValueError(f"Operation type: {op_type} is already registered. "
                                 "Operation type names have to be unique")

            # Add operation type to runtime operations
            rt_ops[op_type] = setup_func

        def exists_op(self, rt_name: str, op_type: str) -> bool:
            return rt_name in self.runtimes and op_type in self.runtimes[rt_name]

        def unregister_op(self, rt_name: str, op_type: str) -> None:
            if rt_name not in self.runtimes:
                raise NotImplementedError(
                    f"The provided runtime: {rt_name} is not registered. Please register a runtime "
                    "with the `register_rt` method before removing operations.")

            rt_ops = self.runtimes[rt_name]
            if op_type not in rt_ops:
                raise ValueError(f"Trying to unregister non existing operation: {op_type}")
            del rt_ops[op_type]

    # storage for the instance reference
    __instance = None

    def __init__(self):
        """
        Create singleton instance
        """
        # Check whether we already have an instance
        if RtManager.__instance is None:
            # Create and remember instance
            RtManager.__instance = RtManager.__RtManager()

        # Store instance reference as the only member in the handle
        self.__dict__['_Rt_Manager__instance'] = RtManager.__instance

    def __getattr__(self, attr):
        """
        Delegate access to implementation
        """
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """
        Delegate access to implementation
        """
        return setattr(self.__instance, attr, value)
