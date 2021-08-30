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
Module for managing target backends
"""

from typing import List, Callable, Optional

import importlib

from .target import Target


class TargetRegistry:

    class __TargetRegistry:
        """
        Implementation of singleton TargetRegistry
        """

        def __init__(self):
            self.targets = {}

        def get_target_build_func(self, target: str) -> Callable:
            """
            Return the build function that creates a dedicated XGraph from
            the original XGraph for execution on the target backend.
            """
            self.check_target(target)
            return self.targets[target].get_xgraph_build_func()

        def get_target_optimizer(self, target: str) -> Callable:
            """
            Return the optimizer that creates a dedicated XGraph from the
            original XGraph for quantization, compilation for and execution
            on the target backend.
            """
            self.check_target(target)
            return self.targets[target].get_xgraph_optimizer()

        def get_target_quantizer(self, target: str) -> Callable:
            """
            Return the XGraph quantizer for the given target
            """
            self.check_target(target)
            return self.targets[target].get_xgraph_quantizer()

        def get_target_compiler(self, target: str) -> Callable:
            """
            Return the XGraph compiler for the given target
            """
            self.check_target(target)
            return self.targets[target].get_xgraph_compiler()

        def check_target(self, target: str):
            """
            Check whether the target exists
            """
            if not self.is_target(target):
                # Try importing it on the fly
                try:
                    importlib.import_module("yolort.contrib.target." + target.split("-")[0])
                except ModuleNotFoundError:
                    pass
            if not self.is_target(target):
                raise ValueError(f"Unknown target: {target}, registered targets "
                                 f"are: {self.get_target_names()}")

        def check_targets(self, targets: List[str]):
            """ Check whether the targets exists """
            for target in targets:
                self.check_target(target)

        def is_target(self, target: str) -> bool:
            return target in self.targets

        def get_target(self, target_name: str) -> Target:
            self.check_target(target_name)
            return self.targets[target_name]

        def get_targets(self) -> List[Target]:
            return self.targets.values()

        def get_target_names(self) -> List[str]:
            return list(self.targets.keys())

        def add_op_support_check(
            self,
            target_name: str,
            xop_name: str,
            check_func: Callable,
        ):
            """
            Add check function for operation support for provided xop name and target
            """
            target = self.get_target(target_name)
            target.add_op_support_check(xop_name, check_func)

        def get_supported_op_check_names(self, target_name: str) -> List[str]:
            target = self.get_target(target_name)
            return target.get_supported_op_checks_names()

        def annotate_ops(self, xg) -> None:
            """
            Method for annotating operations in the provided XGraph with
            supported targets
            """
            for target in self.get_targets():
                target.annotate_supported_ops(xg)

        def register_target(
            self,
            target: str,
            xgraph_optimizer: Callable,
            xgraph_quantizer: Callable,
            xgraph_compiler: Callable,
            xgraph_build_func: Callable,
            xgraph_op_support_annotator: Optional[Callable] = None,
            skip_if_exists: bool = False,
        ):
            """
            Registration of a target and a corresponding xgraph build
            function which converts and XGraph object into an XGraph
            object that exploits this target

            This function is actually doing partitioning of the xgraph
            for the particular target

            TODO Register target quantizer and compiler
            TODO Allow for multiple targets -> automatic partitioning
            """
            if self.is_target(target):
                if not skip_if_exists:
                    raise ValueError(f"Target: {target} is already registered.")
                return

            self.targets[target] = Target(
                name=target,
                xgraph_build_func=xgraph_build_func,
                xgraph_optimizer=xgraph_optimizer,
                xgraph_quantizer=xgraph_quantizer,
                xgraph_compiler=xgraph_compiler,
                xgraph_op_support_annotator=xgraph_op_support_annotator
            )

        def unregister_target(self, target: str):
            """
            Unregister the provided target
            """
            del self.targets[target]

    # storage for the instance reference
    __instance = None

    def __init__(self):
        """
        Create singleton instance
        """
        # Check whether we already have an instance
        if TargetRegistry.__instance is None:
            # Create and remember instance
            TargetRegistry.__instance = TargetRegistry.__TargetRegistry()

        # Store instance reference as the only member in the handle
        self.__dict__['_TargetRegistry__instance'] = TargetRegistry.__instance

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


def register_op_support_check(target_name: str, xop_name: str):
    """
    Return decorator for registering function to check the provided
    operation for the specified target
    """

    target_registry = TargetRegistry()

    def __register_op_support_check(
        op_support_check_func: Callable,
    ) -> Callable:

        target_registry.add_op_support_check(
            target_name,
            xop_name,
            op_support_check_func,
        )

        return op_support_check_func

    return __register_op_support_check
