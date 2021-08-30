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
Module for managing xops
"""

from typing import List, Callable

from .xop import XOp


class XOpRegistry:

    class __XOpRegistry:
        """
        Implementation of singleton XopRegistry
        """

        def __init__(self):

            self.xops = {}

        def check_xop(self, xop: str) -> None:
            """
            Check whether the xop exists
            """
            if not self.is_xop(xop):
                raise ValueError(f"Unknown XOp {device}, registered XOps are: {self.get_xop_names()}")

        def is_xop(self, xop: str) -> bool:
            return xop in self.xops

        def get_xop(self, xop: str):
            self.check_xop(xop)
            return self.xops[xop]

        def get_xops(self):
            return self.xops.values()

        def get_xop_names(self) -> List[str]:
            return list(self.xops.keys())

        def register_xop_layout_transform(
            self,
            xop_name: str,
            layout_transform_func: Callable,
        ) -> None:
            """
            Registration of XOp layout transformation function
            """
            if not self.is_xop(xop_name):
                self.xops[xop_name] = XOp(xop_name)

            self.get_xop(xop_name).add_layout_transform(layout_transform_func)

        def get_xops_with_layout_transform(self) -> List[str]:
            return [name for name, xop in self.xops.items() if xop.has_layout_transform()]

        def get_xop_layout_transform(self, xop: str) -> Callable:
            """
            Return the layout transformation function for the given XOp
            """
            self.check_xop(xop)
            return self.xops[xop].get_layout_transform()

        def register_xop_transpose_transform(
            self,
            xop_name: str,
            transpose_transform_func: Callable,
        ) -> None:
            """
            Registration of XOp transpose transformation function
            """
            if not self.is_xop(xop_name):
                self.xops[xop_name] = XOp(xop_name)

            self.get_xop(xop_name).add_transpose_transform(
                transpose_transform_func)

        def get_xops_with_transpose_transform(self) -> List[str]:
            return [name for name, xop in self.xops.items() if xop.has_transpose_transform()]

        def get_xop_transpose_transform(self, xop: str) -> Callable:
            """
            Return the layout transformation function for the given XOp
            """
            self.check_xop(xop)
            return self.xops[xop].get_transpose_transform()

    # storage for the instance reference
    __instance = None

    def __init__(self):
        """
        Create singleton instance
        """
        # Check whether we already have an instance
        if XOpRegistry.__instance is None:
            # Create and remember instance
            XOpRegistry.__instance = XOpRegistry.__XOpRegistry()

        # Store instance reference as the only member in the handle
        self.__dict__['_XOpRegistry__instance'] = XOpRegistry.__instance

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


# Registry decorators
xop_r = XOpRegistry()


def xop_register_op_layout_transform(xop_name: str) -> Callable:
    """
    Return decorator for performing layout transformation on a given
    xop with the provided transformation function
    """
    def register_op_layout_transform_decorator(transform_func: Callable):
        xop_r.register_xop_layout_transform(xop_name, transform_func)

        return transform_func

    return register_op_layout_transform_decorator


def xop_register_op_transpose_transform(xop_name: str) -> Callable:
    """
    Return decorator for performing a transpose transformation on a given
    xop with the provided transformation function
    """

    def register_op_transpose_transform_decorator(transform_func: Callable):
        xop_r.register_xop_transpose_transform(xop_name, transform_func)

        return transform_func

    return register_op_transpose_transform_decorator
