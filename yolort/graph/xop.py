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
Module for XOp definition
"""
from typing import Callable


class XOp:

    """
    The XOp class manages information about XOperations, like whether the
    layout of the operation can be transformed and whether the operation
    can be transposed
    """

    def __init__(self, name):
        self.name = name
        self.layout_transform_func = None
        self.transpose_transform_func = None

    def has_layout_transform(self) -> bool:
        return self.layout_transform_func is not None

    def get_layout_transform(self) -> Callable:
        return self.layout_transform_func

    def add_layout_transform(
        self,
        layout_transform_func: Callable,
    ) -> None:
        self.layout_transform_func = layout_transform_func

    def has_transpose_transform(self) -> bool:
        return self.transpose_transform_func is not None

    def get_transpose_transform(self) -> Callable:
        return self.transpose_transform_func

    def add_transpose_transform(
        self,
        transpose_transform_func: Callable,
    ) -> None:
        self.transpose_transform_func = transpose_transform_func
