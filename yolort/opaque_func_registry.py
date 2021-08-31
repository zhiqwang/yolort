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
Module for OpaqueFuncRegistry definition and functionality
"""

from typing import List, Callable

from . import libyir

from .type import TypeCode
from .shared.vector import StrVector
from .opaque_func import OpaqueFunc


class OpaqueFuncRegistry:

    def __init__(self):
        self._ofr = libyir.OpaqueFuncRegistry()

    @classmethod
    def from_lib(cls, _ofr: libyir.OpaqueFuncRegistry):
        ofr = OpaqueFuncRegistry.__new__(cls)
        ofr._ofr = _ofr
        return ofr

    def set_func(self, func: OpaqueFunc) -> None:
        self._ofr.set_func(func._of)

    def get_func(self, name: str) -> OpaqueFunc:
        of_ = self._ofr.get_func(name)

    # def __del__(self):
    #     print("Delete OpaqueFuncRegistry")

    @classmethod
    def Register(cls, name: str) -> 'OpaqueFuncRegistry':
        ofr_ = libyir.OpaqueFuncRegistry.Register(name)
        return cls.from_lib(ofr_)

    @classmethod
    def Exists(cls, name: str) -> bool:
        return libyir.OpaqueFuncRegistry.Exists(name)

    @classmethod
    def Get(cls, name: str) -> OpaqueFunc:
        of_ = libyir.OpaqueFuncRegistry.Get(name)
        # args_type_codes_ = libyir.OpaqueFuncRegistry.GetArgsTypeCodes(name)
        return OpaqueFunc.from_lib(of_)  # args_type_codes_)

    @classmethod
    def GetRegisteredFuncs(cls) -> StrVector:
        return StrVector(libyir.OpaqueFuncRegistry.GetRegisteredFuncs())

    @classmethod
    def Size(cls) -> int:
        return libyir.OpaqueFuncRegistry.Size()

    @classmethod
    def Clear(cls) -> int:
        return libyir.OpaqueFuncRegistry.Clear()


def register_opaque_func(
    of_name: str,
    type_codes: List[TypeCode],
):

    def __register_opaque_func(py_func: Callable):
        of = OpaqueFunc(py_func, type_codes)
        OpaqueFuncRegistry.Register(of_name).set_func(of)
        return py_func

    return __register_opaque_func
