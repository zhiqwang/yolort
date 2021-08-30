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
Module for OpaqueFunc definition and functionality

OpaqueFunc is inspired by TVM's PackedFunc and allows writing and calling
functions in all languages where this structure is defined (e.g. C++, Python)
"""

from typing import List, Any, Callable

import libpyxir as lpx

from .type import TypeCode
from .graph.xgraph import XGraph
from .shared.xbuffer import XBuffer
from .shared.container import StrContainer, BytesContainer
from .shared.vector import StrVector, IntVector


class OpaqueFunc:

    # TypeCode conversion functions
    # First: C++ -> Python
    # Second: Python -> C++
    type_codes_ = {
        TypeCode.vInt: (
            lambda arg_: IntVector(arg_.ints),
            lambda arg_: lpx.OpaqueValue(lpx.IntVector(arg_))),
        TypeCode.Str: (
            lambda arg_: arg_.s,
            lambda arg_: lpx.OpaqueValue(arg_)),
        TypeCode.Byte: (
            lambda arg_: arg_.bytes,
            lambda arg_: lpx.OpaqueValue(arg_)),
        TypeCode.vStr: (
            lambda arg_: StrVector(arg_.strings),
            lambda arg_: lpx.OpaqueValue(lpx.StrVector(arg_))),
        TypeCode.StrContainer: (
            lambda arg_: StrContainer.from_lib(arg_.str_c),
            lambda arg_: lpx.OpaqueValue(arg_._str_c)),
        TypeCode.BytesContainer: (
            lambda arg_: BytesContainer.from_lib(arg_.bytes_c),
            lambda arg_: lpx.OpaqueValue(arg_._bytes_c)),
        TypeCode.XGraph: (
            lambda arg_: XGraph._from_xgraph(arg_.xg),
            lambda arg_: lpx.OpaqueValue(arg_._xgraph)),
        TypeCode.XBuffer: (
            lambda arg_: XBuffer.from_lib(arg_.xb),
            lambda arg_: lpx.OpaqueValue(arg_._xb)),
        TypeCode.vXBuffer: (
            lambda arg_: [XBuffer.from_lib(e) for e in arg_.xbuffers],
            lambda arg_: lpx.OpaqueValue(
                lpx.XBufferHolderVector([xb._xb for xb in arg_]))),
        TypeCode.OpaqueFunc: (
            lambda arg_: OpaqueFunc.from_lib(arg_.of),
            lambda arg_: lpx.OpaqueValue(arg_._of))
    }

    def __init__(
        self,
        func: Callable = None,
        type_codes: List[TypeCode] = None,
    ) -> None:

        self._of = lpx.OpaqueFunc()
        if type_codes is None:
            type_codes = []

        if func is not None:
            self.set_func(func, type_codes)

    @classmethod
    def from_lib(cls, _of: lpx.OpaqueFunc) -> 'OpaqueFunc':
        of = OpaqueFunc.__new__(cls)
        of._of = _of
        return of

    def set_func(self, func: Callable, type_codes: List[TypeCode]):

        # if type_codes is not None:
        for tc in type_codes:
            if tc not in OpaqueFunc.type_codes_:
                raise NotImplementedError(
                    f"Function with argument of unsupported type: {tc.name} provided")

        def opaque_func_wrapper(args):
            new_args = []

            if type_codes is not None:
                args_type_codes = type_codes
            else:
                args_type_codes = [TypeCode(args[i].get_type_code_int()) for i in range(len(args))]

            for tc, arg_ in zip(args_type_codes, args):
                if tc not in OpaqueFunc.type_codes_:
                    raise ValueError(f"Unsupported type code: {tc}")
                arg_ = OpaqueFunc.type_codes_[tc][0](arg_)
                new_args.append(arg_)

            func(*new_args)

        arg_type_codes_ = lpx.IntVector([tc.value for tc in type_codes])
        self._of.set_func(opaque_func_wrapper, arg_type_codes_)

    def __call__(self, *args: Any) -> None:
        """
        Call internal lib OpaqueFunc with provided args
        """

        args_type_codes = self.get_arg_type_codes()

        if len(args) != len(args_type_codes):
            raise ValueError("Invalid number of arguments detected. OpaqueFunc is "
                             f"expecting {len(args_type_codes)} arguments but got: {len(args)}")

        oa_v = []
        for tc, arg_ in zip(args_type_codes, args):
            if tc not in OpaqueFunc.type_codes_:
                raise ValueError(f"Unsupported type code: {tc}")
            oa_v.append(OpaqueFunc.type_codes_[tc][1](arg_))

        oa = lpx.OpaqueArgs(oa_v)

        self._of(oa)

    def get_arg_type_codes(self):
        return [TypeCode(i) for i in self._of.get_arg_type_codes()]

    def get_nb_type_codes(self):
        return len(self.get_arg_type_codes())

    def __del__(self):
        pass
