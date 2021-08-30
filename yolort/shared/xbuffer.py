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

""" Module for XBuffer definition and functionality """

import numpy as np
import libpyxir as lpx


class XBuffer:

    def __init__(self, ndarray: np.ndarray) -> None:
        self._xb = lpx.XBuffer(ndarray)

    @classmethod
    def from_lib(cls, _xb: lpx.XBuffer) -> 'XBuffer':
        xb = XBuffer.__new__(cls)
        xb._xb = _xb
        return xb

    def to_numpy(self, copy=False):
        return np.array(self._xb, copy=copy)

    def copy_from(self, b):
        np.copyto(
            self.to_numpy(),
            b.to_numpy() if isinstance(b, XBuffer) else b
        )

    def __getattr__(self, attr):
        """
        Delegate attribute retrieval to numpy instatiation of
        internal buffer
        """
        return getattr(self.to_numpy(), attr)

    def __add__(self, b):
        return XBuffer(self.to_numpy() + b)

    def __floordiv__(self, b):
        return XBuffer(self.to_numpy() // b)

    def __mul__(self, b):
        return XBuffer(self.to_numpy() * b)

    def __pow__(self, b):
        return XBuffer(self.to_numpy() ** b)

    def __sub__(self, b):
        return XBuffer(self.to_numpy() - b)

    def __truediv__(self, b):
        return XBuffer(self.to_numpy() / b)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.to_numpy())
