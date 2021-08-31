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
Module for primitive Container definitions
"""

from yolort import libyir


class StrContainer:

    def __init__(self, s):
        self._str_c = libyir.StrContainer(s)

    @classmethod
    def from_lib(cls, _str_c: libyir.StrContainer) -> 'StrContainer':
        sc = StrContainer.__new__(cls)
        sc._str_c = _str_c
        return sc

    def get_str(self):
        return self._str_c.str

    def __len__(self):
        return len(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return str(self)

    def set_str(self, s):
        self._str_c.str = s

    def __str__(self):
        return self._str_c.str


class BytesContainer:

    def __init__(self, b: bytes):
        self._bytes_c = libyir.BytesContainer(b)

    @classmethod
    def from_lib(cls, _bytes_c: libyir.BytesContainer) -> 'BytesContainer':
        bc = BytesContainer.__new__(cls)
        bc._bytes_c = _bytes_c
        return bc

    def get_bytes(self) -> bytes:
        return self._bytes_c.get_bytes()

    def __len__(self) -> int:
        return len(str(self))

    def __eq__(self, other) -> bool:
        if isinstance(other, bytes):
            return self.get_bytes() == other
        return self.get_bytes() == other.get_bytes()

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return str(self)

    def set_bytes(self, b: bytes):
        self._bytes_c.set_bytes(b)

    def __str__(self) -> str:
        return str(self.get_bytes())
