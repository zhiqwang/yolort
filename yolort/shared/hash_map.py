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
Module for Map definitions
"""

import libpyxir as lpx

from .vector import Vector, StrVector


class HashMap:

    def __init__(self, h_map_):
        self._map = h_map_

    def __contains__(self, key):
        return key in self._map

    def __delitem__(self, value):
        self._map.__delitem__(value)

    def __eq__(self, other):
        try:
            return (len(self) == len(other) and all(
                (k in other and self.__getitem__(k) == other[k] for k in self.keys())))
        except TypeError:
            return False

    def get(self, key: str):
        try:
            return self.__getitem__(key)
        except KeyError:
            return None

    def __getitem__(self, key):
        if key not in self._map:
            raise ValueError(f"Could not retrieve key: `{key}`")
        return self._map[key]

    def _get_copy(self, key):
        raise NotImplementedError

    def get_lpx_map(self):
        return self._map

    def items(self):
        return ((k, self.__getitem__(k)) for k in self.keys())

    def keys(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)

    def __ne__(self, other):
        return not self.__eq__(other)

    def pop(self, key: str):
        raise NotImplementedError()

    def popitem(self):
        raise NotImplementedError()

    def setdefault(self, key: str, value):
        raise NotImplementedError()

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())

    def __setitem__(self, key, value):
        return self._map.__setitem__(key, value)

    def to_json(self):
        return self.to_dict()

    def to_dict(self):
        d = {}
        for k, v in self.items():
            d[k] = v if not isinstance(v, Vector) else v.to_list()
        return d

    def update(self, d):
        for k, v in list(d.items()):
            self.__setitem__(k, v)

    def values(self):
        return [self.__getitem__(key) for key in self.keys()]

    @classmethod
    def from_dict(cls, d):
        raise NotImplementedError


class MapStrStr(HashMap):

    @classmethod
    def from_dict(cls, d):
        instance = MapStrStr(lpx.MapStrStr())
        instance.update(d)
        return instance


class MapStrVectorStr(HashMap):

    def _get_copy(self, key):
        raise NotImplementedError

    def __getitem__(self, key):
        return StrVector(self._map[key])

    def __setitem__(self, key, value):
        if isinstance(value, list):
            value = lpx.StrVector(value)
        elif isinstance(value, StrVector):
            value = value.get_lpx_vector()
        self._map.__setitem__(key, value)

    @classmethod
    def from_dict(cls, d):
        instance = MapStrVectorStr(lpx.MapStrVectorStr())
        instance.update(d)
        return instance
