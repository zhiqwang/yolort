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
Module for XAttrs wrapper definition
"""

import json

from yolort import libyir

from yolort.shared.hash_map import HashMap, MapStrStr, MapStrVectorStr
from yolort.shared.vector import Vector, StrVector, IntVector, FloatVector, IntVector2D


class XAttrDict:

    def __init__(self, xattr_map: libyir.XAttrMap):
        self._xattr_map = xattr_map

    def clear(self):
        for key in list(self.keys()):
            del self._xattr_map[key]

    def __contains__(self, key: str):
        return key in self._xattr_map

    def copy(self):
        copy_xattr_d = XAttrDict(libyir.XAttrMap())
        for k, v in self.items():
            copy_xattr_d.__setitem__(k, v)
        return copy_xattr_d

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        xa_copy = self.copy()
        memo[id(xa_copy)] = xa_copy
        return xa_copy

    def __delitem__(self, key: str):
        del self._xattr_map[key]

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for k, v in self.items():
            if other[k] != v:
                return False
        return True

    def fromkeys(self):
        raise NotImplementedError()

    def get(self, key: str):
        try:
            return self.__getitem__(key)
        except KeyError:
            return None

    def __getitem__(self, key: str):
        """ Parse and return the attribute corresponding to the given key """
        _xattr = self._xattr_map[key]
        assert _xattr.name == key
        if _xattr.type == 'UNDEFINED':
            return None
        elif _xattr.type == 'BOOL':
            return _xattr.b
        elif _xattr.type == 'INT':
            return _xattr.i
        elif _xattr.type == 'INTS':
            return IntVector(_xattr.ints)
        elif _xattr.type == 'INTS2D':
            return IntVector2D(_xattr.ints2d)
        elif _xattr.type == 'FLOAT':
            return _xattr.f
        elif _xattr.type == 'FLOATS':
            return FloatVector(_xattr.floats)
        elif _xattr.type == 'STRING':
            return _xattr.s
        elif _xattr.type == 'STRINGS':
            return StrVector(_xattr.strings)
        elif _xattr.type == 'MAP_STR_STR':
            return MapStrStr(_xattr.map_str_str)
        elif _xattr.type == 'MAP_STR_VSTR':
            return MapStrVectorStr(_xattr.map_str_vstr)
        else:
            raise NotImplementedError(
                f"Unsupported attribute: {_xattr} of type: {_xattr.type}")

    def _get_copy(self, key: str):
        """ Parse and return a copy of the attribute corresponding
            to the given key """
        _xattr = self._xattr_map[key]
        assert _xattr.name == key
        if _xattr.type == 'UNDEFINED':
            return None
        elif _xattr.type in ['INT', 'FLOAT', 'BOOL']:
            return self.__getitem__(key)
        elif _xattr.type == 'INTS':
            return [i for i in _xattr.ints]
        elif _xattr.type == 'INTS2D':
            return [[ii for ii in i] for i in _xattr.ints2d]
        elif _xattr.type == 'FLOATS':
            return [f for f in _xattr.floats]
        elif _xattr.type == 'STRING':
            return str(_xattr.s)
        elif _xattr.type == 'STRINGS':
            return [s for s in _xattr.strings]
        elif _xattr.type == 'MAP_STR_STR':
            return MapStrStr(_xattr.map_str_str).to_dict()
        elif _xattr.type == 'MAP_STR_VSTR':
            return MapStrVectorStr(_xattr.map_str_vstr).to_dict()
        else:
            raise NotImplementedError(f"Unsupported attribute: {_xattr} of type: {_xattr.type}")

    def _get_xattr_map(self):
        return self._xattr_map

    def items(self):
        return ((k, self.__getitem__(k)) for k in self.keys())

    def keys(self):
        return iter(self._xattr_map)

    def __len__(self):
        return len(self._xattr_map)

    def pop(self, key: str):
        value = self._get_copy(key)
        del self._xattr_map[key]
        return value

    def popitem(self):
        raise NotImplementedError()

    def setdefault(self, key: str, value):
        raise NotImplementedError()

    def __setitem__(self, key: str, value):
        value_error = (
            f"Unsupported value: {value} for attribute: {key}. XAttr values can only "
            "be of types: int, float, str, List[int], List[float], List[str], "
            "List[List[int]], Dict[str, str], Dict[Str, List[Str]]"
        )
        # TODO: improve this code
        if isinstance(value, list):
            if all([isinstance(e, int) for e in value]):
                value = libyir.IntVector(value)
            elif all([isinstance(e, float) for e in value]):
                value = libyir.FloatVector(value)
            elif all([isinstance(e, str) for e in value]):
                value = libyir.StrVector(value)
            elif all([isinstance(e, list) for e in value]):
                if all([[isinstance(ee, int) for ee in e] for e in value]):
                    value = libyir.IntVector2D([libyir.IntVector(v) for v in value])
                else:
                    raise ValueError(value_error)
            elif all([isinstance(e, IntVector) for e in value]):
                value = libyir.IntVector2D([e.get_lpx_vector() for e in value])
            else:
                raise ValueError(value_error)
        elif isinstance(value, dict):
            values = list(value.values())
            if len(values) > 0 and isinstance(list(value.values())[0], str):
                value = MapStrStr.from_dict(value).get_lpx_map()
            else:
                value = MapStrVectorStr.from_dict(value).get_lpx_map()

        elif isinstance(value, (MapStrVectorStr, MapStrStr)):
            value = value.get_lpx_map()
        elif isinstance(value, (StrVector, IntVector, IntVector2D, FloatVector)):
            value = value.get_lpx_vector()
        elif value is not None and not isinstance(value, (float, int, str, StrVector, IntVector,
                                                          IntVector2D, FloatVector)):
            raise ValueError(value_error)

        if value is not None:
            xattr = libyir.XAttr(key, value)
        else:
            xattr = libyir.XAttr(key)

        self._xattr_map[key] = xattr

    def update(self, d):
        for k, v in list(d.items()):
            self.__setitem__(k, v)

    def values(self):
        return [self.__getitem__(key) for key in self.keys()]

    def to_dict(self):
        d = {}
        for k, v in self.items():
            if isinstance(v, Vector):
                d[k] = v.to_list()
            elif isinstance(v, HashMap):
                d[k] = v.to_dict()
            else:
                d[k] = v
        return d

    def __str__(self):
        return str(json.dumps(self.to_dict()))

    def __repr__(self):
        return str(json.dumps(self.to_dict()))

