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
Module for Vector definitions
"""

from typing import List

from yolort import libyir


class Vector:

    def __init__(self, vector):
        # self.__vector = vector -> doesn't work because of recursive
        # call to getattr(...)
        self.__dict__['_Vector__vector'] = vector

    def __add__(self, other):
        self.extend(other)
        return self

    def append(self, value: str):
        self.__vector.append(value)

    def __contains__(self, value):
        return value in self.__vector

    def __delitem__(self, value):
        self.__vector.__delitem__(value)

    def __eq__(self, other):
        try:
            return (len(self) == len(other) and all(
                (self.__getitem__(i) == other[i] for i in range(len(self)))))
        except TypeError:
            return False

    def __getattr__(self, attr):
        """
        Delegate access to implementation
        """
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.__vector, attr)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [e for e in self.__vector[key]]
        return self.__vector[key]

    def get_lpx_vector(self):
        return self.__vector

    def index(self, value):
        for i, e in enumerate(self.get_lpx_vector()):
            if e == value:
                return i
        raise ValueError(f"value: {value} not in Vector")

    def __len__(self):
        return len(self.__vector)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return repr(self.__vector)

    def __str__(self):
        return str(self.__vector)

    def __setattr__(self, attr, value):
        """
        Delegate access to implementation
        """
        return setattr(self.__vector, attr, value)

    def __setitem__(self, key, value):
        return self.__vector.__setitem__(key, value)

    def to_list(self):
        return [e for e in self.get_lpx_vector()]


class StrVector(Vector):
    """
    Wrapper class to make libyir.StrVector more python-like
    """

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            value = libyir.StrVector(value)
        return self.get_lpx_vector().__setitem__(key, value)


class IntVector(Vector):
    """
    Wrapper class to make libyir.IntVector more python-like
    """

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            value = libyir.IntVector(value)
        return self.get_lpx_vector().__setitem__(key, value)


class FloatVector(Vector):
    """
    Wrapper class to make libyir.FloatVector more python-like
    """

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            value = libyir.FloatVector(value)
        return self.get_lpx_vector().__setitem__(key, value)


class XBufferVector(Vector):
    """
    Wrapper class to make libyir.XBufferVector more python-like
    """

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            value = libyir.XBufferVector(value)
        return self.get_lpx_vector().__setitem__(key, value)


def make_any_vector(elem_to_any_func, any_to_elem_func):

    class AnyVector(Vector):
        """
        Wrapper class to make libyir.Vector more python-like
        """

        def append(self, value):
            self.get_lpx_vector().append(any_to_elem_func(value))

        def __getitem__(self, key):
            if isinstance(key, slice):
                return [elem_to_any_func(e)
                        for e in self.get_lpx_vector()[key]]
            return elem_to_any_func(self.get_lpx_vector()[key])

        def __setitem__(self, key, value):
            if isinstance(key, slice):
                value = any_to_elem_func(value)
            return self.get_lpx_vector().\
                __setitem__(key, any_to_elem_func(value))

    return AnyVector


class IntVector2D(Vector):
    """
    Wrapper class to make libyir.IntVector2D more python-like
    """

    def __init__(self, vector):
        # self.__vector = vector -> doesn't work because of recursive
        # call to getattr(...)
        # TODO: best approach?
        self.__dict__['_IntVector2D__vector'] = vector
        self.__dict__['_Vector__vector'] = vector

    def append(self, value: List):
        self.__vector.append(libyir.IntVector(value))

    def __contains__(self, value: List):
        if not isinstance(value, list):
            raise TypeError(f"Expecting 'list' argument but got: {type(value)}")
        return libyir.IntVector(value) in self.__vector

    def insert(self, index: int, value):
        self.__vector.insert(index, libyir.IntVector(value))

    def extend(self, value: List):
        self.__vector.extend([libyir.IntVector(v) for v in value])

    def __getattr__(self, attr):
        """
        Delegate access to implementation
        """
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.__vector, attr)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return IntVector2D([IntVector(e) for e in self.__vector[key]])
        return IntVector(self.__vector[key])

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            value = libyir.IntVector2D([libyir.IntVector(v) for v in value])
            self.__vector.__setitem__(key, value)
        else:
            self.__vector.__setitem__(key, libyir.IntVector(value))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"IntVector2D[{', '.join([str(e) for e in self.__vector])}]"

    def to_list(self):
        return [[ii for ii in i] for i in self.get_lpx_vector()]
