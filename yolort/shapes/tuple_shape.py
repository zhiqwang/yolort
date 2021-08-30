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
Module for shared TupleShape data structure
"""

import numpy as np
from collections import UserList

from ..shared.vector import IntVector
from .tensor_shape import TensorShape


class TupleShape(UserList):

    def __init__(self, lst):
        assert all([isinstance(e, (TensorShape, TupleShape, list, IntVector)) for e in lst])

        self.data = lst

    def __getitem__(self, key):
        """
        Override slice to return a TupleShape
        """
        if isinstance(key, slice):
            return TupleShape(self.data[key])
        return TensorShape(self.data[key])

    def get_size(self):
        return [abs(int(np.prod(shape))) for shape in self.data]

    def _replace(self, i, new_i):
        assert isinstance(new_i, (type(None), int))
        return TupleShape([[e if e != i else new_i for e in dim]
                           for dim in self.data])

    def set_value(self, axis, value):
        """
        Set TensorShape children values at provided axis to `value`
        """
        for shape in self.data:
            shape[axis] = value

    def tolist(self):
        return [[s for s in dim] for dim in self.data]
