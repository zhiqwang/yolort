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
Module for shared TensorShape data structure
"""

import numpy as np

from collections import UserList


class TensorShape(UserList):

    def __init__(self, lst):
        if not all([isinstance(e, (int, type(None))) for e in lst]):
            raise ValueError("Invalid tensor shape, expecting list of integers "
                             f"or None, but got: {lst}")

        self.data = lst

    def __getitem__(self, key):
        """
        Override slice to return a TensorShape
        """
        if isinstance(key, slice):
            return TensorShape(self.data[key])
        return self.data[key]

    def get_size(self):
        return [abs(int(np.prod(self.data)))]

    def _replace(self, i, new_i):
        assert isinstance(new_i, (type(None), int))
        return TensorShape([dim if dim != i else new_i for dim in self.data])

    def set_value(self, axis, value):
        """
        Set element at provided axis to `value`
        """
        if not isinstance(value, int):
            raise ValueError(f"Shape values should be integers but got: {type(value)}")
        self.data[axis] = value

    def tolist(self):
        return self.data[:]
