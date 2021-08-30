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
Module for shape utility functions
"""


def get_numpy_broadcasted_shape(shape_a, shape_b):
    if len(shape_a) >= len(shape_b):
        lshape = shape_a[:]
        rshape = [None] * (len(shape_a) - len(shape_b)) + shape_b[:]
    else:
        rshape = shape_b[:]
        lshape = [None] * (len(shape_b) - len(shape_a)) + shape_a[:]

    assert len(lshape) == len(rshape)

    reversed_shape = []
    for ls, rs in zip(reversed(lshape), reversed(rshape)):
        if ls == rs or ls in [1, None] or rs in [1, None]:
            if ls is None:
                reversed_shape.append(rs)
            elif rs is None:
                reversed_shape.append(ls)
            else:
                reversed_shape.append(max(ls, rs))
        else:
            raise ValueError("Invalid shapes for broadcasted additions: "
                             f"{shape_a} and {shape_b}")

    return list(reversed(reversed_shape))
