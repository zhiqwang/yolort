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
Module for Type related functionality
"""

from enum import Enum


class TypeCode(Enum):
    Int = 0
    vInt = 1
    Float = 2
    vFloat = 3
    Str = 4
    vStr = 5
    StrContainer = 6
    BytesContainer = 7
    XGraph = 8
    XBuffer = 9
    vXBuffer = 10
    OpaqueFunc = 11
    Undefined = 12
    Byte = 101  # Stored in string in C++
