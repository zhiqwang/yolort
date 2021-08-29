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
Module containing XLayer condition functions for graph optimization
passes
"""

import numpy as np

from ..layer import xlayer


def is_scaling_by_one(bXs, X, tXs):
    """
    Return true if the given XLayer is just scaling by one

    Raises:
        ValueError: if the XLayer data is not of ScaleData type
    """

    if not isinstance(X.data, xlayer.ScaleData):
        raise ValueError(f"Invalid XLayer data attribute type: {type(X.data)}, "
                         f"should be {type(xlayer.ScaleData)}")

    gamma, beta = X.data.gamma, X.data.beta

    return np.all(gamma == 1) and np.all(beta == 0)
