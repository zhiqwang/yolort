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
Utility for topologically sorting a list of XLayer objects
"""

from typing import List

from ..layer import XLayer


def sort_topologically(net: List[XLayer]) -> List[XLayer]:
    """
    Topologically sort a list of XLayer objects in O(N^2)
    """

    top_net = []

    bottoms_cache = {X.name: X.bottoms[:] for X in net}

    while len(net) > len(top_net):

        for X in net:
            if X.name in bottoms_cache and bottoms_cache[X.name] == []:
                top_net.append(X)

                for t in X.tops:
                    bottoms_cache[t].remove(X.name)

                del bottoms_cache[X.name]

    return top_net
