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
Module for creating XGraph objects
"""

from typing import List

import logging

from . import XGraph, XLayer

logger = logging.getLogger("pyxir")


class XGraphFactory:

    def build_from_xlayer(
        self,
        net: List[XLayer],
        xgraph: XGraph = None,
        name: str = 'xgraph',
        blobs: bool = False,
        output_png: str = None,
    ) -> XGraph:
        """
        Build the XGraph from a list of XLayers
        """
        if xgraph is None:
            xgraph = XGraph(name)

        for X in net:
            # Make sure tops are not set
            X.tops = []
            xgraph.add(X)

        return xgraph
