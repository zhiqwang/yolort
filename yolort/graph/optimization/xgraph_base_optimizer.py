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
Module responsible for optimizing XGraph objects
"""

import logging

from .. import XGraph
from .xgraph_optimization_pass import XGraphOptimizationPass

logger = logging.getLogger("pyxir")


class XGraphBaseOptimizer:

    """
    Generic optimization class for XGraph objects with the set optimization
    passes

    Args:
        xgraph (XGraph): the XGraph object to be optimized
        copy (bool): whether to perform the optimizations on a copy
            of the XGraph object or on the original XGraph object
    """

    def __init__(self, xgraph: XGraph, copy: bool = False):

        self.xgraph = xgraph if not copy else copy.deepcopy(xgraph)
        self.optimization_passes = {}

    def add_optimization_pass(
        self,
        level: int,
        opt_pass: XGraphOptimizationPass,
    ) -> None:
        assert(isinstance(level, int))
        if level in self.optimization_passes:
            self.optimization_passes[level].append(opt_pass)
        else:
            self.optimization_passes[level] = [opt_pass]

    def optimize(self) -> XGraph:
        """
        Start optimization
        """

        xgraph = self.xgraph

        for idx, (level, opt_passes) in enumerate(
                sorted(self.optimization_passes.items(), key=lambda x: x[0])):

            logger.info(f"Optimization pass: {idx} level: {level}: nb: {len(opt_passes)}")

            for opt_pass in opt_passes:
                # TODO
                if isinstance(opt_pass, XGraphOptimizationPass):
                    logger.info(f"-- Name: {opt_pass.name}")
                    xgraph = opt_pass.execute(xgraph)
                else:
                    xgraph = opt_pass(xgraph)

        return self.xgraph
