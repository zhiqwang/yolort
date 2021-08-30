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
Module responsible for basic optimization of XGraph structures
"""

import logging

from .. import optimizations
from ..xgraph_base_optimizer import XGraphBaseOptimizer
from ..xgraph_optimization_pass import XGraphOptimizationPass

logger = logging.getLogger("pyxir")


class XGraphTransposesOptimizer(XGraphBaseOptimizer):

    def __init__(self, xgraph, target=None, copy=False, opt_name=''):
        super().__init__(xgraph, copy)

        # 1. Merge transposes
        opt_pass = XGraphOptimizationPass(
            name='BasicOptimizationPass-1',
            output_png=f'after_{opt_name}_merge_transposes.png',
            repeat_until_stable=True
        )

        logger.info("Add MergeTransposes pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs: all([tX.type[0] == 'Transpose' for tX in tXs]),
            opt_func=optimizations.merge_transposes,
            name='MergeTransposes'
        )

        self.add_optimization_pass(10, opt_pass)

        # 2. Sweep transposes
        opt_pass = XGraphOptimizationPass(
            name='TransposesOptimizationPass-2',
            output_png='after_' + opt_name + '_sweep_transposes.png',
            repeat_until_stable=True
        )

        logger.info("Add SweepTransposesFlow pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs: all([bX.type[0] == 'Transpose' for bX in bXs]),
            opt_func=optimizations.sweep_transposes_flow,
            name='SweepTransposesFlowDirection',
            target=target
        )

        self.add_optimization_pass(20, opt_pass)
