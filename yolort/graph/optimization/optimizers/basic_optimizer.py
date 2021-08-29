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

"""Module responsible for basic optimization of XGraph objects"""

import logging

from .. import optimizations, conditions
from ..xgraph_base_optimizer import XGraphBaseOptimizer
from ..xgraph_optimization_pass import XGraphOptimizationPass

logger = logging.getLogger("pyxir")


class XGraphBasicOptimizer(XGraphBaseOptimizer):
    """
    Basic optimizer used by most targets
    """

    def __init__(self, xgraph,  copy=False):
        super().__init__(xgraph, copy)

        # 1.
        opt_pass = XGraphOptimizationPass(
            name='BasicOptimizationPass-1',
            output_png='after_basic_optimizations.png'
        )
        
        logger.info("Add RemoveScalingBy1Layers pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs:
                'Scale' in X.type and conditions.is_scaling_by_one(bXs, X, tXs),
            opt_func=optimizations.remove,
            name='RemoveScalingBy1Layers'
        )

        logger.info("Add RemoveDropoutLayers pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs:
                'Dropout' in X.type,
            opt_func=optimizations.remove,
            name='RemoveDropoutLayers'
        )

        # TODO: always?
        logger.info("Add RemoveCastLayers pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs: 'Cast' in X.type,
            opt_func=optimizations.remove,
            name='RemoveCastLayers'
        )

        logger.info("Add MergePaddingIntoConvPool pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs: 'Pad' in X.type,
            opt_func=optimizations.merge_padding,
            name='MergePaddingIntoConvPool'
        )

        logger.info("Add MergeBiasIntoConvDense pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs:
                'BiasAdd' in X.type or
                ('Eltwise' in X.type and X.data is not None),
            opt_func=optimizations.merge_bias,
            name='MergeBiasIntoConvDense'
        )

        self.add_optimization_pass(10, opt_pass)
