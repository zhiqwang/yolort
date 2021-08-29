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
Module responsible for optimizing XGraph structures for quantization
"""

import logging

from .. import optimizations, conditions
from ..xgraph_base_optimizer import XGraphBaseOptimizer
from ..xgraph_optimization_pass import XGraphOptimizationPass

logger = logging.getLogger("pyxir")


class QOptimizer(XGraphBaseOptimizer):
    """
    Optimizer built for doing quantization afterwards using internal
    quantization tools
    """

    def __init__(self, xgraph):
        super().__init__(xgraph)

        # 1. Basic Merge/Remove optimizations
        opt_pass = XGraphOptimizationPass(
            name='XDNN-OptimizationPass-1-Basic-Remove-Merge'
            # output_png='after_basic_merge_and_remove.png'
        )

        logger.info("Add RemoveScalingBy1Layers pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs:
                'Scale' in X.type and conditions.is_scaling_by_one(bXs, X, tXs),
            opt_func=optimizations.remove,
            name='RemoveScalingBy1Layers'
        )

        # TODO: always?
        logger.info("Add RemoveCastLayers pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs: 'Cast' in X.type,
            opt_func=optimizations.remove,
            name='RemoveCastLayers'
        )

        logger.info("Add RemoveDropoutLayer pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs: 'Dropout' in X.type,
            opt_func=optimizations.remove,
            name='RemoveDropoutLayer'
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
                'BiasAdd' in X.type or ('Eltwise' in X.type and X.data is not None),
            opt_func=optimizations.merge_bias,
            name='MergeBiasIntoConvDense'
        )
        self.add_optimization_pass(10, opt_pass)

        # 2. CONV/BIAS/BN/SCALE merge optimization
        opt_pass = XGraphOptimizationPass(
            name='XDNN-OptimizationPass-2-Merge_Conv_Bias_BN_Scale',
            # output_png = 'after_merge_conv_bias_bn_scale.png',
            repeat_until_stable=True
        )

        logger.info("Add MergeBiasIntoConvDense pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs: 'Eltwise' in X.type and X.data is not None,
            opt_func=optimizations.merge_bias,
            name='MergeBiasIntoConvDense'
        )

        logger.info("Add MergeBNIntoConv pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs: 'BatchNorm' in X.type,
            opt_func=optimizations.merge_batchnorm_into_conv,
            name='MergeBNIntoConv'
        )

        logger.info("Add MergeScaleIntoConvBN pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs: 'Scale' in X.type,
            opt_func=optimizations.merge_scale_into_conv_bn,
            name='MergeScaleIntoConvBN'
        )
        self.add_optimization_pass(20, opt_pass)

        # 3. More merge optimizations
        opt_pass = XGraphOptimizationPass(
            name='XDNN-OptimizationPass-3-Merge_ReLU',
            output_png='after_q_optimizations.png'
        )

        logger.info("Add MergeRelu pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs: 'ReLU' in X.type,
            opt_func=optimizations.merge_relu,
            name='MergeRelu'
        )

        self.add_optimization_pass(30, opt_pass)
