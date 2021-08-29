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

"""Module defining graph optimization passes"""

import logging

from ..passing.base_pass import XGraphBasePass

logger = logging.getLogger("pyxir")


class XGraphOptimizationPass(XGraphBasePass):

    """
    Responsible for optimizing XGraph models through graph passes

    Attributes
    ----------
    """

    def __init__(
        self,
        name='XGraphOptimization',
        output_png=None,
        repeat_until_stable=False,
    ):
        super().__init__(name, output_png=output_png)

        self.repeat_until_stable = repeat_until_stable
        self.optimizations = []

    def add_optimization(
        self,
        condition_func,
        opt_func,
        name,
        **kwargs,
    ):
        self.optimizations.append({
            'condition_func': condition_func,
            'opt_func': opt_func,
            'name': name,
            'kwargs': kwargs
        })

    def execute(self, xgraph):
        """
        TODO:
        """
        condition_funcs = [opt['condition_func'] for opt in self.optimizations]
        opt_funcs = [opt['opt_func'] for opt in self.optimizations]
        names = [opt['name'] for opt in self.optimizations]
        opt_kwargs_lst = [opt['kwargs'] for opt in self.optimizations]

        # Execute all optimization passes
        # for opt_params in self.optimizations:
        #    logger.debug("-- opt: {}".format(opt_params['name']))

        xgraph = self._optimization_layer_pass(
            xgraph=xgraph,
            condition_funcs=condition_funcs,
            opt_funcs=opt_funcs,
            opt_names=names,
            opt_kwargs_lst=opt_kwargs_lst,
            repeat_until_stable=self.repeat_until_stable,
            name=self.name,
            output_png=self.output_png
        )

        return xgraph
