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
Module for building xgraph for cpu execution
"""

import copy

from yolort.graph import XGraph
from yolort.graph.optimization.optimizers.basic_optimizer import XGraphBasicOptimizer


# TODO move functions
def build_for_cpu_execution(xgraph: XGraph, **kwargs) -> XGraph:
    """
    TODO, docstring
    """
    return copy.deepcopy(xgraph)


def cpu_xgraph_optimizer(xgraph, **kwargs):
    """
    Basic xgraph optimizer
    """

    optimizer = XGraphBasicOptimizer(xgraph)
    optimizer.optimize()

    return xgraph


def cpu_xgraph_quantizer(xgraph, **kwargs):
    """
    Basic xgraph quantizer
    """
    return xgraph


def cpu_xgraph_compiler(xgraph, **kwargs):
    """
    Basic xgraph quantizer
    """
    return xgraph
