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
Module for Device (e.g. CPU, DPUCADX8G, DPUCZDX8G) definition
and functionality
"""

from typing import List, Callable, Optional

from .graph import XGraph, XLayer
from .graph.passing import XGraphVisitor


class Target:
    """
    Data structure for keeping track of a target and corresponding
    xgraph build function, optimizer function and quantization function and
    compilation function

    Args:
        name (str): The name of this target
        xgraph_optimizer (Callable): The optimization function for optimizing
            a xgraph for target backend quantization, compilation and execution
        xgraph_quantizer (Callable): The quantization function for optimizing
            a xgraph for target backend quantization, compilation and execution
        xgraph_optimizer (Callable): The optimization function for optimizing
            a xgraph for target backend quantization, compilation and execution
        xgraph_build_func (Callable): The build function for transforming
            a xgraph for target backend execution
        xgraph_op_support_annotator (Optional[Callable]): The function for
            annoating supported operations in an XGraph.
            Default: None.
    """

    def __init__(
        self,
        name: str,
        xgraph_optimizer: Callable,
        xgraph_quantizer: Callable,
        xgraph_compiler: Callable,
        xgraph_build_func: Callable,
        xgraph_op_support_annotator: Optional[Callable] = None,
    ):
        self.name = name
        self.xgraph_optimizer = xgraph_optimizer
        self.xgraph_quantizer = xgraph_quantizer
        self.xgraph_compiler = xgraph_compiler
        self.xgraph_build_func = xgraph_build_func

        self.xgraph_op_support_annotator = (
            xgraph_op_support_annotator
            if xgraph_op_support_annotator is not None
            else default_op_support_annotator
        )

        self.xop_2_check_func = {}

    def get_xgraph_build_func(self) -> Callable:
        return self.xgraph_build_func

    def get_xgraph_optimizer(self) -> Callable:
        return self.xgraph_optimizer

    def get_xgraph_quantizer(self) -> Callable:
        return self.xgraph_quantizer

    def get_xgraph_compiler(self) -> Callable:
        return self.xgraph_compiler

    def add_op_support_check(
        self,
        xop_name: str,
        check_func: Callable,
    ) -> None:
        """
        Add operation support check for XOp with given name
        """
        if xop_name in self.xop_2_check_func:
            raise ValueError("Could not register check function for operation with "
                             f"name: {xop_name} as a check function for the operation "
                             "has already been registered")

        self.xop_2_check_func[xop_name] = check_func

    def get_supported_op_checks_names(self) -> List[str]:
        """
        Return names of operations that have a registered op support check
        """
        return list(self.xop_2_check_func.keys())

    def annotate_supported_ops(self, xg: XGraph) -> None:
        """
        Method for annotating supported operations in an XGraph
        """
        self.xgraph_op_support_annotator(xg, self)

    def can_execute(
        self,
        X: XLayer,
        bottom_Xs: List[XLayer],
        top_Xs: List[XLayer],
    ) -> bool:
        """
        Check whether this device can execute the given XLayer with provided
        bottoms and tops
        """

        X_type = X.type[0]

        if X_type not in self.xop_2_check_func:
            if 'All' in self.xop_2_check_func:
                return self.xop_2_check_func['All'](X, bottom_Xs, top_Xs)
            return False

        return self.xop_2_check_func[X_type](X, bottom_Xs, top_Xs)


class DefaultOpSupportPass(XGraphVisitor):
    """
    The default operation support pass
    """

    def __init__(self, target: 'Target') -> None:
        super().__init__()
        self.target = target

    def visit(self, X: XLayer) -> None:
        bottom_Xs = self.xgraph.get_bottom_layers(X.name)
        top_Xs = self.xgraph.get_top_layers(X.name)
        if self.target.can_execute(X, bottom_Xs, top_Xs):
            X.targets.append(self.target.name)


def default_op_support_annotator(
    xg: XGraph,
    target: 'Target',
) -> None:
    """
    Default function for annotating supported operations
    """
    DefaultOpSupportPass(target)(xg)
