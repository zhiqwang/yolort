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
Module for transforming the layout of an XGraph
"""

from typing import List
import logging

from yolort.shared import fancy_logging

from .. import XGraph, XLayer
from ..layer.xlayer import defaultXLayer
from ..passing.base_pass import XGraphBasePass
from ..optimization.optimizers.transposes_optimizer import XGraphTransposesOptimizer
from ..xop_registry import XOpRegistry

logger = logging.getLogger("pyxir")
fancy_logger = fancy_logging.getLogger("pyxir")


class XGraphLayoutTransformationPass(XGraphBasePass):

    """
    Responsible for transforming the layout of layout specific layers in
    the XGraph

    NOTE This creates a new XGraph object

    Arguments
    ---------
    target_data_layout: str,
        the target data layout for transformation
    target: str
        the target device. If specified, only layers which are scheduled
        on this target will be transformed.
    name: str
        the new name of the decorated xgraph
    output_png: str
        the name of the png file for graph visualization if specified
    """

    xop_registry = XOpRegistry()

    def __init__(
        self,
        target_data_layout,
        target=None,
        name='XGraphLayoutTransform',
        output_png=None,
    ):
        super().__init__(name=name, output_png=output_png)

        if target_data_layout not in ['NCHW', 'NHWC']:
            raise ValueError("Unsupported target layout for XGraph layout transformation "
                             f"pass: {target_data_layout}, only `NCHW` and `NHWC` are supported.")
        self.target_data_layout = target_data_layout
        self.target = target

    def execute(
        self,
        xgraph: XGraph,
        subgraphs_only: bool = False,
    ) -> XGraph:
        """
        Execute the transformation pass
        """

        def transform_layers(
            bottom_Xs: List[XLayer],
            X: XLayer,
            top_Xs: List[XLayer],
        ) -> List[XLayer]:
            """
            Transform the layers with a specific layout
            """

            new_Xs = []

            layout_transform_ops = XGraphLayoutTransformationPass.xop_registry.get_xops_with_layout_transform()

            # TODO: make more extensible
            if X.type[0] in layout_transform_ops and (self.target is None or X.target == self.target):

                data_layout = X.attrs['data_layout']

                if data_layout == self.target_data_layout:
                    new_Xs.append(X)
                else:
                    # Bottom transpose
                    axes_b = [data_layout.index(e)
                              for e in self.target_data_layout]
                    tb_name = "{}_bottom_{}-{}".format(
                        X.name,
                        data_layout,
                        self.target_data_layout
                    )

                    input_shapes = bottom_Xs[0].shapes[:]
                    input_sizes = bottom_Xs[0].sizes[:]
                    tb_shape = [input_shapes[i] for i in axes_b]

                    Tb = defaultXLayer()
                    Tb = Tb._replace(
                        name=tb_name,
                        type=['Transpose'],
                        shapes=tb_shape,
                        sizes=input_sizes,
                        layer=[tb_name],
                        # tops !
                        tops=[],
                        bottoms=[bottom_Xs[0].name],
                        internal=1,
                        attrs={'axes': axes_b}
                    )
                    logger.debug(f"Insert bottom transpose: {tb_name}, axes: {axes_b}")

                    # Top transpose
                    axes_t = [self.target_data_layout.index(e) for e in data_layout]

                    tt_name = f"{X.name}_top_{self.target_data_layout}-{data_layout}"

                    input_sizes = X.sizes[:]
                    tt_shape = X.shapes[:]

                    Tt = defaultXLayer()
                    Tt = Tt._replace(
                        name=tt_name,
                        type=['Transpose'],
                        shapes=tt_shape,
                        sizes=input_sizes,
                        layer=[tt_name],
                        # Tops !
                        tops=[],
                        bottoms=[X.name],
                        internal=1,
                        attrs={'axes': axes_t}
                    )
                    logger.debug(f"Insert top transpose: {tt_name}, axes: {axes_t}")

                    # X
                    new_bottoms = [(b if b != bottom_Xs[0].name else tb_name) for b in X.bottoms]

                    # Call operation layout transformation function
                    layout_transform_func = XGraphLayoutTransformationPass.xop_registry.get_xop_layout_transform(X.type[0])

                    layout_transform_func(X, self.target_data_layout)

                    X = X._replace(
                        bottoms=new_bottoms,
                        # tops are filled later TODO
                        tops=[],
                    )
                    X.attrs['data_layout'] = self.target_data_layout

                    new_Xs.append(Tb)
                    new_Xs.append(X)
                    new_Xs.append(Tt)

            else:
                new_Xs.append(X)

            return new_Xs

        fancy_logger.banner("LAYOUT TRANSFORMATION PASS")

        output_png = self.output_png.replace(
            ".", "layout_transformation.") if self.output_png is not None else None

        xgraph = self._replace_layer_pass(
            xgraph=xgraph,
            replace_func=transform_layers,
            name=self.name,
            output_png=output_png
        )

        # Merge transpose layers
        t_optimizer = XGraphTransposesOptimizer(xgraph, target=self.target)
        t_optimizer.optimize()

        return xgraph
