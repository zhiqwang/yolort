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
Module for creating pydot graphs
"""

from typing import List, Optional
import pydot
import copy
import logging

from ..layer.xlayer import XLayer
from .base import visualize, LAYER_STYLE_DEFAULT

logger = logging.getLogger("pyxir")


class PydotFactory:
    """
    This class is responsible for creating pydot graphs from a given network

    Class attributes
    ----------------
    cm (Tuple[str]): a tuple containing color hex values for pydot fillcolor attribute

    TODO: Use in xfdnn_compiler_tvm
    """

    cm = {
        'input': "#dfdfdfdf", 
        'variable': "#999999",
        'quantize': "#7d8494",
        'default': "#495469", 
        'target': ["#b8545c", "#7e4f75", "#69dfdb", "#5eab99"]
    }

    def __init__(self):
        pass

    def build_from_parameters_layer(
        self,
        net: List[XLayer],
        name: str = 'pydot',
        blobs: bool = False,
        output_png: Optional[str] = None,
    ) -> pydot.Dot:
        """
        TODO
        """
        pdg = pydot.Dot(name, graph_type='digraph', rankdir='BT')

        if blobs:
            raise NotImplementedError("")
            # net = self._add_blobs(net)

        pydot_nodes = []
        pydot_edges = []
        for P in net:
            logger.debug(f"Name: {P.name}, Layer type: {P.type}, Bottoms: {P.bottoms}")

            # if blobs and P.layer_type and 'blob' in P.layer_type:
            #     pydot_attrs = copy.deepcopy(BLOB_STYLE)
            #     pydot_attrs['LayerParameter'] = P
            # else:
            pydot_attrs = copy.copy(LAYER_STYLE_DEFAULT)
            pydot_attrs['LayerParameter'] = P

            # pydot_attrs['fillcolor'] = PydotFactory.cm[1]
            if P.type[0] in ['Input', 'StrInput']:
                pydot_attrs['shape'] = 'oval'
                pydot_attrs['fillcolor'] = PydotFactory.cm['input']
            elif P.type[0] in ['Cvx']:
                pydot_attrs['fillcolor'] = PydotFactory.cm['input']
            elif 'Variable' in P.type:
                pydot_attrs['shape'] = 'oval'
                pydot_attrs['fillcolor'] = PydotFactory.cm['variable']
            elif set(['Quantize', 'UnQuantize', 'QuantizeInter', 'QuantizeBias']) & set(P.type):
                # Quantization node
                pydot_attrs['fillcolor'] = PydotFactory.cm['quantize']
            elif P.target != 'cpu':
                # TODO multiple targets
                pydot_attrs['fillcolor'] = PydotFactory.cm['target'][0]

            node = pydot.Node(P.name, **pydot_attrs)
            pydot_nodes.append(node)

            for bottom in P.bottoms:
                src = bottom

                dst = P.name
                edge_label = f"{src} -> {dst}"
                logger.debug(f"-- Add bottom edge: {edge_label}")
                edge = pydot.Edge(src, dst, label=edge_label)
                pydot_edges.append(edge)

        # Building the pydot graph
        for node in pydot_nodes:
            pdg.add_node(node)

        for edge in pydot_edges:
            pdg.add_edge(edge)

        # Draw the original DAG computation graph
        if output_png is not None:
            if not output_png.endswith('.png'):
                raise ValueError("PydotFactory can only write pydot graphs to "
                                 f"png files but {output_png} was provided.")
            viz_graph = visualize(pdg, output_png)

        return pdg
