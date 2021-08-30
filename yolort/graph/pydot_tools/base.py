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
Module for working with pydot graphs
"""

from typing import List

import pydot
import logging
import traceback

from ..layer.xlayer import XLayer

logger = logging.getLogger("pyxir")


LAYER_STYLE_DEFAULT = {
    'shape': 'record',
    'fillcolor': '#495469',
    'style': 'filled',
}

BLOB_STYLE = {
    'shape': 'octagon',
    'fillcolor': '#E0E0E0',
    'style': 'filled',
}


def get_bottom_layers(
    layer_name: str,
    pydot_graph: pydot.Dot,
) -> List[XLayer]:
    """
    Retrieve the bottom layers for the given layer and pydot graph
    """
    node = pydot_graph.get_node(pydot.quote_if_necessary(layer_name))[0]
    P = node.get('LayerParameter')

    bottom_layers = []
    for bottom_name in P.bottoms:
        try:
            bottom_node = pydot_graph.get_node(pydot.quote_if_necessary(bottom_name))[0]
            bottom_layers.append(bottom_node.get('LayerParameter'))
        except Exception as e:
            print(P.bottoms)
            print(bottom_name)
            raise e

    return bottom_layers


def get_top_layers(
    layer_name: str,
    pydot_graph: pydot.Dot,
) -> List[XLayer]:
    """
    Retrieve the top layers for the given layer and pydot graph
    ! We have to handle graphs with and without blob layers in between
    """
    node = pydot_graph.get_node(pydot.quote_if_necessary(layer_name))[0]
    P = node.get('LayerParameter')

    top_layers = []
    if len(P.tops) == 1 and P.name in P.tops:
        try:
            node = pydot_graph.get_node(
                pydot.quote_if_necessary(layer_name + '_blob'))[0]
            P = node.get('LayerParameter')
        except Exception as e:
            print("Tops", P.tops)
            print("P", P.name)
            raise e
    for top_name in P.tops:
        try:
            top_node = pydot_graph.get_node(
                pydot.quote_if_necessary(top_name))[0]
            top_layers.append(top_node.get('LayerParameter'))
        except Exception as e:
            print("Tops", P.tops)
            print("P", P.name)
            raise e

    return top_layers


def visualize(
    pydot_graph: pydot.Dot,
    outputfile: str,
) -> pydot.Dot:
    """
    Visualize the provided pydot graph
    """
    logger.info(f"Writing graph visualization to {outputfile}")
    ext = outputfile[outputfile.rfind('.')+1:]
    with open(outputfile, 'wb') as fid:
        Memory = {}
        # for n in pydot_graph.get_nodes():
        # Memory[n.get_name()] = copy.copy(n.get('LayerParameter'))

        # del n.obj_dict['attributes']['LayerParameter']
        # logger.debug(n.to_string())

        try:
            WW = pydot_graph.create(format=ext)
            fid.write(WW)
        except Exception as e:
            logger.error(f"Graph visualizer failed with exception: {e}, "
                         f" traceback: {traceback.format_exc()}")

        fid.close()
    return pydot_graph


def remove_node(
    pydot_graph: pydot.Dot,
    node_name: str,
    bottom_Xs: List[XLayer],
    top_Xs: List[XLayer],
) -> pydot.Dot:
    """
    Remove a node from the provided pydot graph

    TODO: test
    """
    if not (len(bottom_Xs) == 1 or len(top_Xs) == 1):
        raise ValueError("Undefined behaviour: can't remove a node if there are "
                         "multiple bottom nodes and multiple top nodes")

    node_name_q = pydot.quote_if_necessary(node_name)
    pydot_graph.del_node(node_name_q)

    for bX in bottom_Xs:
        b_q = pydot.quote_if_necessary(bX.name)
        pydot_graph.del_edge(b_q, node_name_q)

        for tX in top_Xs:
            t_q = pydot.quote_if_necessary(tX.name)
            edge = pydot.Edge(b_q, t_q, label=f"{bX.name}->ID->{tX.name}")
            pydot_graph.add_edge(edge)

        new_tops = [([bXt] if bXt != node_name else [tX.name for tX in top_Xs]) for bXt in bX.tops]
        # flatten
        new_tops = [e for sl in new_tops for e in sl]
        bX.tops = new_tops

    for tX in top_Xs:
        t_q = pydot.quote_if_necessary(tX.name)
        pydot_graph.del_edge(node_name_q, t_q)

        new_bottoms = [([tXb] if tXb != node_name else [bX.name for bX in bottom_Xs])
                       for tXb in tX.bottoms]
        # flatten
        new_bottoms = [e for sl in new_bottoms for e in sl]
        tX.bottoms = new_bottoms

    return pydot_graph


def insert_node(
    pydot_graph: pydot.Dot,
    node_name: str,
    node: pydot.Node,
    bottom_Xs: List[XLayer],
    top_Xs: List[XLayer],
) -> pydot.Dot:
    """
    Insert a node in the provided pydot graph

    TODO: test
    """
    if not (len(bottom_Xs) == 1 and len(top_Xs) == 1):
        raise ValueError("Undefined behaviour: can't insert a node if there are "
                         "multiple bottom nodes or multiple top nodes")

    n_q = pydot.quote_if_necessary(node_name)
    pydot_graph.add_node(node)

    # for bX in bottom_Xs:
    bX = bottom_Xs[0]
    b_q = pydot.quote_if_necessary(bX.name)

    edge = pydot.Edge(b_q, n_q, label=f"{bX.name}->{node_name}")
    pydot_graph.add_edge(edge)

    # for tX in top_Xs:
    tX = top_Xs[0]
    t_q = pydot.quote_if_necessary(tX.name)
    pydot_graph.del_edge(b_q, t_q)

    new_tops = [(bXt if bXt != tX.name else node_name) for bXt in bX.tops]
    bX.tops = new_tops

    # for tX in top_Xs:
    edge = pydot.Edge(n_q, t_q, label=f"{node_name}->{tX.name}")
    pydot_graph.add_edge(edge)
    new_bottoms = [(tXb if tXb != bX.name else node_name) for tXb in tX.bottoms]
    tX.bottoms = new_bottoms

    return pydot_graph
