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
Module for XGraph graph passes
"""

from typing import Callable, Optional, List, Dict
import abc
import copy
import logging

from .. import XGraph, XLayer
from ..xgraph_factory import XGraphFactory

logger = logging.getLogger("pyxir")


def pass_factory(cls):

    def wrapper(*args, **kwargs):
        obj = cls(*args, **kwargs)
        return obj

    def factory():
        return wrapper

    return factory


class XGraphVisitor:
    """
    Visitor class for visiting XGraph
    """

    def __init__(self):
        self.xgraph = None

    def __call__(self, xgraph: XGraph) -> None:
        """
        Main method to be called on object to start visitor pass
        """
        self.xgraph = xgraph
        for X in self.xgraph.get_layers():
            self.visit(X)
        self.xgraph = None

    def visit(self, X: XLayer) -> XLayer:
        """
        Visit an XLayer
        """
        pass


class XGraphMutator:
    """
    Mutator class for changing XGraph
    """

    def __init__(self):
        self.xgraph = None

    def __call__(self, xgraph: XGraph):
        """
        Main method to be called on object to start mutation pass
        """
        self.xgraph = xgraph
        new_xg = XGraph(self.xgraph.get_name())
        new_xg.copy_meta_attrs(self.xgraph)
        for X in xgraph.get_layers():
            new_X = self.visit(X)
            # if new_X not None
            if new_X:
                # Make sure tops are not set
                new_X.tops = []
                new_xg.add(new_X)
        return new_xg

    def visit(self, X: XLayer) -> XLayer:
        """
        Mutate an XLayer
        """
        return X


class XGraphBasePass:

    __metaclass__ = abc.ABCMeta

    """
    This class is responsible doing graph passing through XGraph objects

    TODO 'replace layer pass' creates a copy but 'optimization pass' doesn't

    Attributes:
        xgraph_factory (XGraphFactory): a factory object for
            constructing XGraph objects
        name (str): the new name of the XGraph pass
        output_png (str): the name of the png file for graph
            visualization if specified
    """

    def __init__(self, name='XGraphBase', output_png=None):

        self.name = name
        self.output_png = output_png

        self.xgraph_factory = XGraphFactory()

    @abc.abstractmethod
    def execute(self, xgraph: XGraph) -> XGraph:
        """
        Execute the XGraph pass, should be overwritten
        """
        raise NotImplementedError

    def _replace_layer_pass(
        self,
        xgraph: XGraph,
        replace_func: Callable,
        name: Optional[str] = None,
        blobs: bool = False,
        output_png: Optional[str] = None,
    ) -> XGraph:
        """
        Wrapper function where replace_func can be used to replace layers in
        the xgraph and this function takes care of the construction of a new
        pydot graph and schedule objects

        Args:
            xgraph (XGraph): the provided graph for the replace layer pass
            replace_func (function): the function to be executed on each
                XLayer object in the graph. This function is expected to
                return a list of new XLayer objects to replace the provided
                ParametersLayer.
            name (str): the name of the adjusted graph
            blobs (bool): whether blobs should be included in the new graph
            output_png (str): if specfied, the name of the png file to save a
                visualization of the graph

        Returns
            new_xgraph (XGraph): A newly created Xgraph
        """
        name = name if name is not None else xgraph.get_name()

        # For mapping layers to the correct bottoms if layers are
        #   replaced/removed
        bottoms_map = {}

        net = []
        time_to_layer = {}
        layer_to_time = {}
        III = 0
        # for idx in range(len(schedule.time_to_layer.keys())):
        for idx, X in enumerate(xgraph.get_layers()):

            P = copy.deepcopy(X)
            # ! It's important that the tops are set to []
            # Later they will be filled again
            P = P._replace(tops=[])

            bottom_Ps = xgraph.get_bottom_layers(P.name)
            top_Ps = xgraph.get_top_layers(P.name)

            # TODO fix top so that they are also always correct -> handle blobs
            # Get bottoms Ps

            logger.info("----------------------")
            logger.info(f"Idx: {idx}, Name: {P.name}")
            logger.info(f"botttom Ps: {[bottom_P.name for bottom_P in bottom_Ps]}")
            logger.info(f"top Ps: {[top_P.name for top_P in top_Ps]}")

            # Call the provided ParametersLayer replace function
            new_Ps = replace_func(bottom_Ps, P, top_Ps)

            if len(new_Ps) == 0:
                # If layer removed, then map this layer to its bottom layer
                if len(P.bottoms) > 1:
                    # warnings.warn("[WARNING] Removing a layer that has"
                    #               " multiple inputs: {} This will propose the"
                    #               " first input of this layer as the input"
                    #               " of the next layer and remove all other"
                    #               " inputs.".format(P.name))
                    bottoms_map[P.name] = bottoms_map[P.bottoms[0]]
                elif len(P.bottoms) == 1:
                    bottoms_map[P.name] = bottoms_map[P.bottoms[0]]

                logger.debug(f"Remove this layer: {P.name}, substitute with "
                             f"bottom: {P.bottoms[0] if len(P.bottoms) > 0 else []}")
                continue

            # Add all the layers to the new network
            for new_P in new_Ps:
                if new_P.name not in layer_to_time:
                    net.append(new_P)
                    time_to_layer[III] = [new_P.name]
                    layer_to_time[new_P.name] = III
                    III += 1

            # Update bottoms and tops
            for i in range(len(new_Ps)):
                # Bottoms
                bottom_names = [bottoms_map[b] if b in bottoms_map else b
                                for b in new_Ps[i].bottoms]
                logger.debug(f"Update bottoms and tops: {new_Ps[i].name}")
                logger.debug(f"-- new bottoms: {bottom_names}")
                idx = layer_to_time[new_Ps[i].name]
                net[idx] = new_Ps[i]._replace(
                    bottoms=bottom_names
                )
                # Bottom tops
                newP_bottoms = [net[layer_to_time[b_name]] for b_name in bottom_names]
                for newP_bottom in newP_bottoms:
                    newP_bottom.tops.append(new_Ps[i].name)

            logger.info(f"new_Ps {[net[layer_to_time[new_P.name]].name for new_P in new_Ps]}")
            logger.info(f"new_Ps bottoms {[net[layer_to_time[new_P.name]].bottoms for new_P in new_Ps]}")

            # Update bottoms map after updating bottoms and tops
            bottoms_map[P.name] = new_Ps[-1].name
            # logger.debug("bottoms_map", bottoms_map)

        logger.info("----------------------")
        logger.info(f"Net: old # elems: {len(xgraph)}")
        logger.info(f"Net: new # elems: {len(net)}")

        new_xgraph = self.xgraph_factory.build_from_xlayer(
            net, name=name, blobs=blobs, output_png=output_png
        )

        return new_xgraph

    def _optimization_layer_pass(
        self,
        xgraph: XGraph,
        condition_funcs: List[Callable],
        opt_funcs: List[Callable],
        opt_names: List[str],
        opt_kwargs_lst: List[Dict],
        repeat_until_stable: bool = False,
        name: str = 'XGraphOptimization',
        output_png: Optional[str] = None,
    ) -> XGraph:
        """
        Wrapper function where opt_funcs can be used to adjust layers in the
        xgraph and this function takes care of passing through the graph

        Args:
            xgraph (XGraph): the provided graph for the replace layer pass
            condition_funcs (List[Callable]): A list of conditional functions.
                Functions return a boolean that indicates whether to execute the
                adaptation function on a certain XLayer. The input is a XLayer
                object.
            opt_funcs (List[Callable]): the optimization functions to be executed
                on each XLayer object in the graph. The input is an XLayer object,
                a list of its bottom layers and a list of its top layers. Should
                return None if the XLayer is to be removed and the XLayer otherwise
            opt_names (List[str]): the names of the optimizations to be performed
            opt_kwargs_lst (List[Dict]): list of kwargs
            repeat_until_stable (bool): whether to repeat the graph optimization pass
                until no more changes are recorded
            name (str): the name of the adjusted graph
            output_png (str): if specfied, the name of the png file to save a
                visualization of the graph

        Returns
            xgraph (XGraph): the adjusted XGraph
        """
        # TODO Multiple optimizations in one pass?

        do_pass = True

        # Enable executing a pass repeatedly if necessary
        while do_pass:

            lx = len(xgraph)
            changes_done = False
            for X_name in xgraph.get_layer_names():

                # tops can be remove inside loop
                if X_name not in xgraph:
                    continue

                X = xgraph.get(X_name)

                # Check all condition/optimization pairs
                for condition_func, opt_func, opt_name, opt_kwargs in zip(
                    condition_funcs, opt_funcs, opt_names, opt_kwargs_lst):

                    if xgraph.exists_layer(X_name):

                        bottom_Xs = xgraph.get_bottom_layers(X.name)
                        top_Xs = xgraph.get_top_layers(X.name)

                        if condition_func is None or condition_func(bottom_Xs, X, top_Xs):
                            logger.debug(f"-- Visit: {X.name} \n-- -- for opt: {opt_name}")

                            changes_done_in_opt = opt_func(xgraph, bottom_Xs, X, top_Xs, **opt_kwargs)
                            changes_done |= changes_done_in_opt

                            if changes_done_in_opt:
                                break
                    else:
                        break

            # If we don't want to repeat a pass or the last pass didn't perform
            #   any changes on the graph, we stop this optimization pass
            if not repeat_until_stable or not changes_done:
                do_pass = False

        if output_png is not None and logger.getEffectiveLevel() <= 10:
            xgraph.visualize(output_png)

        return xgraph
