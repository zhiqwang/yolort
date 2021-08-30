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

"""Module for partitioning XGraph objects"""

import copy
import logging

from typing import List, Optional

from yolort.shapes import TupleShape

from .. import XGraph
from ..xgraph_factory import XGraphFactory
from ..layer import xlayer
from ..optimization.optimizers.transposes_optimizer import XGraphTransposesOptimizer

logger = logging.getLogger("pyxir")


class XGraphPartitioner:
    """Partition an XGraph for a given target"""

    xgraph_factory = XGraphFactory()

    def __init__(self):
        pass

    def partition(
        self,
        xgraph: XGraph,
        targets: List[str],
        last_layer: Optional[str] = None
    ) -> XGraph:
        """
        Partition the provided XGraph according to the provided targets

        NOTE: This is a naive approach which only handles one target
        and just partitions everything it can

        Args:
            xgraph (XGraph):the XGraph to be partitioned
            targets (List[int]):the targets to be partitioned for,
                for now only one target can be passed
            last_layer (str):the last layer to be partitioned
        """
        # TODO Add Base partitioning support for multiple algorithms
        if len(targets) != 1:
            raise NotImplementedError(
                "XGraph partitioning is only supported "
                f"for one target at the moment but got: {len(targets)}"
            )
        target = targets[0]

        # ALGO:
        # 1. loop through all the XLayers
        # 2. For every layer, if it's a target layer, either add it to the
        #   corresponding Partition layer if a bottom is already added to
        #   this layer, else start a new Partition layer
        # TODO set bottoms and tops to [] for partition input respectively
        #   output layers
        name, idx = "xp", 0
        name_2_pars = {}
        par_2_names = {}
        xlayers = []
        # Keep track of layers that have a non-target bottom
        #  e.g. T1 --> NT --> T3 --> T4
        #        `--->  T2 ----^
        # Here, T3 has a non-target bottom layer and a new partition has to be
        #   started
        non_target_bottom_layers = set([])
        # Keep track of layer partition dependencies
        #  e.g. T1 --> NT --> T2 -->
        #        `------------^
        # Here, T1 and T2 can't belong to the same partition because of NT in between

        # Keep track of what partitions a layer depends on (i.e. a partition contains layers
        #   that come after another partition but there are some unsupported layers in between)
        # Partitions that depend on eachother can't be merged together
        partition_dep_map = {}

        logger.debug("Partition for target: {}".format(target))

        stop_partitioning = False

        for X in xgraph.get_layers():

            logger.debug("----------------")
            logger.debug(X.name)

            partition_dependencies = set(
                [
                    e
                    for b in X.bottoms
                    if b in partition_dep_map
                    for e in partition_dep_map[b]
                ]
            )
            if X.name in partition_dep_map:
                partition_dep_map[X.name] |= partition_dependencies
            else:
                partition_dep_map[X.name] = partition_dependencies

            if target not in X.targets or stop_partitioning:

                if X.name in name_2_pars:
                    # Cap partition(s)

                    for p in name_2_pars[X.name]:
                        par_2_names[p].remove(X.name)

                    # This layer depends on these partitions
                    partition_dep_map[X.name] = set(name_2_pars[X.name])

                    del name_2_pars[X.name]

                # for t in X.tops:
                #     non_target_bottom_layers.add(t)

                # partition_dependencies = set([e for b in X.bottoms
                #                               if b in name_2_pars
                #                               for e in name_2_pars[b]])
                # if X.name in partition_dep_map:
                #     partition_dep_map[X.name] |= partition_dependencies

                continue

            elif (
                X.name in name_2_pars
                and X.name not in non_target_bottom_layers
                and partition_dep_map[X.name].isdisjoint(set(name_2_pars[X.name]))
            ):

                # Check whether only one partition added this layer or one or more of the partitions
                #   depend on eachother
                if len(name_2_pars[X.name]) == 1:
                    # or not partition_dep_map[X.name].isdisjoint(set(name_2_pars[X.name])):
                    # -- p_0 --> A
                    logger.debug("Add to partition: {}".format(name_2_pars[X.name]))
                    pass
                else:
                    # -- p_0 --> A
                    # -- p_1 ----^
                    new_p_name = name + str(idx)
                    idx += 1
                    logger.debug("Merge into new partition: {}".format(new_p_name))

                    par_2_names[new_p_name] = []

                    for p in name_2_pars[X.name]:

                        par_2_names[new_p_name].extend(par_2_names[p])

                        for layer in par_2_names[p]:
                            name_2_pars[layer] = [new_p_name]

                        # xlayers.remove(pX)
                        del par_2_names[p]

                    name_2_pars[X.name] = [new_p_name]

            else:
                # Create new partition
                # Also,
                # -- p_0 --> A --> p_1 --> B
                #     `--------------------^
                logger.debug("Partition dep map: {}".format(partition_dep_map[X.name]))
                if X.name in name_2_pars:
                    logger.debug("Name_2_pars[X.name]: {}".format(name_2_pars[X.name]))

                new_p_name = name + str(idx)
                idx += 1
                logger.debug("Create new partition: {}".format(new_p_name))

                par_2_names[new_p_name] = [X.name]
                name_2_pars[X.name] = [new_p_name]

            p = name_2_pars[X.name][0]
            for t in X.tops:
                if t in name_2_pars and p not in name_2_pars[t]:
                    name_2_pars[t].append(p)
                else:
                    name_2_pars[t] = [p]

                if t not in par_2_names[p]:
                    par_2_names[p].append(t)

            if X.name == last_layer:
                stop_partitioning = True

            # logger.debug(name_2_pars)
            # logger.debug(par_2_names)

        logger.debug("----------------")

        # ALGO: keep only largest subgraph, prune all others
        # TODO Make more generic and support multiple criteria

        largest_xp, largest_xp_size = "", 0
        for xp in sorted(par_2_names.keys()):
            if len(par_2_names[xp]) > largest_xp_size:
                largest_xp = xp
                largest_xp_size = len(par_2_names[xp])

        for xp in list(par_2_names.keys()):
            if xp != largest_xp:
                del par_2_names[xp]

        for xname, xp_lst in list(name_2_pars.items()):
            if xp_lst[0] != largest_xp:
                del name_2_pars[xname]

        # Set target and group attributes
        for X in xgraph.get_layers():

            if X.name in name_2_pars:
                xlayers.append(
                    X._replace(target=target, subgraph=name_2_pars[X.name][0])
                )
            else:
                xlayers.append(copy.deepcopy(X))

        # TODO Sort xlayers in topological order
        xgraph = XGraphPartitioner.xgraph_factory.build_from_xlayer(
            net=xlayers,
            name=xgraph.get_name(),
            output_png="tvm_partitioned_graph.png"
            if logger.getEffectiveLevel() <= 10
            else None,
        )

        # Transpose optimizer
        optimizer = XGraphTransposesOptimizer(
            xgraph, target=target, opt_name="partitioning"
        )
        optimizer.optimize()

        return xgraph

    def get_subgraphs(self, xgraph: XGraph) -> List[XGraph]:
        """Return a list of subgraphs for the given xgraph in XGraph format."""

        # ALGO:
        # 1. loop through all the XLayers
        # 2. For every layer, if it's a target layer, either add it to the
        #   corresponding partition if a bottom is already added to
        #   this layer, else start a new partition
        # TODO set bottoms and tops to [] for partition input respectively
        #   output layers

        in_idx = 0

        visited = {}
        subgraphs = {}

        for X in xgraph.get_layers():

            if X.subgraph is not None:

                X_copy = copy.deepcopy(X)

                if X.subgraph not in subgraphs:
                    new_subgraph = xlayer.defaultXLayer()
                    new_subgraph = new_subgraph._replace(
                        name=X.subgraph,
                        type=["SubGraph"],
                        data=[],
                        shapes=TupleShape([]),
                        sizes=[],
                        internal=1,
                        attrs={
                            "target": X.target,
                            "__bottom_tensors": {},
                            "orig_bottom_tensors": {},
                            "__top_tensors": {},
                            "orig_top_tensors": {},
                        },
                    )
                    subgraphs[X.subgraph] = new_subgraph
                    visited[X.subgraph] = set()

                # First check if this layer is a subgraph input layer
                #   by looking at the visited subgraph layers
                for b in X.bottoms:

                    if b not in visited[X.subgraph]:

                        bX = xgraph.get(b)

                        x_in_name = "xinput" + str(in_idx)

                        def find_original_bottom_layers(rX):
                            if not bool(rX.internal):
                                return [rX.name]

                            bottom_layers = []
                            for r_bottom_name in rX.bottoms:
                                rbX = xgraph.get(r_bottom_name)
                                rec_bottom_layers = find_original_bottom_layers(rbX)
                                bottom_layers.extend(rec_bottom_layers)

                            return bottom_layers

                        orig_bottoms = find_original_bottom_layers(bX)

                        if "input_names" not in subgraphs[X.subgraph].attrs:
                            subgraphs[X.subgraph].attrs["input_names"] = [x_in_name]
                        else:
                            subgraphs[X.subgraph].attrs["input_names"].append(x_in_name)

                        # Keep track of input - bottom connections
                        sg_bottoms_ext = subgraphs[X.subgraph].attrs["__bottom_tensors"]
                        if X.name not in sg_bottoms_ext:
                            sg_bottoms_ext.update({x_in_name: [b]})
                        else:
                            new_bottoms_ext = sg_bottoms_ext[x_in_name] + [b]
                            sg_bottoms_ext.update({x_in_name: new_bottoms_ext})

                        # Keep track of input - original (model) bottom
                        #   connections, i.e. exclude internally added
                        #   operations here
                        sg_orig_bottoms_ext = subgraphs[X.subgraph].attrs[
                            "orig_bottom_tensors"
                        ]
                        if X.name not in sg_orig_bottoms_ext:
                            sg_orig_bottoms_ext.update({x_in_name: orig_bottoms})
                        else:
                            new_orig_bottoms_ext = (
                                sg_orig_bottoms_ext[x_in_name] + orig_bottoms
                            )
                            sg_orig_bottoms_ext.update(
                                {x_in_name: new_orig_bottoms_ext}
                            )

                        new_in_X = xlayer.defaultXLayer()
                        new_in_X = new_in_X._replace(
                            name=x_in_name,
                            type=["Input"],
                            shapes=bX.shapes[:],
                            sizes=bX.sizes[:],
                            # Keep track of the first original layer of the
                            #   operation in front of which we are adding an
                            #   input layer
                            layer=[X.layer[0]],
                            tops=[X.name],
                            bottoms=[],
                            internal=1,
                            attrs={},
                            targets=[],
                        )
                        in_idx += 1

                        X_copy.bottoms[:] = [
                            (bc if bc != b else new_in_X.name) for bc in X_copy.bottoms
                        ]

                        subgraphs[X.subgraph].subgraph_data = subgraphs[
                            X.subgraph
                        ].subgraph_data + [new_in_X]
                        # subgraphs[X.subgraph].shapes[:] = new_in_X.shapes[:]
                        # subgraphs[X.subgraph].sizes[:] = new_in_X.sizes[:]
                        subgraphs[X.subgraph].bottoms.append(b)

                        visited[X.subgraph].add(new_in_X.name)

                if X.tops == []:
                    sg_tops_ext = subgraphs[X.subgraph].attrs["__top_tensors"]
                    sg_orig_tops_ext = subgraphs[X.subgraph].attrs["orig_top_tensors"]
                    sg_tops_ext.update({X.name: []})
                    sg_orig_tops_ext.update({X.name: []})

                    if "output_names" not in subgraphs[X.subgraph].attrs:
                        subgraphs[X.subgraph].attrs["output_names"] = [X.name]
                    else:
                        subgraphs[X.subgraph].attrs["output_names"].append(X.name)

                for t in X.tops:
                    tX = xgraph.get(t)

                    if tX.subgraph != X.subgraph:

                        def find_original_top_layers(rX):
                            if not bool(rX.internal):
                                return [rX.name]

                            top_layers = []
                            for r_top_name in rX.tops:
                                rtX = xgraph.get(r_top_name)
                                rec_top_layers = find_original_top_layers(rtX)
                                top_layers.extend(rec_top_layers)

                            return top_layers

                        orig_tops = find_original_top_layers(tX)

                        if "output_names" not in subgraphs[X.subgraph].attrs:
                            subgraphs[X.subgraph].attrs["output_names"] = [X.name]
                        else:
                            subgraphs[X.subgraph].attrs["output_names"].append(X.name)

                        # Keep track of output - top connections
                        sg_tops_ext = subgraphs[X.subgraph].attrs["__top_tensors"]
                        if X.name not in sg_tops_ext:
                            sg_tops_ext.update({X.name: [t]})  # X.tops[:]
                        else:
                            new_tops_ext = sg_tops_ext[X.name] + [t]  # X.tops
                            sg_tops_ext.update({X.name: new_tops_ext})

                        # Keep track of output - original (model) top
                        #   connections, i.e. exclude internally added
                        #   operations here
                        sg_orig_tops_ext = subgraphs[X.subgraph].attrs[
                            "orig_top_tensors"
                        ]
                        if X.name not in sg_orig_tops_ext:
                            sg_orig_tops_ext.update({X.name: orig_tops})
                        else:
                            new_orig_tops_ext = sg_orig_tops_ext[X.name] + orig_tops
                            sg_orig_tops_ext.update({X.name: new_orig_tops_ext})

                        X_copy.tops.remove(t)
                        subgraphs[X.subgraph].tops.append(t)
                        subgraphs[X.subgraph].shapes.append(X.shapes[:])
                        subgraphs[X.subgraph].sizes.extend(X.sizes[:])

                # If no tops
                if X.tops == []:
                    subgraphs[X.subgraph].shapes.append(X.shapes[:])
                    subgraphs[X.subgraph].sizes.extend(X.sizes[:])

                subgraphs[X.subgraph].subgraph_data = subgraphs[
                    X.subgraph
                ].subgraph_data + [X_copy]
                visited[X.subgraph].add(X_copy.name)

        sg_list = []
        for sg, sgX in subgraphs.items():
            # (len(sgX.tops) == len(sgX.shapes))
            # if len(sgX.tops) != 1:
            #    raise ValueError("Subgraphs are only supported for one output"
            #        " but got: {}".format(sgX.tops))

            # TODO Sort xlayers in topological order
            # sub_xgraph = XGraphPartitioner.xgraph_factory.build_from_xlayer(
            #     net=sgX.data,
            #     name=sg
            # )

            sg_list.append(
                sgX._replace(
                    # shapes = sgX.shapes[0],
                    # sizes=[sum([s[0] for s in sgX.sizes])],
                    subgraph_data=sgX.subgraph_data
                )
            )

        return sg_list
