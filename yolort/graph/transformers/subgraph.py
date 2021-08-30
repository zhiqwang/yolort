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
Module for generic subgraph build function
"""

import logging

from yolort.shared import fancy_logging
from yolort.shapes import TensorShape

from .. import XGraph
from ..layer.xlayer import defaultXLayer
from ..algorithms.topological_sorting import sort_topologically
from ..xgraph_factory import XGraphFactory
from ..partitioning.xgraph_partitioner import XGraphPartitioner

from .layout_transformation_pass import XGraphLayoutTransformationPass


logger = logging.getLogger('pyxir')
fancy_logger = fancy_logging.getLogger("pyxir")


def find_indices(lst, condition):
    return [(i, elem) for i, elem in enumerate(lst) if condition(elem)]


def xgraph_build_func(
    xgraph: XGraph,
    target: str,
    xtype,
    layout='NCHW',
    **kwargs,
) -> XGraph:

    fancy_logger.banner(f"Subgraph build func, target: {target}, layout: {layout}")

    compiler_output = xgraph.get_compiler_output() if xgraph.is_compiled() else None
    compiler_output_keys = list(compiler_output.keys()) if compiler_output else []
    logger.debug(f"Compiler output keys: {compiler_output_keys}")
  
    if layout not in ['NCHW', 'NHWC']:
        raise ValueError(f"Supported layouts are [NCHW, NHWC] but got: {layout}")

    layout_transform_pass = XGraphLayoutTransformationPass(layout, target=target)
    xgraph = layout_transform_pass.execute(xgraph, subgraphs_only=False)

    xgraph_factory = XGraphFactory()
    xgraph_partitioner = XGraphPartitioner()

    subgraphs = {
        xp.name: xp for xp in xgraph_partitioner.get_subgraphs(xgraph)
    }

    # Retrieve CompilerOutput if available
    # compiler_output = xgraph.get_compiler_output() if xgraph.is_compiled() else None
    # compiler_output_keys = list(compiler_output.keys()) if compiler_output else []
    # logger.debug("Compiler output keys: {}".format(compiler_output_keys))
    # Keep track of the visited partitions/subgraphs and the layers
    #   inside the partition
    visited_xps = {}

    # Keep track of the subgraph output tensors and the corresponding
    #   new layers (TupleGetItem or Transpose)
    xp_out_tensors_2_layers = {}

    name_changes = {}
    net_map = {}
    net = []
    for X in xgraph.get_layers():

        if X.subgraph is not None and X.subgraph not in visited_xps:

            Xp = subgraphs[X.subgraph]

            if 'target' in Xp.attrs and Xp.attrs['target'] == target:

                visited_xps[Xp.name] = set([X.name])

                logger.debug(f"XSHAPES: {X.shapes}")

                bottoms = Xp.bottoms

                # Keep track of subgraph input and output names
                sub_xgraph = xgraph_factory.build_from_xlayer(Xp.subgraph_data)

                input_names = Xp.attrs['input_names'][:]
                output_names = Xp.attrs['output_names'][:]
                input_layers = [sub_xgraph.get(in_name) for in_name in input_names]
                output_layers = [sub_xgraph.get(out_name) for out_name in output_names]

                attrs = {
                    'input_names': input_names,
                    'output_names': output_names,
                    'input_layers': {
                        il.name: il.layer[:] for il in input_layers
                    },
                    'output_layers': {
                        ol.name: ol.layer[:] for ol in output_layers
                    }
                }
                for k, v in kwargs.items():
                    if k in attrs:
                        raise ValueError(f"Provided claimed subgraph layer key: {k}")
                    attrs[k] = v
                
                if Xp.name in compiler_output_keys:
                    attrs['rt_in_map'] = compiler_output.get_in_map(Xp.name)
                    for in_name in input_names:
                        for merged_layer in attrs['input_layers'][in_name]:
                            attrs['rt_in_map'][merged_layer] = attrs['rt_in_map'][in_name]
                    attrs['rt_out_map'] = compiler_output.get_out_map(Xp.name)
                    for out_name in output_names:
                        for merged_layer in attrs['output_layers'][out_name]:
                            attrs['rt_out_map'][merged_layer] = attrs['rt_out_map'][out_name]

                Xp.attrs.update(attrs)

                shapes = Xp.shapes[:]

                subgraph_X = Xp._replace(
                    # name = X.name,
                    type=[xtype],
                    shapes=shapes,
                    bottoms=bottoms,
                    # Fill tops later
                    tops=[],
                    subgraph_data=[]
                )
                net.append(subgraph_X.name)
                net_map[Xp.name] = subgraph_X

                # Subgraph layers have multiple outputs (Tuple) so we
                #   retrieve the different subgraph outputs
                #   (see output_names variable) using a TupleGetItem
                #   layer
                top_tensors = Xp.attrs['__top_tensors']

                for i, output_name in enumerate(output_names):
                    # Handle merged layers
                    out_tensor = Xp.attrs['output_layers'][output_name][-1]
                    tgi_name = out_tensor
                    # tgi_name = subgraph_X.name + '_tgi' + str(i)
                    
                    top_tensor = top_tensors[output_name]

                    shapes = subgraph_X.shapes[i][:]
                    X_tgi = defaultXLayer()
                    X_tgi = X_tgi._replace(
                        name=tgi_name,
                        type=['TupleGetItem'],
                        shapes=shapes,
                        sizes=shapes.get_size(),
                        layer=[tgi_name],
                        tops=top_tensor[:],
                        bottoms=[subgraph_X.name],
                        internal=1,
                        attrs={'index': i}
                    )
                    net.append(X_tgi.name)
                    # Keep track of TGI layer for both last merged layer and output name
                    net_map[tgi_name] = X_tgi
                    net_map[output_name] = X_tgi

                    subgraph_X.tops.append(tgi_name)

                    xp_out_tensors_2_layers[output_name] = tgi_name

            else:
                net.append(X.name)
                net_map[X.name] = X

        elif X.subgraph is not None and X.subgraph in visited_xps:
            # Remove layer
            visited_xps[X.subgraph].add(X.name)
        elif 'Transpose' in X.type:
            # Possibly merge transpose in TupleGetItem layer
            bX = net_map[X.bottoms[0]]
            new_tops = []
            for t in bX.tops:
                if t != X.name:
                    new_tops.append(t)
                elif len(X.tops) > 0:
                    new_tops.append(X.tops[0])
            if 'TupleGetItem' in bX.type:
                new_X = bX._replace(
                    tops=new_tops
                )
                new_X.attrs['transpose'] = True
                new_X.attrs['axes'] = X.attrs['axes']
                new_X.shapes[:] = TensorShape(X.shapes[:])
                net_map[new_X.name] = new_X
                name_changes[X.name] = bX.name
            else:
                net.append(X.name)
                net_map[X.name] = X
        else:
            net.append(X.name)
            net_map[X.name] = X

        # Reflect possibly merged layers
        new_bottoms = [b if b not in name_changes else name_changes[b] for b in X.bottoms]
        if new_bottoms != X.bottoms:
            new_X = X._replace(bottoms=new_bottoms)
            net_map[X.name] = new_X

    # Set tops and bottoms  & enforce topological sequence
    for xp in visited_xps.keys():
        Xp = subgraphs[xp]

        for b in Xp.bottoms:
            top_name = Xp.name
            bX = xgraph.get(b)
            bX.tops = [
                (bXt if bXt not in visited_xps[Xp.name] else top_name) for bXt in bX.tops
            ]

        for t in Xp.tops:
            tX = xgraph.get(t)
            tX.bottoms = [
                (tXb if tXb not in visited_xps[Xp.name] else xp_out_tensors_2_layers[tXb])
                for tXb in tX.bottoms
            ]

    # Topological sorting
    X_net = [net_map[e] for e in net]
    top_net = sort_topologically(X_net)

    sub_xgraph = xgraph_factory.build_from_xlayer(top_net)

    # Merge transposes if they are cancelling out
    # optimizer = XGraphTransposesOptimizer(sub_xgraph)
    # optimizer.optimize()

    return sub_xgraph
