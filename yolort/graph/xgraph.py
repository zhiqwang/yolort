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
Module for XGraph data structure
"""

from typing import List, Dict
import copy
import logging
import warnings

import libpyxir as lpx

from yolort.shared.vector import StrVector
from yolort.shared.quantizer_output import QuantizerOutput

from . import XLayer
from .layer.xattr_dict import XAttrDict

logger = logging.getLogger('pyxir')


class XGraph:
    """
    The XGraph data structure for storing the model graph, accessing properties
    and doing graph level transformations

    Args:
        name (str): the XGraph name
    """

    @classmethod
    def _from_xgraph(cls, _xgraph: lpx.XGraph):
        xg = XGraph.__new__(cls)
        xg._xgraph = _xgraph
        xg.init()
        return xg

    def __init__(self, name='XGraph'):
        self._xgraph = lpx.XGraph(name)
        self.init()

    def init(self):
        # color map
        self.cm = (
            "#8dd3c7", "#fb8072", "#ffffb3", "#bebada",
            "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
        )

        # Quantization
        self.quantizer_output = None

        # Compilation
        self.compiler_output = None

        self._reset()

    def _reset(self):
        """
        Reset the dependent attributes (e.g. input and output layers) of this
        XGraph based on the current pydot graph
        """

        xlayers = self.get_layers()

        for X in xlayers:
            # Setup targets
            self.__setup_targets_for_X(X)

    def get_name(self) -> str:
        return self._xgraph.get_name()

    def set_name(self, name: str):
        self._xgraph.set_name(name)

    @property
    def meta_attrs(self):
        return XAttrDict(self._xgraph.meta_attrs)

    @meta_attrs.setter
    def meta_attrs(self, d: Dict):
        _xattr_dict = XAttrDict(lpx.XAttrMap())
        for key, value in d.items():
            _xattr_dict[key] = value

        self._xgraph.meta_attrs = _xattr_dict._get_xattr_map()

    ##########
    # LAYERS #
    ##########

    def get_input_names(self) -> List[str]:
        return StrVector(self._xgraph.get_input_names())

    def get_input_layers(self) -> List[XLayer]:
        return [self.get(il) for il in self.get_input_names()]

    def get_input_shapes(self) -> Dict[str, List[int]]:
        ils = self.get_input_layers()
        return {il.name: il.shapes[:] for il in ils}

    def get_output_names(self) -> List[str]:
        return StrVector(self._xgraph.get_output_names())

    def get_output_layers(self) -> List[XLayer]:
        return [self.get(ol) for ol in self.get_output_names()]

    def get_output_shapes(self) -> Dict[str, List[int]]:
        ols = self.get_output_layers()
        return {ol.name: ol.shapes[:] for ol in ols}

    def get(self, layer_name: str) -> XLayer:
        """
        Return an XLayer object by name
        """
        return XLayer._from_xlayer(self._xgraph.get(layer_name))

    def get_layer_names(self) -> List[str]:
        """
        Return all layer names in topological order
        """
        return StrVector(self._xgraph.get_layer_names())

    def get_layers(self) -> List[XLayer]:
        """
        Return all layers in topological order
        """
        return [self.get(ln) for ln in self.get_layer_names()]

    def get_bottom_layers(self, layer_name) -> List[XLayer]:
        """
        Get the bottom layers of the provided layer
        """
        return [self.get(b) for b in self.get(layer_name).bottoms]

    def get_top_layers(self, layer_name) -> List[XLayer]:
        """
        Get the top layers of the provided layer
        """
        return [self.get(t) for t in self.get(layer_name).tops]

    # CHECKS

    def exists_layer(self, layer_name: str) -> bool:
        return layer_name in self._xgraph

    # SET

    def add(self, X: XLayer) -> None:
        """ Add the provided XLayer object to the graph """
        # TODO: topological assumption here??

        if not isinstance(X, XLayer):
            raise ValueError(f"xlayer argument should be of type: XLayer but was: {type(X)}")

        self._xgraph.add(X._get_xlayer())

        # Setup targets
        X = self.get(X.name)
        self.__setup_targets_for_X(X)

        # Check bottom and top layers again
        bottom_Xs = self.get_bottom_layers(X.name)
        top_Xs = self.get_top_layers(X.name)
        for b_X in bottom_Xs:
            self.__setup_targets_for_X(b_X)
        for t_X in top_Xs:
            self.__setup_targets_for_X(t_X)

    def insert(self, X: XLayer) -> None:
        """
        Insert the provided XLayer object in the graph between
        two other layers
        """

        if len(X.bottoms) != 1 or len(X.tops) != 1:
            raise ValueError("Undefined behaviour: can't insert a node if there are "
                             "multiple bottom layers or multiple top layers")

        bX = self.get(X.bottoms[0])
        tX = self.get(X.tops[0])

        new_tops = [(bXt if bXt != tX.name else X.name) for bXt in bX.tops]
        new_bottoms = [(tXb if tXb != bX.name else X.name) for tXb in tX.bottoms]

        self.add(X)

        bX.tops = new_tops
        self.update(bX.name)

        tX.bottoms = new_bottoms
        self.update(tX.name)

    def update(self, X_name: str) -> None:
        """
        Update the given xlayer
        """
        layer_name = X_name  # X.name

        self._xgraph.update(X_name)
        X = self.get(layer_name)

        # Setup targets
        self.__setup_targets_for_X(X)

        # Check bottom and top layers again
        bottom_Xs = self.get_bottom_layers(X.name)
        top_Xs = self.get_top_layers(X.name)

        for b_X in bottom_Xs:
            self.__setup_targets_for_X(b_X)
        for t_X in top_Xs:
            self.__setup_targets_for_X(t_X)

    def remove(self, layer_name: str) -> None:
        """
        Remove the layer with given name and link the bottom and top
        layers.
        """

        # Retrieve bottom and top layers before removal
        bottoms = self.get(layer_name).bottoms[:]
        tops = self.get(layer_name).tops[:]

        # Link bottom and top layers
        bottom_Xs = [self.get(b) for b in bottoms]
        top_Xs = [self.get(t) for t in tops]

        for bX in bottom_Xs:
            new_tops = [([bXt] if bXt != layer_name else [tX.name for tX in top_Xs])
                        for bXt in bX.tops]

            # Flatten
            new_tops = [e for sl in new_tops for e in sl]
            bX.tops = new_tops

        for tX in top_Xs:
            new_bottoms = [([tXb] if tXb != layer_name else [bX.name for bX in bottom_Xs])
                           for tXb in tX.bottoms]

            # Flatten
            new_bottoms = [e for sl in new_bottoms for e in sl]
            tX.bottoms = new_bottoms

        # Bottom and top links have changed so clear X bottoms and tops
        # before removing
        X = self.get(layer_name)
        X.bottoms = []
        X.tops = []

        self._xgraph.remove(layer_name)

        for b in bottoms:
            self.update(b)
        for t in tops:
            self.update(t)

    #############
    # SUBGRAPHS #
    #############

    def get_subgraph_names(self) -> List[str]:
        """
        Return the names of all the subgraphs
        """
        return list(set(
            [X.subgraph for X in self.get_layers() if X.subgraph is not None]
        ))

    ################
    # QUANTIZATION #
    ################

    def is_quantized(self) -> bool:
        return (
            self.quantizer_output is not None
            or "is_quantized" in self.meta_attrs
            and self.meta_attrs["is_quantized"]
        )

    def set_quantizer_output(self, q_output: QuantizerOutput) -> None:
        self.quantizer_output = q_output

    def get_quantizer_output(self) -> QuantizerOutput:
        """
        Quantization information can be stored both in q_output attribute
        and in meta attributes
        TODO: Merge approaches
        """
        if not self.is_quantized():
            raise ValueError("No quantization output found. Quantize this XGraph object "
                             "before retrieving the quantization output")

        if (self.quantizer_output is not None and "is_quantized" in
                self.meta_attrs and self.meta_attrs["is_quantized"]):
            warnings.warn("Quantization info found both in XGraph meta "
                          "attributes and q_output attribute")

        if self.quantizer_output is not None:
            return self.quantizer_output

        # Retrieve quantization output from meta attributes
        q_output = QuantizerOutput(self.get_name())
        if "quant_keys" not in self.meta_attrs:
            raise ValueError("Expected `quant_keys` attribute in meta attributes")

        for q_key in self.meta_attrs["quant_keys"]:
            q_output.add(
                q_key=q_key,
                orig_pb=self.meta_attrs[q_key]['orig_pb'],
                q_eval=self.meta_attrs[q_key]['q_eval']
            )
            logger.debug(f"QOutput q_info: {self.meta_attrs[q_key]['q_eval']}")

        return q_output

    def save_quant_info_txt(self, filename) -> str:
        lines = []
        idx = 1
        for X in self.get_layers():
            if "vai_quant" in X.attrs:
                line = [str(idx), X.name]
                for quant_elem in X.attrs['vai_quant']:
                    line.extend([str(i) for i in X.attrs[quant_elem]])
                lines.append(line)
                idx += 1

        s = '\n'.join([' '.join(line) for line in lines])

        with open(filename, 'w') as f:
            f.write(s)

    ###############
    # COMPILATION #
    ###############

    def is_compiled(self):
        # type () -> boolean
        return self.compiler_output is not None

    def set_compiler_output(self, c_output):
        self.compiler_output = c_output

    def get_compiler_output(self):
        if not self.is_compiled():
            raise ValueError("No compilation output found. Compile this XGraph object "
                             "before retrieving the compilation output")
        return self.compiler_output

    ##################
    # HELPER METHODS #
    ##################

    def copy_meta_attrs(self, other: 'XGraph') -> None:
        self.meta_attrs = other.meta_attrs.to_dict()
        self.quantizer_output = other.quantizer_output
        self.compiler_output = other.compiler_output

    def copy(self):
        xg = XGraph(self.get_name())
        # xg.meta_attrs = self.meta_attrs.to_dict()
        # xg.quantizer_output = self.quantizer_output
        # xg.compiler_output = self.compiler_output
        xg.copy_meta_attrs(self)
        for X in self.get_layers():
            # Make sure top are empty to be able to add layer
            # TODO: slow? how many copies are made in total?
            X_copy = X.copy()
            X_copy.tops = []
            xg.add(X_copy)
        return xg

    def copy_from(self, xg: 'XGraph'):
        self._xgraph.copy(xg._xgraph)

    def visualize(self, outputfile) -> None:
        """ Visualize this xgraph using pydot """
        try:
            from . import pydot_tools
            import pydot
        except ImportError:
            raise ImportError("XGraph functionality depends on the 'pydot' package. "
                              "Please make sure that Pydot is installed before "
                              "trying to visualize XGraphs")

        pdg = pydot.Dot(self.get_name(), graph_type='digraph', rankdir='BT')

        cm_idx = 1
        target_to_cm = {}
        for X in self.get_layers():
            pydot_attrs = copy.copy(pydot_tools.LAYER_STYLE_DEFAULT)

            if 'Input' in X.type:
                pydot_attrs["shape"] = "oval"
                pydot_attrs["fillcolor"] = self.cm[0]

            if X.target != 'cpu':
                if X.target not in target_to_cm:
                    target_to_cm[X.target] = cm_idx
                    if cm_idx < (len(self.cm) - 1):
                        cm_idx += 1
                pydot_attrs["fillcolor"] = self.cm[target_to_cm[X.target]]

            # Add '-pdg' to fix issues of pydot with names with format
            #   '[...]:0' where ':0' gets removed
            node = pydot.Node(pydot.quote_if_necessary(X.name + '-pdg'), **pydot_attrs)

            pdg.add_node(node)

            for b in X.bottoms:
                src_nodes = pdg.get_node(pydot.quote_if_necessary(b + '-pdg'))

                if len(src_nodes) == 0:
                    raise ValueError(f"Pydot could not find layer with name: {b}")
                assert len(src_nodes) == 1

                src_node = src_nodes[0]

                edge_label = f"{b} -> {X.name}"
                # logger.debug("--Add bottom edge: {}".format(edge_label))
                pdg.add_edge(pydot.Edge(src_node, node, label=edge_label))

        pydot_tools.visualize(pdg, outputfile)

    def __setup_targets_for_X(self, X: XLayer) -> None:
        """Setup the supported targets for the provided XLayer"""
        # Check with registered targets, which device can execute
        #   this XLayer and add those targets to the XLayer device
        #   attribute
        pass
        # X.targets = []

        # bottom_Xs = self.get_bottom_layers(X.name)
        # top_Xs = self.get_top_layers(X.name)

        # for device in XGraph.target_registry.get_targets():
        #     if device.can_execute(X, bottom_Xs, top_Xs):
        #         X.targets.append(device.name)

    #########################
    # __*__ IMPLEMENTATIONS #
    #########################

    def __contains__(self, layer_name: str) -> int:
        """
        Reports whether a layer with given name exists in the XGraph
        """
        return layer_name in self._xgraph

    def __len__(self) -> int:
        return len(self._xgraph)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo: Dict) -> 'XGraph':
        """
        NOTE: We override the __deepcopy__ method because of internal C++
        XGraph data structure
        """
        xg_copy = self.copy()
        memo[id(xg_copy)] = xg_copy
        return xg_copy
