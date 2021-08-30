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
Base module for executing XGraphs
"""

from typing import Dict, List, Tuple, Callable, Optional

import abc
import logging
import numpy as np

from yolort.graph import XGraph, XLayer
from yolort.graph.layer import xlayer
from yolort.shared import fancy_logging

logger = logging.getLogger("pyxir")
fancy_logger = fancy_logging.getLogger("pyxir")


class BaseRuntime:

    __metaclass__ = abc.ABCMeta

    """
    Responsible for executing a graph

    Args:
        network (List[XLayer]): the network (list of operations) to be executed
        params (Dict[str, np.ndarray]): the network parameters (weights, biases)
        device (Optional[str]): the target device for graph execution.
            Only 'cpu' supported at the moment.
            Default: 'cpu'.
        batch_size (int): the batch size for the runtime graph, default is -1 (dynamic)

    Attributes:
        net (List[RtLayer]): a list of computational steps in the model network
        inputs (List[RtLayer]): a list of the network input layers
        outputs (List[RtLayer]): a list of the network output layers
        params (Dict): the parameters of the graph by name
        device (str): the target device for graph execution
        batch_size (int): the batch size for the runtime graph, default is -1 (dynamic)
    """

    def __init__(
        self,
        name,
        xgraph: XGraph,
        # params,
        device: str = 'cpu',
        batch_size: int = -1,
        placeholder: bool = False,
        last_layers: List[str] = None,
    ):
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.placeholder = placeholder
        self.xgraph = xgraph

        network, params = self._get_net_and_params(xgraph, last_layers)
        self.params = params
        self._init_net(network, self.params)

        self.name_to_nodes = {
            op.name: op for op in network
        }

    def _get_net_and_params(self, xgraph: XGraph, last_layers: List[str]):
        """
        Return the XGraph submodel as a list of XLayers and the
        parameters provided the given last layers of the runtime model
        """
        # TODO Remove hardcoding parameter retrieval 

        net = []
        params = {}
        last_layer_cnt = 1
        last_layer_tops = set([])

        for X in xgraph.get_layers():

            if X.name in last_layer_tops:
                last_layer_tops = last_layer_tops.union(tuple(X.tops))
                continue

            if 'Convolution' in X.type or 'Conv2DTranspose' in X.type:
                if not isinstance(X.data, xlayer.ConvData):
                    raise ValueError(f"Invalid convolution data type: {type(X.data)}, "
                                     "should be `xlayer.ConvData`")
                # OIHW
                params[X.name + '_kernel'] = X.data.weights
                params[X.name + '_biases'] = X.data.biases
            elif 'Dense' in X.type:
                if not isinstance(X.data, xlayer.ConvData):
                    raise ValueError(f"Invalid inner product data type: {type(X.data)}, "
                                     "should be `xlayer.ConvData`")
                # OIHW
                params[X.name + '_weights'] = X.data.weights
                params[X.name + '_biases'] = X.data.biases
            elif 'BatchNorm' in X.type:
                if not isinstance(X.data, xlayer.BatchData):
                    raise ValueError(f"Invalid batchnorm data type: {type(X.data)}, "
                                     "should be `xlayer.BatchData`")
                # channels
                params[X.name + '_mu'] = X.data.mu
                params[X.name + '_variance'] = X.data.sigma_square
                params[X.name + '_gamma'] = X.data.gamma
                params[X.name + '_beta'] = X.data.beta
            elif 'Scale' in X.type:
                if not isinstance(X.data, xlayer.ScaleData):
                    raise ValueError(f"Invalid scale data type: {type(X.data)}, "
                                     "should be `xlayer.ScaleData`")
                # channels
                params[X.name + '_gamma'] = X.data.gamma
                params[X.name + '_beta'] = X.data.beta
            elif 'BiasAdd' in X.type:
                assert X.data is not None
                params[X.name + '_bias'] = X.data[0]
            elif 'Eltwise' in X.type:
                if X.data != []:
                    params[X.name + '_beta'] = X.data[0]

            net.append(X)

            if last_layers is not None and X.name in last_layers:
                if last_layer_cnt == len(last_layers):
                    break
                else:
                    last_layer_cnt += 1
                    last_layer_tops = last_layer_tops.union(tuple(X.tops))

        return net, params

    @abc.abstractmethod
    def _xfdnn_op_to_exec_op(self, op_type: str) -> Callable:
        """
        Abstract method

        Returns a function of type:
            (XLayer, Dict[str, List[int]], Dict[str, np.ndarray], Dict[str, Dict])
                -> List[rt_layer.RtLayer]
            that takes in a parameters layer object, inputs shapes dict, params
            dict and quantization parameters dict and outputs and returns a list
            of executable RtLayer objects

        TODO: make the returned function more formal
        """
        raise NotImplementedError

    def _init_net(
        self,
        network: List[XLayer],
        params: Dict[str, np.ndarray],
    ) -> None:

        fancy_logger.banner("INIT NET")

        self.net = []
        self.inputs = []
        self.outputs = []

        input_shapes = {}

        for op_idx, op in enumerate(network):

            logger.info("-----------------------")
            logger.info(f"Op idx: {op_idx}, op_name: {op.name}, "
                        f"op_type: {op.type} op shapes: {op.shapes}")

            xfdnn_layers = self._xfdnn_op_to_exec_op(op.type[0])(
                op, input_shapes, params, batch_size=self.batch_size,
                placeholder=self.placeholder)

            logger.debug(f"Add input shape: {op.name}: {xfdnn_layers[-1].shape}")
            input_shapes[op.name] = xfdnn_layers[-1].shape

            self.net = self.net + xfdnn_layers
            # self.params.update(params)

            if op.type[0] in ['Input', 'StrInput'] and op.name not in params:
                self.inputs.append(xfdnn_layers[0])
            if 'Output' in op.type:
                self.outputs.append(xfdnn_layers[0])

    def run_stepwise(
        self,
        inputs: Dict,
        stop: Optional[str] = None,
    ) -> Tuple[int, str, Dict, np.ndarray, np.ndarray]:
        """
        TODO Remove stepwise execution (for tensorflow)?
        """
        fancy_logger.banner("RUN NET STEPWISE")

        inputs.update(self.params)

        for layer_idx, layer in enumerate(self.net):

            # logger.info("-----------------------")
            # logger.info("Run layer idx: {}, op_name: {}"
            #   .format(layer_idx, layer.name))

            inpts = [inputs[name] for name in layer.inputs]

            outpt = layer.forward_exec(inpts)

            # TODO: can we make this more elegant?
            if layer.type in ['Convolution']:
                quant_outpt = layer.get_output_for_quantization(inpts)
            else:
                quant_outpt = outpt

            # TODO: remove unnecessary data as we keep track of
            #   the outputs of all layers
            inputs[layer.name] = outpt

            yield (
                layer_idx,
                layer,
                inpts,
                outpt,
                quant_outpt
            )

            if stop is not None and layer.name == stop:
                break

    def run(
        self,
        inputs: Dict[str, np.ndarray],
        outputs: List[str] = [],
        stop: Optional[str] = None,
        force_stepwise: bool = True,
    ) -> List[np.ndarray]:
        """
        Execute this computational graph on the given inputs.

        Args:
            inputs (Dict[str, np.ndarray]): the inputs for this executable
                computational graph
            outputs (List[str]): the output(s) to be returned
            stop (str): the operation at which to stop running
            force_stepwise (bool): whether to force a stepwise calculation of
                the computational graph on the provided inputs
                ! Unused because this runtime always used stepwise calculations
                Default: True.

        Returns:
            res (List[np.ndarray]): a list of outputs if requested, otherwise
                list containing the last output
        """
        fancy_logger.banner("RUN NET")

        inputs.update(self.params)
        res = {}
        for layer_idx, layer in enumerate(self.net):

            logger.info("-----------------------")
            logger.info(f"Run layer idx: {layer_idx}, op_name: {layer.name}")
            logger.info(f"Inputs: {layer.inputs}")
            inpts = [inputs[name] for name in layer.inputs]

            outpt = layer.forward_exec(inpts)

            # TODO: remove unnecessary data as we keep track of
            #   the outputs of all layers
            inputs[layer.name] = outpt

            if layer.name in outputs:
                res[layer.name] = outpt

            if stop is not None and layer.name == stop:
                break

        if len(outputs) == 0:
            res['output'] = outpt

        return [res[outpt] for outpt in outputs] if len(outputs) > 0 else [res['output']]

    @abc.abstractmethod
    def optimize(
        self,
        inputs: Dict[str, np.ndarray],
        debug: bool=False,
    ) -> Tuple[Dict, List[np.ndarray]]:

        raise NotImplementedError

    # SETTERS/GETTERS

    def get_input_names(self) -> List[str]:
        return [layer.name for layer in self.inputs]

    def get_input_shapes(self) -> List[List[int]]:
        return [layer.shape for layer in self.inputs]

    def get_ouput_names(self) -> List[str]:
        return [layer.name for layer in self.outputs]

    def get_output_shapes(self) -> List[List[int]]:
        return [layer.shape for layer in self.outputs]

    def get_node(self, node_name: str) -> Dict:
        """
        Returns the node with provided name if it exists else throws a KeyError
        TODO:
        Return
        """
        return self.name_to_nodes[node_name]

    def get_input_node_names(self, node_name: str) -> List:
        """
        Returns a list of dictionaries containing the inputs' node information
        """
        node = self.get_node(node_name)
        return node.bottoms

    def get_output_node_names(self, node_name: str) -> List:
        """
        Returns a list of dictionaries containing the outputs' node information
        """
        node = self.get_node(node_name)
        return node.tops
