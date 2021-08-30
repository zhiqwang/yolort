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
Module for Decent quantizer simulation runtime
"""

import os
import copy
import numpy as np

from typing import List, Dict, Union

from yolort.graph import XLayer, XGraph
from yolort.graph.xgraph_factory import XGraphFactory
from yolort.graph.partitioning.xgraph_partitioner import XGraphPartitioner
from yolort.target_registry import TargetRegistry

from ..base_runtime import BaseRuntime


class RuntimeDecentQSim(BaseRuntime):

    """Runtime for Decent quantizer simulation"""
    
    xgraph_partitioner = XGraphPartitioner()
    xgraph_factory = XGraphFactory()
    target_registry = TargetRegistry()

    def __init__(
        self,
        name,
        xgraph: XGraph,
        device: str = 'cpu',
        batch_size: int = -1,
        placeholder: bool = False,
        last_layers: List[str] = None,
        **kwargs,
    ):
        super().__init__(
            name,
            xgraph,
            device,
            batch_size,
            placeholder,
            last_layers
        )

        meta_attrs = self.xgraph.meta_attrs

        if 'quant_keys' not in meta_attrs:
            raise ValueError("Trying to simulate unquantized model. Make sure to first "
                             "quantize the model.")

        qkey = meta_attrs['quant_keys'][0]
        self.q_eval = meta_attrs[qkey]['q_eval']
        self.gpu = 0

        Xps = RuntimeDecentQSim.xgraph_partitioner.get_subgraphs(xgraph)
        assert len(Xps) == 1, "Decent quantizer simulation only supports one partition currently"
        self.Xp = Xps[0]
        target = self.Xp.attrs['target']
        opt_xgraph = RuntimeDecentQSim.target_registry.get_target_optimizer(target)(
            self.xgraph,
            target=target
        )
        self.rt_xgraph = RuntimeDecentQSim.target_registry.get_target_build_func(target)(
            copy.deepcopy(opt_xgraph),
            data_layout='NHWC' # NOTE XGraph's should be built in NHWC data layout, this is
                               # important for DPUCADX8G where DPU execution happens in NCHW
                               # but quantization simulation in NHWC
        )

    def _init_net(self, network: List[XLayer], params: Dict[str, np.ndarray]):
        # Do nothing
        pass

    def run_input(self, X: XLayer, inputs: Dict[str, Union[np.ndarray, List[np.ndarray]]])\
            -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        return None

    def run_transpose(self, X: XLayer, inputs: Dict[str, Union[np.ndarray, List[np.ndarray]]])\
            -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        assert len(X.bottoms) == 1
        return np.transpose(inputs[X.bottoms[0]], axes=tuple(X.attrs['axes'][:]))

    def run_dpu(
        self,
        X: XLayer,
        inputs: Dict[str, Union[np.ndarray, List[np.ndarray]]],
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        input_graph_def = tf.Graph().as_graph_def()
        input_graph_def.ParseFromString(tf.io.gfile.GFile(self.q_eval, "rb").read())
        tf.import_graph_def(input_graph_def, name='')

        input_names = X.attrs["input_names"]
        input_map = {X.attrs["__bottom_tensors"][in_name][0]: in_name for in_name in input_names}
        in_tensors = {
            k: tf.compat.v1.get_default_graph().get_tensor_by_name(input_map[k] + ":0")
            for k in X.bottoms
        }

        feed_dict = {in_tensors[k]: inputs[k] for k in X.bottoms}

        out_names = X.attrs["output_names"]
        out_tensor_names = [X.attrs["output_layers"][o][-1] for o in out_names]
        
        out_tensors = [
            tf.compat.v1.get_default_graph().get_tensor_by_name(o + "/aquant" + ":0")
            for o in out_names
        ]

        with tf.compat.v1.Session() as sess:
            out = sess.run(out_tensors, feed_dict=feed_dict)
            return out if isinstance(out, list) else [out]

    def run_tuple_get_item(
        self,
        X: XLayer,
        inputs: Dict[str, Union[np.ndarray, List[np.ndarray]]],
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:

        assert len(X.bottoms) == 1
        index = X.attrs['index']
        data = inputs[X.bottoms[0]][index]
        if 'transpose' in X.attrs and X.attrs['transpose'] is True:
            return np.transpose(data, axes=tuple(X.attrs['axes'][:]))
        return data

    def run_tuple(
        self,
        X: XLayer,
        inputs: Dict[str, Union[np.ndarray, List[np.ndarray]]],
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:

        return [inputs[b] for b in X.bottoms]

    def run(
        self,
        inputs: Dict[str, np.ndarray],
        outputs: List[str] = [],
        stop: str = None,
        force_stepwise: bool = False,
        debug: bool = False,
    ) -> List[np.ndarray]:
        """
        Override run method
        """
        for X in self.rt_xgraph.get_layers():
            if 'Input' in X.type:
                outs = self.run_input(X, inputs)
            elif 'Transpose' in X.type:
                outs = self.run_transpose(X, inputs)
            elif 'DPU' in X.type:
                outs = self.run_dpu(X, inputs)
            elif 'TupleGetItem' in X.type:
                outs = self.run_tuple_get_item(X, inputs)
            elif 'Tuple' in X.type:
                outs = self.run_tuple(X, inputs)
            else:
                raise NotImplementedError(f"Unsupported operation in decentq simulation: {X.type[0]}")
            if outs is not None:
                inputs[X.name] = outs
        return [inputs[o] for o in outputs]


        
