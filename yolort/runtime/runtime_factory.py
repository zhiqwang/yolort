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
Factory module for creating Runtimes
"""

from typing import List

import logging

from yolort.graph import XGraph
from yolort.graph.xgraph_factory import XGraphFactory

from .base_runtime import BaseRuntime

logger = logging.getLogger('pyxir')


class RuntimeFactory:

    """
    The RuntimeFactory Singleton is responsible for creating
    BaseRuntime objects
    """

    class __RuntimeFactory:

        def __init__(self):
            self.xgraph_factory = XGraphFactory()
            self._runtimes = {}

        def build_runtime(
            self,
            xgraph: XGraph,
            runtime: str = 'cpu-tf',
            target: str = 'cpu',
            last_layers: List[str] = None,
            batch_size: int = -1,
            placeholder: int = False,
            out_tensor_names: int = None,
            **kwargs,
        ) -> BaseRuntime:
            """
            Build an runtime graph based on the given target (e.g. tensorflow)
            """

            # logger.info("End building Runtime")
            # logger.info(f"Layers: {len(net)}")
            # logger.debug([X.name for X in net])

            # input_names = xgraph.get_input_names()
            output_names = set(xgraph.get_output_names())
            hidden_out_tensor_names = [
                otn for otn in out_tensor_names if otn not in output_names
            ] if out_tensor_names is not None else []

            return self._runtimes[runtime](
                xgraph.get_name(),
                xgraph,
                target,
                batch_size,
                placeholder,
                last_layers,
                hidden_out_tensor_names=hidden_out_tensor_names,
                **kwargs,
            )

        def register_exec_graph(self, rt_name: str, runtime: BaseRuntime):
            """
            Register a creator for a new Runtime subclass
            """
            if rt_name in self._runtimes:
                raise ValueError("This runtime is already registered")
            if not issubclass(runtime, BaseRuntime):
                raise ValueError("Provided runtime should be a subclass of Runtime")

            self._runtimes[rt_name] = runtime

    # storage for the instance reference
    __instance = None

    def __init__(self):
        """
        Create singleton instance
        """
        # Check whether we already have an instance
        if RuntimeFactory.__instance is None:
            # Create and remember instance
            RuntimeFactory.__instance = RuntimeFactory.__RuntimeFactory()

        # Store instance reference as the only member in the handle
        self.__dict__['_RuntimeFactory__instance'] = RuntimeFactory.__instance

    def __getattr__(self, attr):
        """
        Delegate access to implementation
        """
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """
        Delegate access to implementation
        """
        return setattr(self.__instance, attr, value)
