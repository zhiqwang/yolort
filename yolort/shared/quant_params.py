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
Module for handling quantization parameters
"""

from typing import Dict

import os
import json


class QuantParams:

    """
    This class is responsible for storing, loading and adjusting
    quantization parameters

    Attributes
        network (List[str]): keeps track of the order in which
            quantization parameters are added
        quant_params (Dict[str, dict]): mapping from quantization
            parameter name to quantization parameters
    """

    def __init__(self, quantizecfg=None):

        self.network = []
        self.quant_params = {}

        if quantizecfg is not None:
            self.load_file(quantizecfg)

    def empty(self) -> bool:
        if len(self.network) == 0:
            assert(not self.quant_params)
            return True
        return False

    def _get_internal_key(self, key: str) -> str:
        """
        Find the name internal name for the provided key
        """
        if key in self.quant_params:
            return key
        if key + '_QUANT_UTIL' in self.quant_params:
            return key + '_QUANT_UTIL'
        else:
            raise ValueError(f"Provided key: {key} does not exist as the key of a "
                             "quantization parameter specification.")

    def __getitem__(self, key: str) -> Dict:
        return self.quant_params[self._get_internal_key(key)]

    def __setitem__(self, key: str, value: Dict) -> None:
        if not isinstance(value, dict):
            raise ValueError("Provided quantization parameters should be of "
                             f"type: `dict`, but is: `{type(value)}`")

        self.quant_params[self._get_internal_key(key)] = value

    def __contains__(self, key: str) -> bool:
        # quant_util_key used for maxpool because otherwise it messes with the
        #   FPGA quant params
        quant_util_key = key + '_QUANT_UTIL'
        return key in self.quant_params or quant_util_key in self.quant_params

    def append(self, name: str, value: Dict) -> None:
        if not isinstance(value, dict):
            raise ValueError("Provided quantization parameters should be of "
                             f"type: `dict`, but is: `{type(value)}`")
        if name in self:
            raise ValueError("Could not append new quantization parameter with "
                             f"name: {name} because this key already exists")

        self.network.append(name)
        self.quant_params[name] = value

    def insert(
        self,
        name: str,
        value: Dict,
        after: str,
    ) -> None:
        if not isinstance(value, dict):
            raise ValueError("Provided quantization parameters should be of "
                             f"type: `dict`, but is: `{type(value)}`")
        if after not in self:
            raise ValueError("Could not insert a new quantization parameter "
                             f"after {after} because this key does not exist")
        if name in self:
            raise ValueError("Could not insert new quantization parameter with "
                             f"name: {name} because this key already exists")

        after_idx = self.network.index(self._get_internal_key(after))
        self.network.insert(after_idx + 1, name)
        self.quant_params[name] = value

    def insert_with_replace(
        self,
        name: str,
        value: Dict,
        after: str
    ) -> None:
        if not isinstance(value, dict):
            raise ValueError("Provided quantization parameters should be of "
                             f"type: `dict`, but is: `{type(value)}`")
        if after not in self:
            raise ValueError("Could not insert a new quantization parameter "
                             f"after {after} because this key does not exist")
        if name in self:
            index = self.network.index(self._get_internal_key(name))
            after_idx = self.network.index(self._get_internal_key(after))
            if index <= after_idx:
                raise ValueError("Can't do insertion with replacement. Replacement is only valid "
                                 "when index of element to be replaced is on the correct place "
                                 "in the sequence (behind the element after which the insertion "
                                 f"should happen), but found indexes: {index} and {after_idx + 1} "
                                 "for respectively index of element to replaced and index of "
                                 "element after which it should be inserted.")
            self.quant_params[self._get_internal_key(name)] = value
        else:
            self.insert(name, value, after)

    # IO

    def load_file(self, quantizecfg: str) -> None:
        """
        Load quantization parameters from specified quantization
        configuration file
        """
        if quantizecfg is None:
            return None

        if not os.path.isfile(quantizecfg):
            raise ValueError(f"Provided quantization file: {quantizecfg} for xfdnn "
                             "execution graph does not exist")

        with open(quantizecfg) as quant_json_file:
            quant_params_d = json.load(quant_json_file)

            self.load(quant_params_d)

    def load(self, quant_params_d: Dict) -> None:
        """
        Load quantization parameters from specified dictionary
        """

        network = []
        quant_params = {}

        quant_params_lst = quant_params_d["network"]

        # TODO: issue with multiple inputs/outputs !!!
        # quant_params['Input'] = quant_params_lst[0]
        # quant_params['Output'] = quant_params_lst[-1]

        for qp in quant_params_lst:
            # Keep track of the topological sequence
            if qp['name'] in ['network']:
                raise ValueError("Operation names: `network` are used internally and "
                                 "should not be used for naming operations at this moment")
            network.append(qp['name'])
            quant_params[qp['name']] = qp

        self.network = network
        self.quant_params = quant_params

    def save(self, quantfile: str) -> None:
        """
        Save the quantization parameters to json
        """
        if not quantfile.endswith('.json'):
            raise ValueError("Invalid quantization filename provided for storing "
                             "quantization parameters. The file should be of type "
                             f"`json` but was: {quantfile.split('.')[-1]}")

        d = {'network': []}
        for name in self.network:
            d['network'].append(self.quant_params[name])

        # Create directory if not exists
        dir_name = os.path.dirname(quantfile)
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            pass

        with open(quantfile, 'w') as f:
            json.dump(d, f, indent=4, sort_keys=True)
