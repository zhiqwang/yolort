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
Data structure for compiler output
"""

import os
import json


class CompilerOutputElem:

    def __init__(self, c_key, code_files=None, in_map=None, out_map=None):
        self.c_key = c_key
        self.code_files = code_files if code_files is not None else []
        self.in_map = in_map if in_map is not None else {}
        self.out_map = out_map if out_map is not None else {}

    def set_code_files(self, code_files):
        self.code_files = code_files

    def get_code_files(self):
        return self.code_files

    def set_in_map(self, in_map):
        self.in_map = in_map

    def get_in_map(self):
        return self.in_map

    def set_out_map(self, out_map):
        self.out_map = out_map

    def get_out_map(self):
        return self.out_map


class CompilerOutput:

    def __init__(self, name):
        self.name = name
        self.data = {}

    def get_name(self):
        return self.name

    def keys(self):
        return self.data.keys()

    def add(self, c_key, code_files, in_map, out_map):
        self.data[c_key] = CompilerOutputElem(c_key, code_files, in_map, out_map)

    def get_code_files(self, c_key):
        return self.data[c_key].get_code_files()

    def get_in_map(self, c_key):
        return self.data[c_key].get_in_map()

    def get_out_map(self, c_key):
        return self.data[c_key].get_out_map()

    def dump_in_out_name_maps(self, file_dir):
        """
        Dump the in and output name maps
        """
        for c_key in self.data.keys():
            d = {}
            d.update(self.get_in_map(c_key))
            d.update(self.get_out_map(c_key))

            file_name = os.path.join(file_dir, f"compiler_comp_{c_key}.json")
            with open(file_name, 'w') as f:
                json.dump(d, f)
