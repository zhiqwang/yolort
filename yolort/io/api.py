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
Module for Pyxir IO APIs
"""

import io
import os
import json
import zipfile

from yolort.graph import XGraph
from yolort.graph.io.xgraph_io import XGraphIO
from yolort.opaque_func_registry import register_opaque_func, OpaqueFuncRegistry
from yolort.type import TypeCode
from yolort.shared.container import BytesContainer
from .util import zip_dir


def visualize(xgraph: XGraph, pngfile: str = 'xgraph.png') -> None:

    xgraph.visualize(pngfile)


def save(xgraph: XGraph, filename: str) -> None:
    """
    Save this XGraph to disk. The network graph information is written to
    json and the network paraemeters are written to an h5 file

    Args:
        xgraph (XGraph): The XGraph to be saved
        filename (str): The name of the files storing the graph inormation and network
            parameters the graph information is stored in `filename`.json
            the network paraemeters are stored in `filename`.h5
    """
    XGraphIO.save(xgraph, filename)


@register_opaque_func('pyxir.io.save', [
    TypeCode.XGraph,
    TypeCode.Str,
])
def save_opaque_func(xg, filename):
    save(xg, filename)


def load(net_file: str, params_file: str) -> XGraph:
    """
    Load the graph network information and weights from the json network file
    respectively h5 parameters file

    Args:
        net_file (str): The path to the file containing the
            network graph information
        params_file (str): The path to the file containing the
            network weights
    """
    xgraph = XGraphIO.load(net_file, params_file)

    return xgraph


@register_opaque_func('pyxir.io.load', [
    TypeCode.Str,
    TypeCode.Str,
    TypeCode.XGraph,
])
def load_opaque_func(net_file, params_file, xg_callback):
    xg_callback.copy_from(load(net_file, params_file))


@register_opaque_func('pyxir.io.load_scheduled_xgraph_from_meta', [
    TypeCode.Str,
    TypeCode.XGraph,
])
def load_scheduled_xgraph_opaque_func(
    build_dir: str,
    cb_scheduled_xgraph: XGraph,
):
    """
    Expose the load scheduled xgraph function as an opaque function
    so it can be called in a language agnostic way

    Args:
        build_dir (str): The path to the build directory containing
            a meta.json file
        cb_scheduled_xgraph (XGraph): return the scheduled XGraph
    """
    meta_file = os.path.join(build_dir, 'meta.json')

    if (not os.path.isfile(meta_file)):
        raise ValueError(f"Could not find meta file at: {meta_file}")

    with open(meta_file) as json_file:
        meta_d = json.load(json_file)

    px_net_file = meta_d['px_model']
    px_params_file = meta_d['px_params']

    if not os.path.isabs(px_net_file):
        px_net_file = os.path.join(build_dir, px_net_file)

    if not os.path.isabs(px_params_file):
        px_params_file = os.path.join(build_dir, px_params_file)

    scheduled_xgraph = load(px_net_file, px_params_file)
    cb_scheduled_xgraph.copy_from(scheduled_xgraph)


@register_opaque_func('pyxir.io.to_string', [
    TypeCode.XGraph,
    TypeCode.BytesContainer,
    TypeCode.BytesContainer,
])
def write_to_string(
    xg,
    xgraph_json_str_callback,
    xgraph_params_str_callback,
):
    graph_str, data_str = XGraphIO.to_string(xg)
    xgraph_json_str_callback.set_bytes(graph_str)
    xgraph_params_str_callback.set_bytes(data_str)


def get_xgraph_str(xg: XGraph):
    of = OpaqueFuncRegistry.Get("pyxir.io.get_serialized_xgraph")
    s = BytesContainer(b"")
    of(xg, s)
    # import pdb; pdb.set_trace()
    return s.get_bytes()


def read_xgraph_str(xg_str: bytes):
    of = OpaqueFuncRegistry.Get("pyxir.io.deserialize_xgraph")
    xg = XGraph()
    s = BytesContainer(xg_str)
    # import pdb; pdb.set_trace()
    of(xg, s)
    return xg


@register_opaque_func('pyxir.io.from_string', [
    TypeCode.XGraph,
    TypeCode.Byte,
    TypeCode.Byte,
])
def read_from_string(
    xg,
    xgraph_json_str,
    xgraph_params_str,
):
    # graph_str, data_str = xgraph_str.split(";")
    xg_load = XGraphIO.from_string(xgraph_json_str, xgraph_params_str)
    xg.copy_from(xg_load)


@register_opaque_func('pyxir.io.serialize_dir', [
    TypeCode.Str,
    TypeCode.BytesContainer,
])
def serialize_dir(
    dir_path,
    serial_str_cb,
):
    if not os.path.isdir(dir_path):
        serial_str_cb.set_bytes(b"")
    else:
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as zip_f:
            zip_dir(dir_path, zip_f)

        s = bio.getvalue() # .hex()
        serial_str_cb.set_bytes(s)


@register_opaque_func('pyxir.io.deserialize_dir', [
    TypeCode.Str,
    TypeCode.Byte,
])
def deserialize_dir(
    dir_path,
    serial_str,
):
    if serial_str != b"" and not os.path.exists(dir_path):
        bio = io.BytesIO(serial_str) # .encode('latin1') bytes.fromhex(serial_str))
        with zipfile.ZipFile(bio, 'r') as zip_f:
            zip_f.extractall(dir_path)
        
        # If empty directory got zipped, recreate empty directory
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
