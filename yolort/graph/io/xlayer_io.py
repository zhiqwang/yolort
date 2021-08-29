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
Module for XLayer IO utility functions
"""
from typing import List, Dict, Tuple
import os
import json
import h5py
import numpy as np

import logging

from ..layer.xlayer import XLayer

logger = logging.getLogger("pyxir")


def _read_params(name: str, params_loc: str) -> np.ndarray:
    with h5py.File(params_loc) as f:
        dset = f[name]
        params = np.empty(dset.shape, dtype=np.float32)
        dset.read_direct(params)  # copy h5py dataset to numpy array
        return params


def get_empty_params(name, params_idx: int, params_loc: str) -> Tuple[Dict, int]:
    return dict(), params_idx


def get_dense_params(name, params_idx, params_loc):
    params = {}
    params[name + "_weights"] = _read_params(f"fc_{params_idx}", params_loc)
    params[name + '_biases'] = _read_params(f"fc_bias_{params_idx}",  params_loc)
    return params, params_idx + 1


def get_scaling_params(name, params_idx, params_loc):
    params = {}
    params[name + "_gamma"] = _read_params(f"fwbqb_{params_idx}", params_loc)
    params[name + '_beta'] = _read_params(f"fwbqb_bias_{params_idx}", params_loc)
    return params, params_idx + 1


def get_convolution_params(name, params_idx, params_loc):
    params = {}
    params[name + "_kernel"] = _read_params(f"fwbqb_{params_idx}", params_loc)
    params[name + '_biases'] = _read_params(f"fwbqb_bias_{params_idx}", params_loc)
    return params, params_idx + 1


PARAMS = {
    'Input': get_empty_params,
    'Output': get_empty_params,
    'Dense': get_dense_params,
    'Softmax': get_empty_params,

    # MATH
    'Scale': get_scaling_params,
    'Eltwise': get_empty_params,

    # CONVOLUTION
    'Convolution': get_convolution_params,
    'Pooling': get_empty_params,

    'Concat': get_empty_params,
    'Reshape': get_empty_params,
}


def _compiler_op_to_xlayer(op: Dict) -> XLayer:
    """
    Transform a compiled operation (as in the compiler json file) to
    a XLayer object
    """
    # logger.debug(op)

    X = XLayer()

    attrs = op['attrs']
    if 'insize_h' in op['xdnn_kv'] and 'insize_w' in op['xdnn_kv']:
        attrs['insize'] = [int(op['xdnn_kv']['insize_h']), int(op['xdnn_kv']['insize_w'])]
    if 'outsize_h' in op['xdnn_kv'] and 'outsize_w' in op['xdnn_kv']:
        attrs['outsize'] = [int(op['xdnn_kv']['outsize_h']), int(op['xdnn_kv']['outsize_w'])]

    # relu
    relu = 'relu' in op['xdnn_kv'] and op['xdnn_kv']['relu'] in ['1', 'true']

    # operation
    add = 1 if 'add' in op['xdnn_kv'] and op['xdnn_kv']['add'] in ['1', 'true'] else 0

    # pooling
    if 'XNOp' in op['xdnn_kv'] and op['xdnn_kv']['XNOp'] == 'XNMaxPool':
        pool_op = 'Max'
    elif 'XNOp' in op['xdnn_kv'] and op['xdnn_kv']['XNOp'] == 'XNAvgPool':
        pool_op = 'Avg'
    else:
        pool_op = None

    # kernel size
    kernel_sizes = [int(op['xdnn_kv']['kernel_h']), int(op['xdnn_kv']['kernel_w'])] if (
        'kernel_h' in op['xdnn_kv'] and 'kernel_w' in op['xdnn_kv']
    ) else None
    group = [1] if op['type'] == 'Convolution'else None

    # paddings ! in conv this is called padding_h ..., in pool paddings_h
    if 'padding_h' in op['xdnn_kv'] and 'padding_w' in op['xdnn_kv']:
        pad_h = int(op['xdnn_kv']['padding_h'])
        pad_w = int(op['xdnn_kv']['padding_w'])
        paddings = [[0, 0], [0, 0], [pad_h, pad_h], [pad_w, pad_w]]
    elif 'paddings_h' in op['xdnn_kv'] and 'paddings_w' in op['xdnn_kv']:
        pad_h = int(op['xdnn_kv']['paddings_h'])
        pad_w = int(op['xdnn_kv']['paddings_w'])
        paddings = [[0, 0], [0, 0], [pad_h, pad_h], [pad_w, pad_w]]
    else:
        paddings = None

    # strides
    strides = [int(op['xdnn_kv']['strides_h']), int(op['xdnn_kv']['strides_w'])] if (
        'strides_h' in op['xdnn_kv'] and 'strides_w' in op['xdnn_kv']
    ) else None

    # dilation
    dilation = [int(op['xdnn_kv']['dilation_h']), int(op['xdnn_kv']['dilation_w'])] if (
        'dilation_h' in op['xdnn_kv'] and 'dilation_w' in op['xdnn_kv']
    ) else None

    attrs.update({
        'padding': paddings,
        'strides': strides,
        'dilation': dilation,
        'groups': group[0] if group is not None else None,
        'kernel_size': kernel_sizes,
        'activation': 'ReLU' if relu else None,
        'pool_type': pool_op
    })

    X = X._replace(
        name=str(op['name']),
        type=[str(op['type'])],
        shapes=op['outputshapes'],
        bottoms=op['bottoms'],
        attrs=attrs
    )

    return X


def load_model_and_params_from_file(
    model_loc: str,
    params_loc: str,
) -> Tuple[List[XLayer], Dict[str, np.ndarray]]:

    if not os.path.isfile(model_loc):
        raise ValueError(f"Provided model file: {model_loc} for xfdnn "
                         "execution graph does not exist")

    with open(model_loc) as model_json_file:
        model_cfg = json.load(model_json_file)

    net = []
    params = {}
    params_idx = 0
    for _, op in enumerate(model_cfg["network"]):
        X = _compiler_op_to_xlayer(op)
        net.append(X)

        op_params, params_idx = PARAMS[X.type[0]](X.name, params_idx,
                                                  params_loc)
        params.update(op_params)

    return net, params


def load_model_as_fpga_xlayer(netcfg, quantizecfg, params_file) -> XLayer:
    """
    TODO
    """
    X = XLayer()  # XLayer(*[ None for i in XLayer._fields] )

    with open(netcfg) as f:
        netcfg_json = json.load(f)
    net = {layer['name']: layer for layer in netcfg_json['network']}

    input_names = [inpt['input_name'] for inpt in netcfg_json['inputs']]
    output_names = [outpt['previous_layers'][0]
                    for outpt in netcfg_json['outputs']]
    # fpga_ouput_names = [outpt['output_name'] for oupt in netcfg_json['outputs']]
    assert(len(input_names) > 0 and len(output_names) > 0)
    if len(input_names) > 1:
        raise NotImplementedError("Only one input for FPGA graph supported"
                                  f" at the moment, but found: {input_names}")
    if len(output_names) > 1:
        raise NotImplementedError("Only one output for FPGA graph supported"
                                  f" at the moment, but found: {output_names}")

    attrs = {
        'netcfg': netcfg,
        'quantizecfg': quantizecfg,
        'params_file': params_file,
        'inputs': input_names,
        'outputs': output_names,
        'layers': set(net.keys())
    }

    X = X._replace(
        name='fpga_layer',
        type=['FPGA'],
        shapes=net[output_names[0]]['outputshapes'],
        bottoms=input_names,
        tops=output_names,
        attrs=attrs
    )

    return X
