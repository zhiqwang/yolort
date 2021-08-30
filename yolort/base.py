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
PyXIR base API
"""

from typing import List, Dict, Optional, Callable

import os
import re
import copy
import json
import warnings

import numpy as np

import logging

from .shared import fancy_logging
from .shared.xbuffer import XBuffer
from .graph.xgraph import XGraph
from .io.api import save
from .runtime import runtime_factory
from .runtime.base_runtime import BaseRuntime
from .graph.partitioning.xgraph_partitioner import XGraphPartitioner
from .graph.transformers.layout_transformation_pass import XGraphLayoutTransformationPass

from .type import TypeCode
from .target_registry import TargetRegistry
from .opaque_func_registry import register_opaque_func
from .opaque_func import OpaqueFunc

logger = logging.getLogger("pyxir")
fancy_logger = fancy_logging.getLogger("pyxir")

xgraph_partitioner = XGraphPartitioner()
target_registry = TargetRegistry()


def stringify(name: str):
    # Tensorflow raises invalid scope name errors if name is invalid
    name = re.sub('[^A-Za-z0-9_.\\-/]', '-', name)
    try:
        # Some modules in Tensorflow subgraph contrib have issues with names
        #   that look like ints
        int(name)
        return str(name) + "_"
    except ValueError:
        return str(name)


################
# Transformers #
################


def transform_layout(xgraph: XGraph, layout: str):
    """
    Transform the layout of the XGraph model to the given layout
    """

    if layout not in ['NCHW', 'NHWC']:
        raise ValueError(f"Unsupported layout for model: {layout}. The supported "
                         "layouts are: `NCHW` and `NHWC`")

    layout_transform_pass = XGraphLayoutTransformationPass(layout)
    xgraph = layout_transform_pass.execute(xgraph, subgraphs_only=False)
    return xgraph


def partition(
    xgraph: XGraph,
    targets: List[str],
    last_layer: Optional[str] = None,
) -> XGraph:
    """
    Partition the model for the given targets
    """

    target_registry.check_targets(targets)
    target_registry.annotate_ops(xgraph)

    p_xgraph = xgraph_partitioner.partition(xgraph, targets, last_layer)
    return p_xgraph


@register_opaque_func('pyxir.partition', [
    TypeCode.XGraph,
    TypeCode.vStr,
    TypeCode.Str,
])
def partition_opaque_func(
    xgraph: XGraph,
    targets: List[str],
    last_layer: Optional[str] = None,
):
    """
    Expose the XGraph partition function an opaque function
    so it can be called from both Python and C++
    """

    if last_layer == "":
        last_layer = None

    p_xgraph = partition(xgraph, targets, last_layer)
    xgraph.copy_from(p_xgraph)


######################
# Graph optimization #
######################

def optimize(xgraph: XGraph, target: str, **kwargs) -> XGraph:
    """
    Optimize the XGraph for the given target
    """

    fancy_logger.banner(f"START GRAPH OPTIMIZATION FOR TARGET: {target}")
    opt_xgraph = target_registry.get_target_optimizer(target)(
        xgraph, target=target, **kwargs)
    if xgraph.is_quantized():
        opt_xgraph.set_quantizer_output(xgraph.get_quantizer_output())
    
    return opt_xgraph


##############
# SCHEDULING #
##############

def schedule(xgraph: XGraph, target: str, **kwargs) -> XGraph:
    """
    Schedule a xgraph for execution on the given target

    Returns:
        XGraph containing only executable operations
    """
    fancy_logger.banner(f"SCHEDULE `{target}` EXECUTION GRAPH")

    xgraph = target_registry.get_target_build_func(target)(
        xgraph.copy(), **kwargs)

    return xgraph


###############
# Compilation #
###############

def compile(xgraph: XGraph, target: str, **kwargs) -> XGraph:
    """
    Compile the XGraph for the given target
    """

    fancy_logger.banner(f"START GRAPH COMPILATION FOR TARGET: {target}")

    c_xgraph = target_registry.get_target_compiler(target)(xgraph, **kwargs)

    return c_xgraph


@register_opaque_func('pyxir.compile', [
    TypeCode.XGraph,
    TypeCode.Str,
    TypeCode.vStr,
    TypeCode.vStr,
    TypeCode.Str,
    TypeCode.Str,
    TypeCode.XGraph,
])
def compile_opaque_func(
    xgraph: XGraph,
    target: str,
    in_tensor_names: List[str],
    out_tensor_names: List[str],
    build_dir: str,
    work_dir: str,
    cb_scheduled_xgraph: XGraph,
) -> None:
    """
    Expose the compile function as an opaque function
    so it can be called from both Python and C++

    Args:
        xgraph (XGraph): the XGraph model
        target (str): the target backend for executing this xgraph
            the target should be registered with a corresponding
            build function.
        in_tensor_names (List)[str]: the names of the input
            tensors (in the order that they will be provided at runtime)
        out_tensor_names (List)[str]: the names of the output
            tensors (in the order that they will be retrieved at runtime)
        build_dir (str): the directory to be used for the final build files
        work_dir (str): the directory to be used for temporary work files
        cb_scheduled_xgraph (XGraph): return the scheduled XGraph
    """
    in_tensor_names = [stringify(itn) for itn in in_tensor_names]
    out_tensor_names = [stringify(otn) for otn in out_tensor_names]

    # if work_dir is None:
    if not work_dir:
        work_dir = os.path.abspath(os.path.join(os.getcwd(), f"{target}_work"))
    if not build_dir:
        build_dir = os.path.abspath(os.path.join(os.getcwd(), f"{target}_build"))

    opt_xgraph = optimize(xgraph, target)
    c_xgraph = compile(opt_xgraph, target, work_dir=work_dir, build_dir=build_dir)
    # full_graph_input_names = xgraph.get_input_names()

    # Create scheduled XGraph
    # TODO: work_dir <-> build_dir
    scheduled_xgraph = schedule(c_xgraph, target, work_dir=build_dir)

    # Save and add to meta file
    model_file = os.path.join(build_dir, 'px_model')
    save(scheduled_xgraph, model_file)

    meta_file = os.path.join(build_dir, 'meta.json')

    if (not os.path.isfile(meta_file)):
        raise ValueError(f"Could not find meta file at: {meta_file}")

    with open(meta_file, 'r') as json_file:
        meta_d = json.load(json_file)

    meta_d['px_model'] = 'px_model.json'
    meta_d['px_params'] = 'px_model.h5'

    with open(meta_file, 'w') as f:
        json.dump(meta_d, f, indent=4, sort_keys=True)

    # Set callback
    cb_scheduled_xgraph.copy_from(scheduled_xgraph)


################
# Quantization #
################

def quantize(
    xgraph: XGraph,
    target: str,
    inputs_func: Callable,
    **kwargs,
) -> XGraph:
    """
    Quantize the provided XGraph for the given target
    """

    return _quantize(xgraph, target, inputs_func, **kwargs)


def _quantize(
    xgraph: XGraph,
    target: str,
    inputs_func: Callable,
    **kwargs,
) -> XGraph:
    """
    Quantize the provided XGraph for the given target
    """

    fancy_logger.banner(f"START GRAPH QUANTIZATION FOR TARGET: {target}")

    q_xgraph = target_registry.get_target_quantizer(target)(
        xgraph, inputs_func, **kwargs
    )
    return q_xgraph


@register_opaque_func('pyxir.quantize', [
    TypeCode.XGraph,
    TypeCode.Str,
    TypeCode.vStr,
    TypeCode.vXBuffer,
])
def quantization_opaque_func(
    xgraph: XGraph,
    target: str,
    in_names: List[str],
    in_tensors: List[XBuffer],
) -> None:
    """
    Expose quantization as an opaque function so it can be called from
    both Python and C++

    Args:
        xgraph (XGraph): The XGraph model
        target (str): The target backend for executing this xgraph
            the target should be registered with a corresponding
            build function.
        in_names (List[str]): The names of the input names
            (in the same order as the input data)
        in_tensors (List[XBuffer]): The input tensors (in the same order as the `in_names`)
    """

    def inputs_func(iter):
        inputs = {in_name: it.to_numpy()
                  for in_name, it in zip(in_names, in_tensors)}
        return inputs

    q_xgraph = _quantize(xgraph, target, inputs_func)

    xgraph.copy_from(q_xgraph)


#########
# Build #
#########

def build(
    xgraph: XGraph,
    target: str,
    runtime: str = 'cpu-tf',
    last_layers: List[str] = None,
    work_dir: Optional[str] = None,
    build_dir: Optional[str] = None,
    **kwargs,
):
    """
    Build a runtime module from the provided XGraph model for the given
    target

    Args:
        xgraph (XGraph): The XGraph model
        target (str): The target backend for executing this xgraph
            the target should be registered with a corresponding
            build function.
        runtime (str): The target runtime to used for xgraph execution.
            Default: 'cpu-tf', which is the tensorflow runtime.
        last_layers (List[str]): The list of last layers for
            execution/quantization simulation.
        work_dir (str): The directory where to put the temporary work files
        build_dir (str): The directory where to put the build files
    """

    fancy_logger.banner(f"BUILD `{target}` RUNTIME GRAPH")

    if not work_dir:
        work_dir = os.path.join(os.getcwd(), f"{target}_work")
    if not build_dir:
        build_dir = os.path.join(os.getcwd(), f"{target}_build")

    c_xgraph = compile(xgraph, target, work_dir=work_dir, build_dir=build_dir)

    rt_xgraph = target_registry.get_target_build_func(target)(
        copy.deepcopy(c_xgraph),
        work_dir=work_dir,
        **kwargs,
    )

    return runtime_factory.build_runtime(
        xgraph=rt_xgraph,
        runtime=runtime,
        target=target,
        last_layers=last_layers
    )


@register_opaque_func('pyxir.build_rt', [
    TypeCode.XGraph,
    TypeCode.Str,
    TypeCode.Str,
    TypeCode.vStr,
    TypeCode.vStr,
    TypeCode.OpaqueFunc,
])
def build_rt_opaque_func(
    xgraph: XGraph,
    target: str,
    runtime: str,
    in_tensor_names: List[str],
    out_tensor_names: List[str],
    rt_callback: OpaqueFunc,
) -> None:
    """
    Expose the build runtime function as an opaque function
    so it can be called from both Python and C++

    Args:
    xgraph (XGraph): The XGraph model
    target (str): The target backend for executing this xgraph
        the target should be registered with a corresponding
        build function.
    runtime (str): The target runtime to used for xgraph execution
    in_tensor_names (List[str]): The names of the input tensor
        names (in the order that they will be provided at runtime)
    out_tensor_names (List[str]): The names of the output tensor
        names (in the order that they will be retrieved at runtime)
    rt_callback (OpaqueFunc): The callback function to be
        initialized with an opaque runtime function that takes
        a list of input buffers and output buffers and
        executes the model. This enables runtime modules
        in both C++ and Python.
    """
    in_tensor_names = [stringify(itn) for itn in in_tensor_names]
    out_tensor_names = [stringify(otn) for otn in out_tensor_names]

    rt_mod = build(
        xgraph=xgraph,
        target=target,
        runtime=runtime,
        last_layers=None
    )

    def rt_func(in_tensors, out_tensors):
        """
        The Python runtime function around the created runtime module
        """
        inputs = {
            it_name: it.to_numpy() for it_name, it in zip(in_tensor_names, in_tensors)
        }

        outs = rt_mod.run(inputs, out_tensor_names)

        # for out, out_tensor in zip(outs, out_tensors):
        #     out_tensor.copy_from(out)

        # TODO: hacky way to get in right layout. We possibly have transposes in the
        #   model to get the output in the right but we are retrieving just before
        #   those transposes
        for idx, ot_name in enumerate(out_tensor_names):
            tXs = xgraph.get_top_layers(ot_name)
            if len(tXs) == 1 and 'Transpose' in tXs[0].type:
                outs[idx] = np.transpose(outs[idx], axes=tuple(tXs[0].attrs['axes']))

        for out, out_tensor in zip(outs, out_tensors):
            out_tensor.copy_from(out)

    # Set the internal function in the rt_callback OpaqueFunc
    rt_callback.set_func(rt_func, [TypeCode.vXBuffer, TypeCode.vXBuffer])


@register_opaque_func('pyxir.build_online_quant_rt', [
    TypeCode.XGraph,
    TypeCode.Str,
    TypeCode.Str,
    TypeCode.vStr,
    TypeCode.vStr,
    TypeCode.Str,
    TypeCode.Str,
    TypeCode.OpaqueFunc,
    TypeCode.OpaqueFunc,
])
def build_online_quant_rt_opaque_func(
    xgraph: XGraph,
    target: str,
    runtime: str,
    in_tensor_names: List[str],
    out_tensor_names: List[str],
    build_dir: str,
    work_dir: str,
    quantization_callback: OpaqueFunc,
    rt_cpu_callback: OpaqueFunc,
) -> None:
    """
    Expose the online quantization build runtime function as an
    opaque function so it can be called from both Python and C++

    Args:
        xgraph (XGraph): The XGraph model
        target (str): The target backend for executing this xgraph
            the target should be registered with a corresponding
            build function.
        runtime (str): The target runtime to used for xgraph execution
        in_tensor_names (List[str]): The names of the input tensor
            names (in the order that they will be provided at runtime)
        out_tensor_names (List[str]): The names of the output tensor
            names (in the order that they will be retrieved at runtime)
        build_dir (str): The directory to be used for the final build files
        work_dir (str): The directory to be used for temporary work files
        quantization_callback (OpaqueFunc): The callback to be used for
            starting calibration based quantization using the collected input data
        rt_cpu_callback (OpaqueFunc): The callback function to be initialized
            with an opaque runtime function that takes a list of input buffers
            and output buffers and executes the model in CPU.
    """

    logger.info("Build On-the-fly quantization rt func")

    in_tensor_names = [stringify(itn) for itn in in_tensor_names]
    out_tensor_names = [stringify(otn) for otn in out_tensor_names]

    # Quantization callback function

    calibration_inputs = {}

    def inputs_func(iter):
        return calibration_inputs

    def quant_func():
        opt_xgraph = optimize(xgraph, target)
        q_xgraph = _quantize(opt_xgraph, target, inputs_func, work_dir=work_dir)
        # TODO
        xgraph.meta_attrs = q_xgraph.meta_attrs.to_dict()
        # xgraph.copy_from(q_xgraph)

    quantization_callback.set_func(quant_func, [])

    # CPU runtime function to be used during online quantization
    rt_mod = build(
        xgraph=xgraph,
        target="cpu",
        runtime="cpu-tf",
        last_layers=None,
        build_dir=build_dir,
        work_dir=work_dir
    )

    def rt_func(in_tensors, out_tensors):
        """
        The Python runtime function around the created runtime module
        """

        # Collect data for quantization
        for in_name, it in zip(in_tensor_names, in_tensors):
            if in_name in calibration_inputs:
                calibration_inputs[in_name] = np.concatenate(
                    [calibration_inputs[in_name], it.to_numpy(copy=True)], axis=0)
            else:
                calibration_inputs[in_name] = it.to_numpy(copy=True)

        # Run on inputs
        inputs = {
            it_name: it.to_numpy() for it_name, it in zip(in_tensor_names, in_tensors)
        }


        outs = rt_mod.run(inputs, out_tensor_names)
        # TODO: hacky way to get in right layout. We possibly have transposes in the
        #   model to get the output in the right but we are retrieving just before
        #   those transposes
        for idx, ot_name in enumerate(out_tensor_names):
            tXs = xgraph.get_top_layers(ot_name)
            # TODO previous: if len(tXs) == 1 and 'Transpose' in tXs[0].type:
            tp_layers = [tX for tX in tXs if 'Transpose' in tX.type]
            if len(tp_layers) > 0:
                outs[idx] = np.transpose(outs[idx], axes=tuple(tp_layers[0].attrs['axes']))

        # TODO: output order does not match
        for out, out_tensor in zip(outs, out_tensors):
            out_tensor.copy_from(out)
            

    # Set the internal function in the rt_cpu_callback OpaqueFunc
    rt_cpu_callback.set_func(rt_func, [TypeCode.vXBuffer, TypeCode.vXBuffer])


###########
# Runtime #
###########


def run(
    rt_mod: BaseRuntime,
    inputs: Dict[str, np.ndarray],
    outputs: List[str] = [],
    batch_size: int = 100,
    stop: Optional[str] = None,
    force_stepwise: bool = False,
) -> List[np.ndarray]:
    """
    Execute this computational graph on the given inputs and retrieve
    the requested outputs

    Args:
        inputs (Dict[str, numpy.ndarray]): The inputs for this executable
            computational graph
        outputs (List[str]): The output(s) to be returned
        batch_size (int): The batch size to be used to computing outputs
            for the given inputs
        stop (str): The operation at which to stop running
        force_stepwise (bool): whether to force a stepwise calculation of
            the computational graph on the provided inputs (used for
            debugging purposes)

    Returns:
        res (List[numpy.ndarray]): a list of outputs if requested, otherwise
            list containing the last output
    """
    if rt_mod.device not in ['cpu', 'qsim'] and batch_size != 1:
        warnings.warn("[WARNING] For the moment device type != 'cpu' "
                      "only supports batch size one! Changing batch size "
                      f"from {batch_size} to 1")
        batch_size = 1

    # Stringify because internal dependencies have issues with certain names
    inputs = {stringify(in_name): data for in_name, data in inputs.items()}

    input_keys = list(inputs.keys())

    inptsz = (inputs[input_keys[0]].shape[0]
              if isinstance(inputs[input_keys[0]], np.ndarray)
              else len(inputs[input_keys[0]]))
    # round up
    nb_batches = inptsz // batch_size + (inptsz % batch_size > 0)

    res = {}

    for batch_idx in range(nb_batches):

        logger.info("-----------------------")
        logger.info(f"Run batch: {batch_idx + 1}/{nb_batches}")

        input_batch = {}
        for inpt_key in inputs.keys():
            input_batch[inpt_key] = inputs[inpt_key][
                batch_idx*batch_size:(batch_idx+1)*batch_size]

        batch_outpts = rt_mod.run(input_batch, outputs, stop)

        # TODO: can we make the next two blocks more elegant?
        for idx, output_name in enumerate(outputs):
            if output_name in res:
                res[output_name] = np.concatenate([res[output_name], batch_outpts[idx]])
            else:
                res[output_name] = batch_outpts[idx]

        if len(outputs) == 0:
            if 'output' in res:
                res['output'] = [
                    np.concatenate([res['output'][i], batch_outpts[i]])
                    for i in range(len(batch_outpts))
                ]
            else:
                res['output'] = batch_outpts

    return [res[outpt] for outpt in outputs] if len(outputs) > 0 else res['output']

########
# Test #
########


@register_opaque_func('pyxir.test.copy_xbuffers', [
    TypeCode.vXBuffer,
    TypeCode.vXBuffer,
])
def copy_xbuffers(in_buffers, out_buffers):
    for idx, xb in enumerate(in_buffers):
        out_buffers[idx].copy_from(xb)
