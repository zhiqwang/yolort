# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.

from typing import Dict, Optional, Callable

from yolort import models
from yolort import data
from yolort import utils
from yolort import graph
from yolort.graph import ops

from .base import stringify

from .targets.cpu import (
    build_for_cpu_execution,
    cpu_xgraph_optimizer,
    cpu_xgraph_quantizer,
    cpu_xgraph_compiler,
)
from .graph.xop_registry import (
    xop_register_op_layout_transform,
    xop_register_op_transpose_transform,
)
from .runtime.base_runtime import BaseRuntime
from .runtime import rt_manager
from .target_registry import TargetRegistry, register_op_support_check

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass


device_r = TargetRegistry()


# RUNTIME APIs

def register_rt(
    rt_name: str,
    rt_graph: BaseRuntime,
    rt_ops: Dict,
) -> None:
    rt_manager.register_rt(rt_name, rt_graph, rt_ops)


def register_op(
    rt_name: str,
    op_type: str,
    setup_func: Callable,
) -> None:
    rt_manager.register_op(rt_name, op_type, setup_func)


def register_target(
    target: str,
    xgraph_optimizer: Callable,
    xgraph_quantizer: Callable,
    xgraph_compiler: Callable,
    xgraph_build_func: Callable,
    xgraph_op_support_annotator: Optional[Callable] = None,
    skip_if_exists: bool = False,
) -> None:
    device_r.register_target(
        target,
        xgraph_optimizer,
        xgraph_quantizer,
        xgraph_compiler,
        xgraph_build_func,
        xgraph_op_support_annotator=xgraph_op_support_annotator,
        skip_if_exists=skip_if_exists,
    )


register_target(
    'cpu',
    cpu_xgraph_optimizer,
    cpu_xgraph_quantizer,
    cpu_xgraph_compiler,
    build_for_cpu_execution,
)


@register_op_support_check('cpu', 'All')
def cpu_op_support_check(X, bXs, tXs):
    """
    Enable all operations
    """
    return True


def register_op_layout_transform(xop_name):
    return xop_register_op_layout_transform(xop_name)


def register_op_transpose_transform(xop_name):
    return xop_register_op_transpose_transform(xop_name)
