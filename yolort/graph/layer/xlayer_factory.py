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
Module for creating XLayer objects
"""

from typing import List, Any, Callable
import logging

from .xlayer import XLayer

logger = logging.getLogger("pyxir")


XLAYER_FACTORY = {}


def __general_xfactory_func(xop_name, register_func):
    """
    The register function is specific to each registered operation
    """

    def factory_func(op_name: str, in_xlayers: List[XLayer], **kwargs: Any) -> XLayer:
        """
        Generic XLayer factory function
        """

        attrs = kwargs

        bottoms = [iX.name for iX in in_xlayers]
        d = register_func(attrs, in_xlayers)
        shape = d['shape'][:]
        size = shape.get_size()  # abs(int(np.prod(shape)))

        X = XLayer()
        X = X._replace(
            name=op_name,
            type=[xop_name],
            shapes=shape,
            sizes=size,
            layer=[op_name],
            tops=[],
            bottoms=bottoms,
            attrs=attrs,
            targets=[])

        return X

    return factory_func


def xop_register(xop_name: str) -> Callable:
    """ Return decorator for registering factory function under
        provided name """

    def xop_register_decorator(register_func: Callable):
        if xop_name in XLAYER_FACTORY:
            raise ValueError("Can't register factory function for operation:"
                             " {} as the function has already been registered."
                             .format(xop_name))
        XLAYER_FACTORY[xop_name] = __general_xfactory_func(xop_name, register_func)

        return __general_xfactory_func(xop_name, register_func)

    return xop_register_decorator


def xop_register_factory(xop_name: str) -> Callable:
    """ Return decorator for registering flexible factory function under
        provided name """

    def xop_register_factory_decorator(factory_func: Callable):
        if xop_name in XLAYER_FACTORY:
            raise ValueError(
                "Can't register factory function for operation: "
                f" {xop_name} as the function has already been registered.")

        XLAYER_FACTORY[xop_name] = factory_func

        return factory_func

    return xop_register_factory_decorator


def get_xop_factory_func(xop_name: str, internal: bool = False) -> Callable:
    """
    Return a wrapper around the factory function for the specified
    XOp name. The wrapper adjusts the 'internal' attribute of the
    XLayer which specifies whether the layer is an original imported
    layer or an internally added one.
    """

    if xop_name not in XLAYER_FACTORY:
        raise ValueError(f"The provided operation: {xop_name} is not supported")

    def factory_wrapper(*args, **kwargs):
        X = XLAYER_FACTORY[xop_name](*args, **kwargs)
        X = X._replace(internal=int(internal))

        return X

    return factory_wrapper
