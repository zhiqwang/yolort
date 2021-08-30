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
Module for XLayer definition
"""

from typing import Dict, List
import json
import numpy as np
import pyxir as px
import libpyxir as lpx

from collections import namedtuple

from yolort.shapes import TensorShape, TupleShape
from yolort.shared.vector import StrVector, IntVector, IntVector2D

from .xattr_dict import XAttrDict


# Convolution Data: WX + B
ConvData = namedtuple("ConvData", ['weights', 'biases'])

# Scale Data: gamma*X + beta
ScaleData = namedtuple("ScaleData", ['gamma', 'beta'])

# Batch Norm Data: gamma*(X-mu)/\sqrt(sigma_square+epsilon) + beta
BatchData = namedtuple("BatchData", ['mu', 'sigma_square', 'gamma', 'beta'])


class XLayer:

    @classmethod
    def _from_xlayer(cls, _xlayer):
        X = XLayer()
        X._set_xlayer(_xlayer)
        return X

    @classmethod
    def from_dict(cls, d: Dict):
        return XLayer(**d)

    def __init__(self, *args, **kwargs):
        self._xlayer = lpx.XLayer()
        # Initialize shapes ! important for length
        self.shapes = []

        # Default
        if 'target' not in kwargs:
            kwargs['target'] = 'cpu'

        self._set(*args, **kwargs)

    def _get_xlayer(self):
        return self._xlayer

    def _set_xlayer(self, _xlayer):
        assert isinstance(_xlayer, lpx.XLayer)
        self._xlayer = _xlayer

    def _set(self, *args, **kwargs):
        assert len(args) == 0
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _replace(self, *args, **kwargs):
        new_X = self.copy()
        new_X._set(*args, **kwargs)
        return new_X

    def copy(self):
        new_X = XLayer()
        new_X._set(
            name=self.name,
            type=[t for t in self.type],
            shapes=self.shapes.tolist(),
            sizes=[s for s in self.sizes],
            tops=[t for t in self.tops],
            bottoms=[b for b in self.bottoms],
            layer=[l for l in self.layer],
            data=[d for d in self.data] if isinstance(self.data, list)
            else self.data._replace(),  # TODO
            targets=[t for t in self.targets],
            target=self.target,
            subgraph=self.subgraph,
            subgraph_data=self.subgraph_data,
            internal=self.internal,
            attrs=self.attrs.copy()
        )
        return new_X

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        x_copy = self.copy()
        memo[id(x_copy)] = x_copy
        return x_copy

    def __eq__(self, other):
        return isinstance(other, XLayer) and \
            self.name == other.name and \
            self.type == other.type and \
            self.shapes == other.shapes and \
            self.sizes == other.sizes and \
            self.tops == other.tops and \
            self.bottoms == other.bottoms and \
            self.layer == other.layer and \
            self.data == other.data and \
            self.targets == other.targets and \
            self.target == other.target and \
            self.subgraph == other.subgraph and \
            self.subgraph_data == other.subgraph_data and \
            self.internal == other.internal and \
            self.attrs == other.attrs

    @property
    def name(self):
        return self._xlayer.name

    @name.setter
    def name(self, name_: str):
        # Stringify for TF quantizer
        self._xlayer.name = px.stringify(name_)

    @property
    def type(self):
        return StrVector(self._xlayer.xtype)

    @type.setter
    def type(self, value: List):
        self._xlayer.xtype = lpx.StrVector(value)

    @property
    def shapes(self):
        _shapes = self._xlayer.shapes
        _shapes_t = self._xlayer.shapes_t
        if _shapes_t == 'TensorShape' and len(_shapes) != 1:
            raise ValueError("TensorShape can only be one dimensional"
                             " but got: {}".format(len(_shapes)))

        if _shapes_t == 'TensorShape':
            return TensorShape(IntVector(_shapes[0]))
        elif _shapes_t == 'TupleShape':
            return TupleShape(IntVector2D(_shapes))
        else:
            raise ValueError("Unsupported shapes type: {}, should be"
                             " TensorShape or TupleShape"
                             .format(_shapes_t))

    @shapes.setter
    def shapes(self, shapes_):
        if isinstance(shapes_, TensorShape) or (isinstance(shapes_, list) and (
                len(shapes_) == 0 or isinstance(shapes_[0], int))):
            self._xlayer.shapes = lpx.IntVector2D([lpx.IntVector(shapes_)])
            self._xlayer.shapes_t = "TensorShape"
        else:
            assert all([isinstance(e, (list, TensorShape)) for e in shapes_])
            self._xlayer.shapes = lpx.IntVector2D([lpx.IntVector(s) for s in shapes_])
            self._xlayer.shapes_t = "TupleShape"

    @property
    def sizes(self):
        return IntVector(self._xlayer.sizes)

    @sizes.setter
    def sizes(self, sizes_: list):
        self._xlayer.sizes = lpx.IntVector(sizes_)

    @property
    def tops(self):
        return StrVector(self._xlayer.tops)

    @tops.setter
    def tops(self, tops_: list):
        self._xlayer.tops = lpx.StrVector([px.stringify(t) for t in tops_])

    @property
    def bottoms(self):
        return StrVector(self._xlayer.bottoms)

    @bottoms.setter
    def bottoms(self, bottoms_: list):
        self._xlayer.bottoms = lpx.StrVector([px.stringify(b) for b in bottoms_])

    @property
    def layer(self):
        return StrVector(self._xlayer.layer)

    @layer.setter
    def layer(self, layer_: list):
        self._xlayer.layer = lpx.StrVector(layer_)

    @property
    def data(self):
        # TODO: list??
        # TODO: remove op specific if else
        _data = [np.array(d, copy=False) for d in self._xlayer.data]
        if len(self.type) > 0 and self.type[0] in ['Convolution', 'Conv2DTranspose', 'Dense']:
            assert len(_data) == 2, (
                f"{self.type[0]} layer should have data attribute "
                f"of size 2 but got: {len(_data)}")
            return ConvData(*_data)
        elif 'Scale' in self.type:
            return ScaleData(*_data)
        elif 'BatchNorm' in self.type:
            return BatchData(*_data)
        else:
            return _data

    @data.setter
    def data(self, data_):
        # TODO: remove op specific if else
        if isinstance(data_, ConvData):
            data_ = [data_.weights, data_.biases]
        elif isinstance(data_, ScaleData):
            data_ = [data_.gamma, data_.beta]
        elif isinstance(data_, ScaleData):
            data_ = [data_.mu, data_.sigma_square, data_.gamma, data_.beta]

        assert all([isinstance(e, np.ndarray) for e in data_])
        buffer_data = lpx.XBufferVector([lpx.XBuffer(d) for d in data_])
        self._xlayer.data = buffer_data

    @property
    def targets(self):
        return StrVector(self._xlayer.targets)

    @targets.setter
    def targets(self, targets_: list):
        self._xlayer.targets = lpx.StrVector(targets_)

    @property
    def target(self):
        return self._xlayer.target if self._xlayer.target != '' else None

    @target.setter
    def target(self, target_: str):
        self._xlayer.target = target_ if target_ is not None else ''

    @property
    def subgraph(self):
        return self._xlayer.subgraph if self._xlayer.subgraph != '' else None

    @subgraph.setter
    def subgraph(self, subgraph_: str):
        self._xlayer.subgraph = subgraph_ if subgraph_ is not None else ''

    @property
    def subgraph_data(self):
        return [XLayer._from_xlayer(x) for x in self._xlayer.subgraph_data]
        # return make_any_vector(XLayer._from_xlayer, lpx.XLayerVector)(self._xlayer.subgraph_data)

    @subgraph_data.setter
    def subgraph_data(self, subgraph_data_: list):
        self._xlayer.subgraph_data = [(
            X._get_xlayer() if isinstance(X, XLayer) else XLayer(**X)._get_xlayer()
        ) for X in subgraph_data_]

    @property
    def internal(self):
        return self._xlayer.internal

    @internal.setter
    def internal(self, internal_):
        self._xlayer.internal = bool(internal_)

    @property
    def attrs(self):
        return XAttrDict(self._xlayer.attrs)

    @attrs.setter
    def attrs(self, d: dict):
        _xattr_dict = XAttrDict(lpx.XAttrMap())
        for key, value in d.items():
            _xattr_dict[key] = value

        self._xlayer.attrs = _xattr_dict._get_xattr_map()

    def __repr__(self):
        return str(json.dumps(self.to_dict(), indent=2))

    def __str__(self):
        return str(json.dumps(self.to_dict(), indent=2))

    def to_dict(self, data=False):
        return {
            'name': self.name,
            'type': [t for t in self.type],
            'shapes': self.shapes.tolist(),
            'sizes': [s for s in self.sizes],
            'tops': [t for t in self.tops],
            'bottoms': [b for b in self.bottoms],
            'layer': [l for l in self.layer],
            'data': [] if not data else self.data,
            'targets': [t for t in self.targets],
            'target': self.target,
            'subgraph': self.subgraph,
            'subgraph_data': [sd.to_dict(data) for sd in self.subgraph_data],
            'internal': self.internal,
            'attrs': self.attrs.to_dict()
        }


def defaultXLayer():
    return XLayer(
        layer=[],
        tops=[],
        bottoms=[],
        targets=[],
        target='cpu'
    )
