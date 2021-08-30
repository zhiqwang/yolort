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
Module for pattern matching on XGraph
"""

import logging

from ..layer.xlayer import defaultXLayer, XLayer
from ..passing import XGraphMutator, XGraphVisitor

logger = logging.getLogger("pyxir")


def is_mul_max_leaky_relu_pattern(
    inX: XLayer,
    mulX: XLayer,
    maxX: XLayer,
) -> bool:
    """
    Check whether provided layer is part of leaky relu pattern
    TODO: make more extensible
    """
    return ('Scale' in mulX.type and 'Maximum' in maxX.type
            and mulX.bottoms == [inX.name] and inX.name in maxX.bottoms
            and all(mulX.data.gamma == .1)
            and all(mulX.data.beta == 0.))


class XGraphPatternAnnotator(XGraphVisitor):
    """
    Annotate patterns in XGraph (for mul + max = leaky_relu)
    """

    def __init__(self):
        super().__init__()
        self.lr_layers = set([])

    def visit(self, X: XLayer) -> XLayer:
        if 'Scale' in X.type and len(X.tops) == 1:
            inX = self.xgraph.get(X.bottoms[0])
            topX = self.xgraph.get(X.tops[0])
            if is_mul_max_leaky_relu_pattern(inX, X, topX):
                self.lr_layers.add(X.name)
                if 'patterns' in X.attrs: 
                    X.attrs['patterns'].append('LeakyReLU')
                else:
                    X.attrs['patterns'] = ['LeakyReLU']
        elif 'Maximum' in X.type and any([b in self.lr_layers for b in X.bottoms]):
            if 'patterns' in X.attrs: 
                X.attrs['patterns'].append('LeakyReLU')
            else:
                X.attrs['patterns'] = ['LeakyReLU']


class XGraphPatternMutator(XGraphMutator):
    """
    Mutate patterns in XGraph (for mul + max = leaky_relu)
    """

    def __init__(self):
        super().__init__()
        self.lr_layers = {}
        self.lr_layers_bottoms = {}

    def visit(self, X: XLayer) -> XLayer:
        if 'Scale' in X.type and 'patterns' in X.attrs and 'LeakyReLU' in X.attrs['patterns']:
            self.lr_layers[X.name] = X.layer[:]
            self.lr_layers_bottoms[X.name] = X.bottoms
            return None
        elif 'Maximum' in X.type and 'patterns' in X.attrs and 'LeakyReLU' in X.attrs['patterns']:
            scale_name = X.bottoms[0] if X.bottoms[0] in self.lr_layers else X.bottoms[1]
            lR = defaultXLayer()
            lR = lR._replace(
                name=X.name,
                type=['LeakyReLU'],
                shapes=X.shapes[:],
                sizes=X.sizes[:],
                layer=self.lr_layers[scale_name] + X.layer[:],
                tops=[],
                bottoms=self.lr_layers_bottoms[scale_name],
                attrs={'alpha': .1}
            )
            return lR
        return super().visit(X)
