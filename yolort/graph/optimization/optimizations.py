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
Module containing XLayer optimization functions for graph optimization passes
"""

from typing import List, Any, Optional

import logging
import numpy as np

from .. import XGraph
from ..layer import xlayer, XLayer
from ..xop_registry import XOpRegistry

logger = logging.getLogger("pyxir")


def remove(
    xgraph: XGraph,
    bottom_Xs: List[XLayer],
    X: XLayer,
    top_Xs: List[XLayer],
    **kwarg: Any,
) -> bool:
    """
    Remove the provided XLayer from the graph
    """
    if not (len(bottom_Xs) == 1 or len(top_Xs) == 1):
        raise ValueError("Undefined behaviour: can't remove a XLayer if there "
                         "are multiple bottom layers and multiple top layers")

    # Remove from graph
    xgraph.remove(X.name)

    return True


def merge_transposes(
    xgraph: XGraph,
    bottom_Xs: List[XLayer],
    X: XLayer,
    top_Xs: List[XLayer],
    **kwargs: Any,
) -> bool:
    """
    Try to merge transpose layers in the XGraph. This is used for
    supporting networks with operations in multiple layouts

    TRANSFORMATIONS
        Transform X --> T  in  T --> X if X is a valid layer and `--> T
            the transposes are the same
        Transform X --> T in T --> X if X is a valid layer
    """

    # Used for getting information on operations
    xop_registry = XOpRegistry()

    # tX_all_transposes = all([tX.type[0] == 'Transpose' for tX in top_Xs])
    tX_all_eq_axes = all([tX.attrs['axes'] == top_Xs[0].attrs['axes'] for tX in top_Xs])

    changes = False
    logger.debug(f"-- -- Merge transposes: {[bX.name for bX in bottom_Xs]}, "
                 f"{X.name}, {[tX.name for tX in top_Xs]}")

    # assert len(top_Xs) <= 1 or len(bottom_Xs) <= 1, (
    #     f" top_Xs: {top_Xs}, bottom_Xs: {bottom_Xs}")

    if len(top_Xs) > 0 and tX_all_eq_axes and X.type[0] in ['Transpose']:
        # Check if we have two transposes that cancel eachother out
        tX = top_Xs[0]
        axes = X.attrs['axes']
        tX_axes = tX.attrs['axes']

        if [axes[i] for i in tX_axes] == [0, 1, 2, 3]:
            logger.debug(f"-- -- Merge transposes: bX: {[bX.name for bX in bottom_Xs]}, "
                         f"X: {X.name}, tX: {[tX.name for tX in top_Xs]}")
            changes = True
            xgraph.remove(X.name)

            for tX in top_Xs:
                xgraph.remove(tX.name)

    elif (len(top_Xs) > 0 and tX_all_eq_axes and
          X.type[0] in xop_registry.get_xops_with_transpose_transform()):
        # ['ReLU', 'BiasAdd', 'Concat', 'Eltwise', 'BatchNorm', 'Scale', 'Pad']:
        changes = True
        logger.debug(f"-- -- Move transpose: bX: {[bX.name for bX in bottom_Xs]}, "
                     f"X: {X.name}, tX: {[tX.name for tX in top_Xs]}")

        tX = top_Xs[0]
        tX_name = tX.name
        ttXs = [xgraph.get(tt_name) for tX in top_Xs
                for tt_name in tX.tops]
        axes = tX.attrs['axes'][:]

        top_names = [tX.name for tX in top_Xs]

        for tX in top_Xs:
            xgraph.remove(tX.name)

        for i, bX in enumerate(bottom_Xs):

            if len(bottom_Xs) > 1:
                t_name = "{}_split_{}".format(
                    i, "_".join(top_names))
            elif len(top_Xs) > 1:
                t_name = "merge_{}".format("_".join(top_names))
            else:
                t_name = tX_name

            t_shape = [bX.shapes[i] for i in axes]
            logger.debug("-- -- t_shape: {}".format(t_shape))
            attrs = {'axes': axes}

            T = xlayer.defaultXLayer()
            T = T._replace(
                name=t_name,
                type=['Transpose'],
                shapes=t_shape,
                sizes=bX.sizes[:],
                layer=[t_name],
                tops=[X.name],
                bottoms=[bX.name],
                internal=1,
                attrs=attrs
            )

            logger.debug("-- -- insert: {}".format(T))

            xgraph.insert(T)

        # TODO: test this functionality more thoroughly, lots of edge cases
        if (len(ttXs) > 0 and all([ttX.subgraph == ttXs[0].subgraph for ttX in ttXs])
                and ttXs[0].subgraph is not None):
            logger.debug("-- -- update subgraph of {} from {} to: {}"
                         .format(X.name, X.subgraph, ttXs[0].subgraph))
            X.subgraph = ttXs[0].subgraph

        # TRANSFORM X
        old_shape = X.shapes[:]

        transpose_transform_func = xop_registry.get_xop_transpose_transform(X.type[0])

        transpose_transform_func(X, axes)

        logger.debug("-- -- X old shapes: {}, axes: {}, new shapes: {}"
                     .format(old_shape, axes, X.shapes))

    return changes


def sweep_transposes_flow(
    xgraph: XGraph,
    bottom_Xs: List[XLayer],
    X: XLayer,
    top_Xs: List[XLayer],
    target: Optional[str] = None,
    **kwargs: Any,
) -> XLayer:
    """
    Sweep transpose layers in the XGraph following the flow of the directed
    graph.

    If target is specified, only sweep transposes from inside to outside
    target subgraphs.

    This is functionality is used for supporting layout transformation for
    models with subgraphs

    TRANSFORMATIONS
        Transform T --> X in X --> T if X is a valid layer and T ----^
            the transposes are the same
        Transform T --> X in X --> T if X is a valid layer
    """

    # Used for getting information on operations
    xop_registry = XOpRegistry()

    bX_all_transposes = all([bX.type[0] == 'Transpose' for bX in bottom_Xs])
    bX_all_eq_axes = all([bX.attrs['axes'] == bottom_Xs[0].attrs['axes']
                          for bX in bottom_Xs])

    changes = False

    if (len(bottom_Xs) > 0 and bX_all_transposes and bX_all_eq_axes
            and X.type[0] in ['Transpose']):
        # Check if we have two transposes that cancel eachother out
        bX = bottom_Xs[0]
        axes = X.attrs['axes']
        bX_axes = bX.attrs['axes']

        if [axes[i] for i in bX_axes] == [0, 1, 2, 3]:
            logger.debug(f"-- -- Merge transposes: bX: {[bX.name for bX in bottom_Xs]}, "
                         f"X: {X.name}, tX: {[tX.name for tX in top_Xs]}")
            changes = True
            xgraph.remove(X.name)

            [xgraph.remove(bX.name) for bX in bottom_Xs]

    elif (bX_all_transposes and bX_all_eq_axes
            and X.type[0] in xop_registry.get_xops_with_transpose_transform()):

        logger.debug(f"-- -- Sweep transpose: bX: {[bX.name for bX in bottom_Xs]}, "
                     f"X: {X.name}, tX: {[tX.name for tX in top_Xs]}")

        axes = bottom_Xs[0].attrs['axes'][:]

        # Transposes can have only one input
        assert all([len(bX.bottoms) == 1 for bX in bottom_Xs])
        bbXs = [xgraph.get(bX.bottoms[0]) for bX in bottom_Xs]

        # Sweep transposes outside the subgraph
        if target is None or (X.target == target and all(
            [(bbX.target == target and bbX.subgraph == X.subgraph) for bbX in bbXs])):

            changes = True

            bottom_names = [bX.name for bX in bottom_Xs]

            for bX, bbX in zip(bottom_Xs, bbXs):

                new_tops = [b for b in bX.tops if b != X.name]

                if new_tops == []:
                    xgraph.remove(bX.name)
                    # Important to not touch bX after removal
                    continue
                else:
                    bX.tops = new_tops
                    bbX.tops.append(X.name)
                    X.bottoms = [b if b != bX.name else bbX.name for b in X.bottoms]

            if len(top_Xs) > 0:
                # Insert transposes
                for i, tX in enumerate(top_Xs):

                    if len(top_Xs) > 1:
                        t_name = f"{i}_split_{'_'.join(bottom_names)}"
                        # bX.name for bX in bottom_Xs
                    elif len(bottom_Xs) > 1:
                        t_name = f"merge_{'_'.join(bottom_names)}"
                        # [bX.name for bX in bottom_Xs]
                    else:
                        t_name = "moved_" + bX.name

                    t_shape = X.shapes[:]
                    logger.debug(f"-- -- t_shape: {t_shape}")
                    attrs = {'axes': axes}

                    T = xlayer.defaultXLayer()
                    T = T._replace(
                        name=t_name,
                        type=['Transpose'],
                        shapes=t_shape,
                        sizes=X.sizes[:],
                        layer=[t_name],
                        tops=[tX.name],
                        bottoms=[X.name],
                        internal=1,
                        attrs=attrs
                    )

                    # logger.debug("-- -- insert: {}".format(T))

                    xgraph.insert(T)
            else:
                # No top layers: Insert 1 transpose
                if len(bottom_Xs) > 1:
                    t_name = f"merge_{'_'.join(bottom_names)}"
                else:
                    t_name = "moved_" + bX.name

                t_shape = X.shapes[:]
                logger.debug(f"-- -- t_shape: {t_shape}")
                attrs = {'axes': axes}

                T = xlayer.defaultXLayer()
                T = T._replace(
                    name=t_name,
                    type=['Transpose'],
                    shapes=t_shape,
                    sizes=X.sizes[:],
                    layer=[t_name],
                    tops=[],
                    bottoms=[X.name],
                    internal=1,
                    attrs=attrs
                )

                xgraph.add(T)


            # TRANSFORM X
            axes_t = [axes[i] for i in axes]
            old_shape = X.shapes[:]

            transpose_transform_func = xop_registry.get_xop_transpose_transform(X.type[0])

            transpose_transform_func(X, axes_t)

            logger.debug(f"-- -- X old shapes: {old_shape}, axes: {axes_t}, "
                         f"new shapes: {X.shapes}")

    return changes


def merge_padding(
    xgraph: XGraph,
    bottom_Xs: List[XLayer],
    X: XLayer,
    top_Xs: List[XLayer],
    **kwargs: Any,
) -> bool:
    """
    Try to merge padding layer into succeeding Convolution or Pooling layer
    """
    changes = False

    if 'Pad' in X.type:
        if len(top_Xs) != 1:
            raise ValueError("Impossible to merge padding layer. Padding layer "
                             "must always be followed by at least one other "
                             f"layer, but found: {len(top_Xs)}.")

        top_X = top_Xs[0]
        if top_X.type[0] in ['Convolution', 'Pooling']:
            # Merge padding layer
            changes = True

            padding = top_X.attrs['padding']
            # padding_hw = top_X.paddings
            layout = top_X.attrs['data_layout']
            if layout == 'NCHW':
                batches, channels, height, width = padding
            else:
                batches, height, width, channels = padding

            if width[0] != width[1] and height[0] != height[1]:
                logger.warn("[WARNING] Merging asymmetric padding into"
                            " succeeding Convolution or Pooling layer")

            # [batches, channels, height, width]
            new_padding = [()]
            attrs = top_X.attrs
            attrs.update(
                {
                    'padding': [[t_pad_b + pad_b, t_pad_a + pad_a]
                                for (t_pad_b, t_pad_a), (pad_b, pad_a) in
                                zip(padding, X.attrs['padding'])]
                }
            )
            top_X.layer = [X.name] + top_X.layer[:]

            xgraph.remove(X.name)

    return changes


def merge_bias(
    xgraph: XGraph,
    bottom_Xs: List[XLayer],
    X: XLayer,
    top_Xs: List[XLayer],
    **kwargs: Any,
) -> bool:
    """
    Try to merge bias layer into preceding Convolution or Dense layer
    """
    changes = False

    # Eltwise operation with bias add
    if 'BiasAdd' in X.type or ('Eltwise' in X.type and X.data != []):
        if len(bottom_Xs) != 1:
            raise ValueError("Impossible to merge bias layer. Bias layer"
                             " must always be preceded by exactly one layer,"
                             f" but found: {len(bottom_Xs)}.")

        bottom_X = bottom_Xs[0]
        if bottom_X.type[0] in ['Convolution', 'Dense', 'Conv2DTranspose']:
            bias = bottom_X.data.biases + X.data[0]
            # if bottom_X.bias and X.data is not None else X.data
            changes = True

            # TODO: remove Relay specific code in core functionality
            if 'relay_id' in bottom_X.attrs and 'relay_id' in X.attrs:
                bottom_X.attrs['relay_id'] += X.attrs['relay_id']

            bottom_X.data = xlayer.ConvData(bottom_X.data.weights, bias)
            bottom_X.layer = bottom_X.layer[:] + [X.name]

            # Remove the bias addition node
            xgraph.remove(X.name)

    return changes


def merge_batchnorm_into_conv(
    xgraph: XGraph,
    bottom_Xs: List[XLayer],
    X: XLayer,
    top_Xs: List[XLayer],
    **kwargs: Any,
) -> bool:
    """
    Try to merge batch normalization layer into preceding Convolution

    Conv = Conv + BN
         = Gamma*((Wx+B) - Mu)/Sigma + Beta
         = Gamma*(W/Sigma)x + Gamma*(-Mu+B)/Sigma + Beta
    """
    changes = False
    if 'BatchNorm' in X.type:
        if not all(['Convolution' in b_X.type for b_X in bottom_Xs]):
            # Batch norm can only be merged with preceding Convolution
            return changes
        changes = True

        if not isinstance(X.data, xlayer.BatchData):
            raise ValueError(f"Invalid batch normalization data type: {type(X.data)}, "
                             "should be of type: xlayer.BatchData")

        for bottom_X in bottom_Xs:
            if not isinstance(bottom_X.data, xlayer.ConvData):
                raise ValueError("Invalid convolution parameters data type: "
                                 f"{type(bottom_X.data)}, should be of type: xlayer.ConvData")

            # Weights should have layout: OIHW at this point
            conv_weights = bottom_X.data.weights
            conv_biases = bottom_X.data.biases
            bn_mu, bn_sigma_square = X.data.mu, X.data.sigma_square
            bn_gamma, bn_beta = X.data.gamma, X.data.beta
            shape = (conv_weights.shape[0], 1, 1, 1)
            epsilon = X.attrs['epsilon']

            assert(conv_weights.shape[0] == conv_biases.shape[0] ==
                   bn_mu.shape[0] == bn_sigma_square.shape[0])
            conv_weights = bn_gamma.reshape(shape) * (
                conv_weights / np.sqrt(bn_sigma_square.reshape(shape) + epsilon))
            conv_biases = bn_gamma * (
                (conv_biases - bn_mu) / np.sqrt(bn_sigma_square + epsilon)) + bn_beta

            bottom_X.data = xlayer.ConvData(conv_weights, conv_biases)
            bottom_X.layer = bottom_X.layer[:] + [X.name]

        # Remove the batch norm node
        xgraph.remove(X.name)

    return changes


def merge_scale_into_conv_bn(
    xgraph: XGraph,
    bottom_Xs: List[XLayer],
    X: XLayer,
    top_Xs: List[XLayer],
    **kwargs: Any,
) -> bool:
    """
    Try to merge scaling layer into preceding Convolution(s) or BatchNorm(s)

    Conv = Conv + Scale : (Wx+B)*gamma + beta = (W*gamma)x + B*gamma + beta
    Scale = BN + Scale  :
        ((x- mu)/sigma)*gamma + beta = x*(gamma/sigma) +
        (beta - mu * gamma / sigma)
    """
    changes = False
    if 'Scale' in X.type:
        # TODO: Scale + Scale
        if not all([b_X.type[0] in ['Convolution', 'BatchNorm']
                    for b_X in bottom_Xs]):
            # Scaling can only be merged into Convolution and BatchNorm
            return changes

        changes = True

        if not isinstance(X.data, xlayer.ScaleData):
            raise ValueError(f"Invalid batch normalization data type: {type(X.data)}, "
                             " should be of type: xlayer.ScaleData")

        for bottom_X in bottom_Xs:

            if bottom_X.type[0] == 'Convolution':
                if not isinstance(bottom_X.data, xlayer.ConvData):
                    raise ValueError(f"Invalid convolution parameters data type: {type(bottom_X.data)}, "
                                     "should be of type: xlayer.ConvData")

                # Weights should have layout: OIHW at this point
                conv_weights = bottom_X.data.weights
                conv_biases = bottom_X.data.biases
                gamma, beta = X.data.gamma, X.data.beta
                shape = (conv_weights.shape[0], 1, 1, 1)

                assert(conv_weights.shape[0] == conv_biases.shape[0] ==
                       gamma.shape[0] == beta.shape[0])
                conv_weights = conv_weights * gamma.reshape(shape)
                conv_biases = conv_biases * gamma + beta

                bottom_X.data = xlayer.ConvData(conv_weights, conv_biases)
                bottom_X.layer = bottom_X.layer[:] + [X.name]

            elif bottom_X.type[0] == 'BatchNorm':
                if not isinstance(bottom_X.data, xlayer.BatchData):
                    raise ValueError(f"Invalid BatchNorm parameters data type: {type(bottom_X.data)}"
                                     "should be of type: xlayer.BatchData")

                if not isinstance(X.data, xlayer.ScaleData):
                    raise ValueError(f"Invalid scaling layer data type: {type(X.data)}, "
                                     "should be of type: xlayer.ScaleData")

                # Weights should have layout: OIHW at this point
                gamma, beta = X.data.gamma, X.data.beta
                bn_mu, bn_sigma_square = bottom_X.data.mu, bottom_X.data.sigma_square
                bn_gamma, bn_beta = bottom_X.data.gamma, bottom_X.data.beta

                epsilon = bottom_X.attrs['epsilon']

                assert (bn_mu.shape[0] == bn_sigma_square.shape[0] == \
                        bn_gamma.shape[0] == bn_beta.shape[0] == \
                        gamma.shape[0] == beta.shape[0])

                new_gamma = gamma * bn_gamma
                new_beta = gamma * bn_beta + beta

                bottom_X.data = xlayer.BatchData(bn_mu, bn_sigma_square, new_gamma, new_beta)
                bottom_X.layer = bottom_X.layer[:] + [X.name]

        # Remove the Scale node
        xgraph.remove(X.name)

    return changes


def merge_relu(
    xgraph: XGraph,
    bottom_Xs: List[XLayer],
    X: XLayer,
    top_Xs: List[XLayer],
    **kwargs: Any,
) -> bool:
    """
    Try to merge Relu layer into preceding
    Convolution/Eltwise/BatchNorm/Scale layer
    """
    changes = False
    if 'ReLU' in X.type or 'pReLU' in X.type:
        if len(bottom_Xs) != 1:
            raise ValueError("Impossible to merge (p)ReLU layer. (p)ReLU layer "
                             "must always be preceded by exactly one layer, "
                             f"but found: {len(bottom_Xs)}.")

        bottom_X = bottom_Xs[0]
        if bottom_X.type[0] not in ['Convolution', 'Eltwise', 'BatchNorm', 'Scale']:
            return changes
        changes = True

        attrs = bottom_X.attrs
        if X.type[0] == 'pReLU':
            attrs['alpha'] = P.attrs['alpha']
            attrs['activation'] = 'pReLU'
        elif X.type[0] == 'ReLU':
            attrs['activation'] = 'ReLU'

        # bottom_X.attrs = attrs
        bottom_X.layer = bottom_X.layer[:] + [X.name]

        # Remove the ReLU node
        xgraph.remove(X.name)

    return changes
