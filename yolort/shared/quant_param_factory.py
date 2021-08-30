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
Module for creating, tracking and storing quantization parameters
"""

import collections
import numpy as np
import logging

from yolort.graph.layer import xlayer

from .quant_params import QuantParams

logger = logging.getLogger('pyxir')


class LayerParams:

    """
    Class for storing layer parameters
    ! Necessary to make quantize_base work, which assumes that
    layer parameters are be passed in a LayerParams object with
    a data attribute (Caffe specific?)
    # TODO: We should not depend on a specific framework at this level
    #   in the stack

    Args:
        data (np.ndarray): The layer parameter data to be stored in this object

    Attributes:
        data (np.ndarray): The layer parameter data

    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class QuantParamFactory:
    # bw = bitwidth
    # th = thresholdsd
    # bw_layer_in = None
    # th_layer_in = None

    # bw_layer_out = None
    # th_layer_out = None

    # bw_params = None
    # th_params = None

    def __init__(self):
        self.bw_layer_in = collections.OrderedDict()
        self.th_layer_in = collections.OrderedDict()

        self.bw_layer_out = collections.OrderedDict()
        self.th_layer_out = collections.OrderedDict()

        self.bw_params = collections.OrderedDict()
        self.th_params = collections.OrderedDict()

        self.quant_params = QuantParams()

    def get_default_quant_params(
        self,
        name,
        bitwidth,
        channels,
        th_layer_in,
        th_layer_out,
    ):
        """
        TODO, docstring
        """
        sf_layer_in = th_layer_in / (np.power(2.0, bitwidth - 1) - 1)
        sf_layer_out = th_layer_out / (np.power(2.0, bitwidth - 1) - 1)
        sf_params = np.array([1.0]) / (np.power(2.0, bitwidth - 1) - 1)

        # Scale and postscale shift for division
        multiplier = th_layer_in / th_layer_out
        logger.info(f"th_layer_in: {th_layer_in}, th_layer_out: {th_layer_out}, "
                    f"ratio: {multiplier}")

        # TODO
        prescale_shift_max = 0
        scale_bitwidth = 16
        postscale_shift_max = 40

        canonical_factor = np.power(2, np.ceil(np.log2(multiplier)))
        canonical_form = multiplier / canonical_factor

        shift = np.log2(canonical_factor)
        lshift = np.clip(shift, 0, None)
        rshift = -np.clip(shift, None, 0)

        prescale_shift = np.clip(rshift, 0, prescale_shift_max)
        rshift -= prescale_shift

        postscale_shift = np.clip(rshift, 0, postscale_shift_max)
        rshift -= postscale_shift

        scale = np.clip(canonical_form * np.power(2, lshift - rshift), 0,
                        np.power(2, scale_bitwidth - 1))

        remaining_available_lshift = np.floor(np.log2(np.power(2, scale_bitwidth - 1) / scale))
        remaining_available_rshift = postscale_shift_max - postscale_shift
        remaining_available_shift = np.fmin(remaining_available_lshift, remaining_available_rshift)

        scale *= np.power(2, remaining_available_shift)
        postscale_shift += remaining_available_shift

        prescale_shift = prescale_shift.astype(int)
        scale = np.round(scale).astype(int)
        postscale_shift = postscale_shift.astype(int)

        prescale_shift = np.clip(prescale_shift, 0, prescale_shift_max)
        scale = np.clip(scale, 0, np.power(2, scale_bitwidth) - 1)
        postscale_shift = np.clip(postscale_shift, 0, postscale_shift_max)

        logger.info(f"prescale: {prescale_shift}, scale: {scale}, "
                    f"postscale: {postscale_shift}")
        logger.info(f"Type: prescale: {type(prescale_shift.astype(np.int32))}, "
                    f"scale: {type(scale.astype(np.int32))}, "
                    f"postscale: {type(postscale_shift.astype(np.int32))}")

        qp = {
            "name": name,
            "bw_layer_in": bitwidth,  # unused by xfdnn
            "bw_layer_out": bitwidth,  # unused by xfdnn
            "bw_params": bitwidth,
            "th_layer_in": th_layer_in,
            "th_layer_out": th_layer_out,
            "th_params": [1] * channels,
            "sf_layer_in": sf_layer_in,  # unused by xfdnn
            "sf_layer_out": sf_layer_out,  # unused by xfdnn
            "sf_params": sf_params.tolist() * channels,
            "prescale_shift": [int(prescale_shift.astype(np.int32))] * channels,
            "scale": [int(scale.astype(np.int32))] * channels,
            "postscale_shift": [int(postscale_shift.astype(np.int32))] * channels
        }
        return qp

    def save_to_dpu_v1_json(self, quant_layers, fname):
        json_payload = {}
        json_payload["network"] = []

        # Declare local quantization variables
        bw_layer_in = collections.OrderedDict()
        bw_layer_out = collections.OrderedDict()
        bw_params = collections.OrderedDict()
        sf_layer_in = collections.OrderedDict()
        sf_layer_out = collections.OrderedDict()
        sf_params = collections.OrderedDict()
        prescale_shift = collections.OrderedDict()
        scale = collections.OrderedDict()
        postscale_shift = collections.OrderedDict()

        logger.info("Writing output files to %s" % fname)
        # with open(fname, "w") as g:

        # xDNN v3
        isV2 = False

        for (name, layer_type, layer_params) in quant_layers:

            if layer_type in ['Concat']:
                bw_layer_in[name] = self.bw_layer_in[name]
                bw_layer_out[name] = self.bw_layer_out[name]

                sf_layer_in[name] = self.th_layer_in[name][0] / (
                    np.power(2.0, self.bw_layer_in[name] - 1) - 1)
                sf_layer_out[name] = self.th_layer_out[name][0] / (
                    np.power(2.0, self.bw_layer_out[name] - 1) - 1)

                multiplier = np.repeat(sf_layer_in[name] / sf_layer_out[name], 1)

                prescale_shift[name] = np.zeros_like(multiplier)
                scale[name] = np.ones_like(multiplier)
                postscale_shift[name] = np.zeros_like(multiplier)

                self.quant_params.append(name, {
                    "name": name,
                    "bw_layer_in": self.bw_layer_in[name],  # unused by xfdnn
                    "bw_layer_out": self.bw_layer_out[name],  # unused by xfdnn
                    "th_layer_in": self.th_layer_in[name][0].tolist(),
                    "th_layer_out": self.th_layer_out[name][0].tolist(),
                    # unused by xfdnn
                    "sf_layer_in": sf_layer_in[name].tolist(),
                    # unused by xfdnn
                    "sf_layer_out": sf_layer_out[name].tolist(),
                    "prescale_shift": prescale_shift[name].astype(int).tolist(),
                    "scale": scale[name].astype(int).tolist(),
                    "postscale_shift": postscale_shift[name].astype(int).tolist()
                })
            if layer_type in ["Convolution"]:
                bw_layer_in[name] = self.bw_layer_in[name]
                bw_layer_out[name] = self.bw_layer_out[name]
                bw_params[name] = self.bw_params[name]

                sf_layer_in[name] = self.th_layer_in[name][0] / (
                    np.power(2.0, self.bw_layer_in[name] - 1) - 1)
                sf_layer_out[name] = self.th_layer_out[name][0] / (
                    np.power(2.0, self.bw_layer_out[name] - 1) - 1)
                sf_params[name] = self.th_params[name] / (
                    np.power(2.0, self.bw_params[name] - 1) - 1)

                multiplier = np.repeat(sf_layer_in[name] * sf_params[name] / sf_layer_out[name], 1)

                prescale_shift[name] = np.zeros_like(multiplier)
                scale[name] = np.ones_like(multiplier)
                postscale_shift[name] = np.zeros_like(multiplier)

                for i in range(len(multiplier)):
                    if multiplier[i] == 0:
                        prescale_shift[name][i] = 0
                        scale[name][i] = 0
                        postscale_shift[name][i] = 0
                    else:
                        if isV2:
                            if bw_layer_in[name] == 8:
                                prescale_shift_max = 16
                                scale_bitwidth = 8
                                postscale_shift_max = 8
                            elif bw_layer_in[name] == 16:
                                prescale_shift_max = 32
                                scale_bitwidth = 8
                                postscale_shift_max = 8
                        else:
                            if bw_layer_in[name] == 8:
                                prescale_shift_max = 0
                                scale_bitwidth = 16
                                postscale_shift_max = 32
                            elif bw_layer_in[name] == 16:
                                prescale_shift_max = 24
                                scale_bitwidth = 16
                                postscale_shift_max = 24

                        canonical_factor = np.power(2, np.ceil(np.log2(multiplier[i])))
                        canonical_form = multiplier[i] / canonical_factor

                        shift = np.log2(canonical_factor)
                        lshift = np.clip(shift, 0, None)
                        rshift = -np.clip(shift, None, 0)

                        prescale_shift[name][i] = np.clip(rshift, 0, prescale_shift_max)
                        rshift -= prescale_shift[name][i]

                        postscale_shift[name][i] = np.clip(rshift, 0, postscale_shift_max)
                        rshift -= postscale_shift[name][i]

                        scale[name][i] = np.clip(canonical_form * np.power(2, lshift - rshift),
                                                 0, np.power(2, scale_bitwidth - 1))

                        remaining_available_lshift = np.floor(
                            np.log2(np.power(2, scale_bitwidth - 1) / scale[name][i]))

                        remaining_available_rshift = postscale_shift_max - postscale_shift[name][i]
                        remaining_available_shift = np.fmin(remaining_available_lshift,
                                                            remaining_available_rshift)

                        scale[name][i] *= np.power(2, remaining_available_shift)
                        postscale_shift[name][i] += remaining_available_shift

                        prescale_shift[name][i] = prescale_shift[name][i].astype(int)
                        scale[name][i] = np.round(scale[name][i]).astype(int)
                        postscale_shift[name][i] = postscale_shift[name][i].astype(int)

                        prescale_shift[name][i] = np.clip(
                            prescale_shift[name][i], 0, prescale_shift_max)
                        scale[name][i] = np.clip(
                            scale[name][i], 0, np.power(2, scale_bitwidth) - 1)
                        postscale_shift[name][i] = np.clip(
                            postscale_shift[name][i], 0, postscale_shift_max)

                self.quant_params.append(
                    name,
                    {
                        "name": name,
                        "bw_layer_in": self.bw_layer_in[name],  # unused by xfdnn
                        "bw_layer_out": self.bw_layer_out[name],  # unused by xfdnn
                        "bw_params": self.bw_params[name],
                        "th_layer_in": self.th_layer_in[name][0].tolist(),
                        "th_layer_out": self.th_layer_out[name][0].tolist(),
                        "th_params": self.th_params[name].tolist(),
                        # unused by xfdnn
                        "sf_layer_in": sf_layer_in[name].tolist(),
                        # unused by xfdnn
                        "sf_layer_out": sf_layer_out[name].tolist(),
                        # unused by xfdnn
                        "sf_params": sf_params[name].tolist(),
                        "prescale_shift": prescale_shift[name].astype(int).tolist(),
                        "scale": scale[name].astype(int).tolist(),
                        "postscale_shift": postscale_shift[name].astype(int).tolist()
                    }
                )
            elif layer_type in ["BatchNorm", "Scale"]:
                bw_layer_in[name] = self.bw_layer_in[name]
                bw_layer_out[name] = self.bw_layer_out[name]
                bw_params[name] = self.bw_params[name]

                sf_layer_in[name] = self.th_layer_in[name][0] / (
                    np.power(2.0, self.bw_layer_in[name] - 1) - 1)
                sf_layer_out[name] = self.th_layer_out[name][0] / (
                    np.power(2.0, self.bw_layer_out[name] - 1) - 1)
                sf_params[name] = self.th_params[name] / (
                    np.power(2.0, self.bw_params[name] - 1) - 1)

                if layer_type in ["BatchNorm"]:
                    lp_data = layer_params[1].data + layer_params[2].data
                    multiplier = np.repeat(
                        sf_layer_in[name] * sf_params[name] / sf_layer_out[name] * np.round(
                            np.clip(
                                1.0 / np.sqrt(lp_data) / sf_params[name],
                                -np.power(2.0, bw_params[name]-1) + 1,
                                np.power(2.0, bw_params[name]-1) - 1
                            )
                        ), 1)

                elif layer_type in ["Scale"]:
                    multiplier = np.repeat(
                        sf_layer_in[name] * sf_params[name] / sf_layer_out[name] * np.round(
                            np.clip(
                                layer_params[0].data / sf_params[name],
                                -np.power(2.0, bw_params[name]-1) + 1,
                                np.power(2.0, bw_params[name]-1) - 1
                            )
                        ), 1)

                # TODO
                # logger.debug(f"multiplier: {name}")
                # logger.debug(multiplier)
                multiplier_sign = np.sign(multiplier)
                multiplier = np.absolute(multiplier)

                prescale_shift[name] = np.zeros_like(multiplier)
                scale[name] = np.ones_like(multiplier)
                postscale_shift[name] = np.zeros_like(multiplier)

                for i in range(len(multiplier)):
                    if multiplier[i] == 0:
                        prescale_shift[name][i] = 0
                        scale[name][i] = 0
                        postscale_shift[name][i] = 0
                    else:
                        if isV2:
                            if bw_layer_in[name] == 8:
                                prescale_shift_max = 0
                                scale_bitwidth = 8
                                postscale_shift_max = 8
                            elif bw_layer_in[name] == 16:
                                prescale_shift_max = 0
                                scale_bitwidth = 8
                                postscale_shift_max = 8
                        else:
                            if bw_layer_in[name] == 8:
                                prescale_shift_max = 0
                                scale_bitwidth = 16
                                postscale_shift_max = 40
                            elif bw_layer_in[name] == 16:
                                prescale_shift_max = 0
                                scale_bitwidth = 16
                                postscale_shift_max = 32

                        canonical_factor = np.power(2, np.ceil(np.log2(multiplier[i])))
                        canonical_form = multiplier[i] / canonical_factor

                        shift = np.log2(canonical_factor)
                        lshift = np.clip(shift, 0, None)
                        rshift = -np.clip(shift, None, 0)

                        prescale_shift[name][i] = np.clip(rshift, 0, prescale_shift_max)
                        rshift -= prescale_shift[name][i]

                        postscale_shift[name][i] = np.clip(rshift, 0, postscale_shift_max)
                        rshift -= postscale_shift[name][i]
                        # TODO
                        # scale[name][i] = np.clip(canonical_form *
                        #   np.power(2, lshift - rshift), 0,
                        #   np.power(2, scale_bitwidth - 1))
                        scale[name][i] = np.clip(
                            canonical_form * np.power(2, lshift - rshift),
                            0,
                            np.power(2, scale_bitwidth - 1)
                        )

                        remaining_available_lshift = np.floor(
                            np.log2(np.power(2, scale_bitwidth - 1) / scale[name][i]))
                        remaining_available_rshift = postscale_shift_max - postscale_shift[name][i]
                        remaining_available_shift = np.fmin(remaining_available_lshift,
                                    remaining_available_rshift)

                        scale[name][i] *= np.power(2, remaining_available_shift)
                        postscale_shift[name][i] += remaining_available_shift

                        prescale_shift[name][i] = prescale_shift[name][i].astype(int)
                        scale[name][i] = np.round(scale[name][i]).astype(int)
                        postscale_shift[name][i] = postscale_shift[name][i].astype(int)

                        prescale_shift[name][i] = np.clip(
                            prescale_shift[name][i], 0, prescale_shift_max)
                        scale[name][i] = np.clip(
                            scale[name][i], 0, np.power(2, scale_bitwidth) - 1)
                        postscale_shift[name][i] = np.clip(
                            postscale_shift[name][i], 0, postscale_shift_max)

                # TODO
                scale[name] *= multiplier_sign

                self.quant_params.append(
                    name,
                    {
                        "name": name,
                        "bw_layer_in": self.bw_layer_in[name],  # unused by xfdnn
                        "bw_layer_out": self.bw_layer_out[name],  # unused by xfdnn
                        "bw_params": self.bw_params[name],
                        "th_layer_in": self.th_layer_in[name][0].tolist(),
                        "th_layer_out": self.th_layer_out[name][0].tolist(),
                        "th_params": self.th_params[name].tolist(),
                        # unused by xfdnn
                        "sf_layer_in": sf_layer_in[name].tolist(),
                        # unused by xfdnn
                        "sf_layer_out": sf_layer_out[name].tolist(),
                        "sf_params": sf_params[name].tolist(),  # unused by xfdnn
                        "prescale_shift": prescale_shift[name].astype(int).tolist(),
                        "scale": scale[name].astype(int).tolist(),
                        "postscale_shift": postscale_shift[name].astype(int).tolist()
                    }
                )
            elif layer_type in ["Eltwise"]:
                bw_layer_in[name] = self.bw_layer_in[name]
                bw_layer_out[name] = self.bw_layer_out[name]

                sf_layer_in[name] = self.th_layer_in[name][0] / (
                    np.power(2.0, self.bw_layer_in[name] - 1) - 1)
                sf_layer_out[name] = self.th_layer_out[name][0] / (
                    np.power(2.0, self.bw_layer_out[name] - 1) - 1)

                multiplier = np.repeat(sf_layer_in[name] / sf_layer_out[name], 1)

                prescale_shift[name] = np.zeros_like(multiplier)
                scale[name] = np.ones_like(multiplier)
                postscale_shift[name] = np.zeros_like(multiplier)

                for i in range(len(multiplier)):
                    if multiplier[i] == 0:
                        prescale_shift[name][i] = 0
                        scale[name][i] = 0
                        postscale_shift[name][i] = 0
                    else:
                        if isV2:
                            if bw_layer_in[name] == 8:
                                prescale_shift_max = 0
                                scale_bitwidth = 8
                                postscale_shift_max = 8
                            elif bw_layer_in[name] == 16:
                                prescale_shift_max = 0
                                scale_bitwidth = 8
                                postscale_shift_max = 8
                        else:
                            if bw_layer_in[name] == 8:
                                prescale_shift_max = 0
                                scale_bitwidth = 16
                                postscale_shift_max = 40
                            elif bw_layer_in[name] == 16:
                                prescale_shift_max = 0
                                scale_bitwidth = 16
                                postscale_shift_max = 32

                        canonical_factor = np.power(2, np.ceil(np.log2(multiplier[i])))
                        canonical_form = multiplier[i] / canonical_factor

                        shift = np.log2(canonical_factor)
                        lshift = np.clip(shift, 0, None)
                        rshift = -np.clip(shift, None, 0)

                        prescale_shift[name][i] = np.clip(rshift, 0, prescale_shift_max)
                        rshift -= prescale_shift[name][i]

                        postscale_shift[name][i] = np.clip(rshift, 0, postscale_shift_max)
                        rshift -= postscale_shift[name][i]

                        scale[name][i] = np.clip(
                            canonical_form * np.power(2, lshift - rshift),
                            0,
                            np.power(2, scale_bitwidth - 1)
                        )

                        remaining_available_lshift = np.floor(
                            np.log2(np.power(2, scale_bitwidth - 1) / scale[name][i]))
                        remaining_available_rshift = postscale_shift_max - postscale_shift[name][i]
                        remaining_available_shift = np.fmin(
                            remaining_available_lshift, remaining_available_rshift)

                        scale[name][i] *= np.power(2, remaining_available_shift)
                        postscale_shift[name][i] += remaining_available_shift

                        prescale_shift[name][i] = prescale_shift[name][i].astype(int)
                        scale[name][i] = np.round(scale[name][i]).astype(int)
                        postscale_shift[name][i] = postscale_shift[name][i].astype(int)

                        prescale_shift[name][i] = np.clip(
                            prescale_shift[name][i],
                            0,
                            prescale_shift_max
                        )
                        scale[name][i] = np.clip(scale[name][i], 0, np.power(2, scale_bitwidth) - 1)
                        postscale_shift[name][i] = np.clip(
                            postscale_shift[name][i],
                            0,
                            postscale_shift_max
                        )

                self.quant_params.append(
                    name,
                    {
                        "name": name,
                        "bw_layer_in": self.bw_layer_in[name],  # unused by xfdnn
                        "bw_layer_out": self.bw_layer_out[name],  # unused by xfdnn
                        "th_layer_in": self.th_layer_in[name][0].tolist(),
                        "th_layer_out": self.th_layer_out[name][0].tolist(),
                        # unused by xfdnn
                        "sf_layer_in": sf_layer_in[name].tolist(),
                        # unused by xfdnn
                        "sf_layer_out": sf_layer_out[name].tolist(),
                        "prescale_shift": prescale_shift[name].astype(int).tolist(),
                        "scale": scale[name].astype(int).tolist(),
                        "postscale_shift": postscale_shift[name].astype(int).tolist()
                    }
                )
            elif layer_type in ["Pooling"]:
                # L = next(L for L in net_parameter.layer if L.name in
                #   net.top_names[name])
                # num_output = net.blobs[net.top_names[name][0]].data.shape[1]
                num_output = 1

                # if L.pooling_param.pool != caffe.params.Pooling.AVE:
                #    continue

                if isV2:
                    continue

                bw_layer_in[name] = self.bw_layer_in[name]
                bw_layer_out[name] = self.bw_layer_out[name]

                sf_layer_in[name] = self.th_layer_in[name][0] / (
                    np.power(2.0, self.bw_layer_in[name] - 1) - 1)
                sf_layer_out[name] = self.th_layer_out[name][0] / (
                    np.power(2.0, self.bw_layer_out[name] - 1) - 1)

                # multiplier = np.repeat(sf_layer_in / sf_layer_out /
                #   pow(L.pooling_param.kernel_size, 2), num_output)
                multiplier = np.repeat(
                    sf_layer_in[name] / sf_layer_out[name] / layer_params.data[0], num_output)

                prescale_shift[name] = np.zeros_like(multiplier)
                scale[name] = np.ones_like(multiplier)
                postscale_shift[name] = np.zeros_like(multiplier)

                for i in range(len(multiplier)):
                    if multiplier[i] == 0:
                        prescale_shift[name][i] = 0
                        scale[name][i] = 0
                        postscale_shift[name][i] = 0
                    else:
                        if isV2:
                            if bw_layer_in[name] == 8:
                                prescale_shift_max = 0
                                scale_bitwidth = 8
                                postscale_shift_max = 8
                            elif bw_layer_in[name] == 16:
                                prescale_shift_max = 0
                                scale_bitwidth = 8
                                postscale_shift_max = 8
                        else:
                            if bw_layer_in[name] == 8:
                                prescale_shift_max = 0
                                scale_bitwidth = 16
                                postscale_shift_max = 40
                            elif bw_layer_in[name] == 16:
                                prescale_shift_max = 0
                                scale_bitwidth = 16
                                postscale_shift_max = 32

                        canonical_factor = np.power(2, np.ceil(np.log2(multiplier[i])))
                        canonical_form = multiplier[i] / canonical_factor

                        shift = np.log2(canonical_factor)
                        lshift = np.clip(shift, 0, None)
                        rshift = -np.clip(shift, None, 0)

                        prescale_shift[name][i] = np.clip(rshift, 0,
                                                          prescale_shift_max)
                        rshift -= prescale_shift[name][i]

                        postscale_shift[name][i] = np.clip(rshift, 0,
                                                           postscale_shift_max)
                        rshift -= postscale_shift[name][i]

                        scale[name][i] = np.clip(
                            canonical_form * np.power(2, lshift - rshift),
                            0,
                            np.power(2, scale_bitwidth - 1)
                        )

                        remaining_available_lshift = np.floor(
                            np.log2(np.power(2, scale_bitwidth - 1) / scale[name][i]))
                        remaining_available_rshift = postscale_shift_max - postscale_shift[name][i]
                        remaining_available_shift = np.fmin(
                            remaining_available_lshift,
                            remaining_available_rshift
                        )

                        scale[name][i] *= np.power(2, remaining_available_shift)
                        postscale_shift[name][i] += remaining_available_shift

                        prescale_shift[name][i] = prescale_shift[name][i].astype(int)
                        scale[name][i] = np.round(scale[name][i]).astype(int)
                        postscale_shift[name][i] = postscale_shift[name][i].astype(int)

                self.quant_params.append(
                    name, {
                        "name": name,
                        "bw_layer_in": self.bw_layer_in[name],  # unused by xfdnn
                        "bw_layer_out": self.bw_layer_out[name],  # unused by xfdnn
                        "th_layer_in": self.th_layer_in[name][0].tolist(),
                        "th_layer_out": self.th_layer_out[name][0].tolist(),
                        # unused by xfdnn
                        "sf_layer_in": sf_layer_in[name].tolist(),
                        # unused by xfdnn
                        "sf_layer_out": sf_layer_out[name].tolist(),
                        "prescale_shift": prescale_shift[name].astype(int).tolist(),
                        "scale": scale[name].astype(int).tolist(),
                        "postscale_shift": postscale_shift[name].astype(int).tolist()
                    }
                )
            # json.dump(json_payload, g, indent=4, sort_keys=True)
        self.quant_params.save(fname)

    def rebuild_from_scratch(
        self,
        xgraph,
        quant_params: QuantParams,
        quantizecfg: str,
        bitwidth: int = 8,
    ) -> QuantParams:
        """
        Rebuild a quant_params object from scratch. This might happen
        when the objects thresholds have changed
        """

        quant_layers = []

        # TODO implement as a graph pass??
        for X in xgraph.get_layers():
            P = X

            bottom_Ps = xgraph.get_bottom_layers(P.name)
            top_Ps = xgraph.get_top_layers(P.name)

            if P.name in quant_params:
                if ('Convolution' in P.type or 'Pooling' in P.type
                    or 'Eltwise' in P.type or 'Scale' in P.type
                    or 'Concat' in P.type):

                    assert(len(P.type) == 1)

                    if 'Scale' in P.type:
                        assert(isinstance(P.data, xlayer.ScaleData))
                        gamma = P.data.gamma
                        quant_layers.append((P.name, P.type[0], [LayerParams(gamma)]))
                    elif 'Pooling' in P.type:
                        # P.pool == 0 => max pooling
                        pool_divisor = [1] if P.attrs['pool_type'] == 'Max' else [np.prod(P.attrs['kernel_size'])]
                        quant_layers.append((P.name, P.type[0], LayerParams(pool_divisor)))
                    else:
                        quant_layers.append((P.name, P.type[0], None))

                    th_in = quant_params[P.name]['th_layer_in']
                    th_out = quant_params[P.name]['th_layer_out']

                    self.bw_layer_in[P.name] = bitwidth
                    self.th_layer_in[P.name] = np.array([th_in])
                    self.bw_layer_out[P.name] = bitwidth
                    self.th_layer_out[P.name] = np.array([th_out])

                    if 'th_params' in quant_params[P.name]:
                        th_params = quant_params[P.name]['th_params']
                        self.bw_params[P.name] = bitwidth
                        self.th_params[P.name] = np.array(th_params)
                else:
                    raise NotImplementedError(
                        f"Operation: {P.name} of type: {P.type[0]} not supported in quantization "
                        "parameter rebuilding functionaility"
                    )

        self.save_to_dpu_v1_json(quant_layers, quantizecfg)
