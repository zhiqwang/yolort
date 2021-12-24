# Copyright (c) 2021, yolort team. All rights reserved.
#
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of TensorRT source tree.
#

import logging
from pathlib import Path
from typing import Optional

try:
    import tensorrt as trt
except ImportError:
    trt = None

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False, workspace=4):
        """
        Args:
            verbose: If enabled, a higher verbosity level will be
                set on the TensorRT logger.
            workspace: Max memory workspace to allow, in Gb.
        """
        self.logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.logger, namespace="")

        self.builder = trt.Builder(self.logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = workspace * 1 << 30

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path: str):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.

        Args:
            onnx_path: The path to the ONNX graph to load.
        """

        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(flag)
        self.parser = trt.OnnxParser(self.network, self.logger)
        if not self.parser.parse_from_file(onnx_path):
            raise RuntimeError(f"Failed to load ONNX file: {onnx_path}")

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info(f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}")
        for output in outputs:
            log.info(f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}")
        assert self.batch_size > 0
        self.builder.max_batch_size = self.batch_size

    def create_engine(
        self,
        engine_path: str,
        precision: str = "fp32",
        calib_input: Optional[str] = None,
        calib_cache: Optional[str] = None,
        calib_num_images: int = 5000,
        calib_batch_size: int = 8,
    ):
        """
        Build the TensorRT engine and serialize it to disk.

        Args:
            engine_path: The path where to serialize the engine to.
            precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
            calib_input: The path to a directory holding the calibration images.
            calib_cache: The path where to write the calibration cache to, or if it already
                exists, load it from.
            calib_num_images: The maximum number of images to use for calibration.
            calib_batch_size: The batch size to use for the calibration process.
        """
        engine_path = Path(engine_path)

        engine_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"Building {precision} Engine in {engine_path}")

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "fp32":
            log.info("Using fp32 mode.")
        else:
            raise NotImplementedError(f"Currently hasn't been implemented: {precision}.")

        with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
            f.write(engine.serialize())
            log.info(f"Serialize engine success, saved as {engine_path}")
