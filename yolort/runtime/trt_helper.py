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

import torch
from torch import Tensor

try:
    import tensorrt as trt
except ImportError:
    trt = None

from yolort.relay.trt_graphsurgeon import YOLOTRTGraphSurgeon

logging.basicConfig(level=logging.INFO)
logging.getLogger("TRTHelper").setLevel(logging.INFO)
logger = logging.getLogger("TRTHelper")


def export_tensorrt_engine(
    checkpoint_path,
    *,
    score_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    version: str = "r6.0",
    onnx_path: Optional[str] = None,
    engine_path: Optional[str] = None,
    input_sample: Optional[Tensor] = None,
    detections_per_img: int = 100,
    precision: str = "fp32",
    verbose: bool = False,
    workspace: int = 12,
) -> None:
    """
    Export ONNX models and TensorRT serialized engines that can be used for TensorRT inferencing.

    Args:
        checkpoint_path (str): Path of the YOLOv5 checkpoint model.
        score_thresh (float): Score threshold used for postprocessing the detections. Default: 0.25
        nms_thresh (float): NMS threshold used for postprocessing the detections. Default: 0.45
        version (str): upstream version released by the ultralytics/yolov5, Possible
            values are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
        onnx_path (string, optional): The path to the ONNX graph to load. Default: None
        engine_path (string, optional): The path where to serialize the engine to. Default: None
        input_sample (Tensor, optional): Specify the input shape to export ONNX, and the
            default shape for the sample is (1, 3, 640, 640).
        detections_per_img (int): Number of best detections to keep after NMS. Default: 100
        precision (string): The datatype to use for the engine inference, either 'fp32', 'fp16' or
            'int8'. Default: 'fp32'
        verbose (bool): If enabled, a higher verbosity level will be set on the TensorRT
            logger. Default: False
        workspace (int): Max memory workspace to allow, in Gb. Default: 12
    """

    if input_sample is None:
        input_sample = torch.rand(1, 3, 640, 640)

    yolo_gs = YOLOTRTGraphSurgeon(checkpoint_path, version=version, input_sample=input_sample)

    # Register the `EfficientNMS_TRT` into the graph.
    yolo_gs.register_nms(
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
    )

    # Set the path of ONNX and Tensorrt Engine to export
    checkpoint_path = Path(checkpoint_path)
    onnx_path = onnx_path or str(checkpoint_path.with_suffix(".onnx"))
    engine_path = engine_path or str(checkpoint_path.with_suffix(".engine"))

    # Save the exported ONNX models.
    yolo_gs.save(onnx_path)

    # Build and export the TensorRT engine.
    engine_builder = EngineBuilder(verbose=verbose, workspace=workspace, precision=precision)
    engine_builder.create_network(onnx_path)
    engine_builder.create_engine(engine_path)


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(
        self,
        verbose: bool = False,
        workspace: int = 4,
        precision: str = "fp32",
        enable_dynamic: bool = False,
        max_batch_size: int = 16,
        calib_input: Optional[str] = None,
        calib_cache: Optional[str] = None,
        calib_num_images: int = 5000,
        calib_batch_size: int = 8,
    ):
        """
        Args:
            verbose (bool): If enabled, a higher verbosity level will be set on the TensorRT
                logger. Default: False
            workspace (int): Max memory workspace to allow, in Gb. Default: 4
            precision (string): The datatype to use for the engine inference, either 'fp32',
                'fp16' or 'int8'. Default: 'fp32'
            enable_dynamic (bool): Whether to enable dynamic shapes. Default: False
            max_batch_size (int): Maximum batch size reserved for dynamic shape inference.
                Default: 16
            calib_input (string, optinal): The path to a directory holding the calibration images.
                Default: None
            calib_cache (string, optinal): The path where to write the calibration cache to,
                or if it already exists, load it from. Default: None
            calib_num_images (int): The maximum number of images to use for calibration. Default: 5000
            calib_batch_size (int): The batch size to use for the calibration process. Default: 8
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

        # Leaving some interfaces and parameters for subsequent use, but we have not yet
        # implemented the following functionality
        self.precision = precision
        self.enable_dynamic = enable_dynamic
        self.max_batch_size = max_batch_size
        self.calib_input = calib_input
        self.calib_cache = calib_cache
        self.calib_num_images = calib_num_images
        self.calib_batch_size = calib_batch_size

    def create_network(self, onnx_path: str):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.

        Args:
            onnx_path (string): The path to the ONNX graph to load.
        """

        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(flag)
        self.parser = trt.OnnxParser(self.network, self.logger)
        if not self.parser.parse_from_file(onnx_path):
            raise RuntimeError(f"Failed to load ONNX file: {onnx_path}")

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        logger.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            logger.info(f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}")
        for output in outputs:
            logger.info(f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}")

    def create_engine(self, engine_path: str):
        """
        Build the TensorRT engine and serialize it to disk.

        Args:
            engine_path (string): The path where to serialize the engine to.
        """
        engine_path = Path(engine_path)
        engine_path.parent.mkdir(parents=True, exist_ok=True)

        precision = self.precision
        logger.info(f"Building {precision} Engine in {engine_path}")

        # Process the batch size and profile
        assert self.batch_size > 0, "Currently only supports static shape."
        self.builder.max_batch_size = self.batch_size

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                logger.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "fp32":
            logger.info("Using fp32 mode.")
        else:
            raise NotImplementedError(f"Currently hasn't been implemented: {precision}.")

        with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
            f.write(engine.serialize())
            logger.info(f"Serialize engine success, saved as {engine_path}")
