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
from typing import Optional, List, Tuple, Union

try:
    import tensorrt as trt
except ImportError:
    trt = None

import torch
from torch import nn, Tensor
from yolort.models import YOLO
from yolort.models.anchor_utils import AnchorGenerator
from yolort.models.backbone_utils import darknet_pan_backbone
from yolort.models.box_head import _concat_pred_logits, _decode_pred_logits
from yolort.utils import load_from_ultralytics

logging.basicConfig(level=logging.INFO)
logging.getLogger("TRTHelper").setLevel(logging.INFO)
logger = logging.getLogger("TRTHelper")


__all__ = ["YOLOTRTModule", "EngineBuilder"]


class LogitsDecoder(nn.Module):
    """
    This is a simplified version of post-processing module, we manually remove
    the ``torchvision::ops::nms``, and it will be used later in the procedure for
    exporting the ONNX Graph to YOLOTRTModule or others.
    """

    def __init__(self, strides: List[int]) -> None:
        """
        Args:
            strides (List[int]): Strides of the AnchorGenerator.
        """

        super().__init__()
        self.strides = strides

    def forward(
        self,
        head_outputs: List[Tensor],
        grids: List[Tensor],
        shifts: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Just concat the predict logits, ignore the original ``torchvision::nms`` module
        from original ``yolort.models.box_head.PostProcess``.

        Args:
            head_outputs (List[Tensor]): The predicted locations and class/object confidence,
                shape of the element is (N, A, H, W, K).
            grids (List[Tensor]): Anchor grids.
            shifts (List[Tensor]): Anchor shifts.
        """
        batch_size = len(head_outputs[0])
        device = head_outputs[0].device
        dtype = head_outputs[0].dtype
        strides = torch.as_tensor(self.strides, dtype=torch.float32, device=device).to(dtype=dtype)

        all_pred_logits = _concat_pred_logits(head_outputs, grids, shifts, strides)

        bbox_regression = []
        pred_scores = []

        for idx in range(batch_size):  # image idx, image inference
            pred_logits = all_pred_logits[idx]
            boxes, scores = _decode_pred_logits(pred_logits)
            bbox_regression.append(boxes)
            pred_scores.append(scores)

        # The default boxes tensor has shape [batch_size, number_boxes, 4].
        boxes = torch.stack(bbox_regression)
        scores = torch.stack(pred_scores)
        return boxes, scores


class YOLOTRTModule(nn.Module):
    """
    TensorRT deployment friendly wrapper for YOLO.

    Remove the ``torchvision::nms`` in this warpper, due to the fact that some third-party
    inference frameworks currently do not support this operator very well.
    """

    def __init__(
        self,
        checkpoint_path: str,
        version: str = "r6.0",
    ):
        super().__init__()
        model_info = load_from_ultralytics(checkpoint_path, version=version)

        backbone_name = f"darknet_{model_info['size']}_{version.replace('.', '_')}"
        depth_multiple = model_info["depth_multiple"]
        width_multiple = model_info["width_multiple"]
        use_p6 = model_info["use_p6"]
        backbone = darknet_pan_backbone(
            backbone_name,
            depth_multiple,
            width_multiple,
            version=version,
            use_p6=use_p6,
        )
        num_classes = model_info["num_classes"]
        anchor_generator = AnchorGenerator(model_info["strides"], model_info["anchor_grids"])
        post_process = LogitsDecoder(model_info["strides"])
        model = YOLO(
            backbone,
            num_classes,
            anchor_generator=anchor_generator,
            post_process=post_process,
        )

        model.load_state_dict(model_info["state_dict"])
        self.model = model
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs (Tensor): batched images, of shape [batch_size x 3 x H x W]
        """
        # Compute the detections
        outputs = self.model(inputs)

        return outputs

    @torch.no_grad()
    def to_onnx(
        self,
        file_path: Union[str, Path],
        input_sample: Optional[Tensor] = None,
        opset_version: int = 11,
        enable_dynamic: bool = True,
        **kwargs,
    ):
        """
        Saves the model in ONNX format.

        Args:
            file_path: The path of the file the onnx model should be saved to.
            input_sample: An input for tracing. Default: None.
            opset_version: Opset version we export the model to the onnx submodule. Default: 11.
            enable_dynamic: Whether to specify axes of tensors as dynamic. Default: True.
            **kwargs: Will be passed to torch.onnx.export function.
        """
        if input_sample is None:
            input_sample = torch.rand(1, 3, 640, 640).to(next(self.parameters()).device)

        dynamic_axes = (
            {
                "images": {0: "batch", 2: "height", 3: "width"},
                "boxes": {0: "batch", 1: "num_objects"},
                "scores": {0: "batch", 1: "num_objects"},
            }
            if enable_dynamic
            else None
        )

        input_names = ["images"]
        output_names = ["boxes", "scores"]

        torch.onnx.export(
            self.model,
            input_sample,
            file_path,
            do_constant_folding=True,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            **kwargs,
        )


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

        logger.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            logger.info(f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}")
        for output in outputs:
            logger.info(f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}")

    def create_engine(
        self,
        engine_path: str,
        *,
        precision: str = "fp32",
        max_batch_size: int = 32,
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
