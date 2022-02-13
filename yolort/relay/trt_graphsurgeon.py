# Copyright (c) 2021, yolort team. All rights reserved.

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import torch
from onnx import shape_inference
from torch import Tensor

try:
    import onnx_graphsurgeon as gs
except ImportError:
    gs = None

from .trt_inference import YOLOTRTInference

logging.basicConfig(level=logging.INFO)
logging.getLogger("YOLOTRTGraphSurgeon").setLevel(logging.INFO)
logger = logging.getLogger("YOLOTRTGraphSurgeon")

__all__ = ["YOLOTRTGraphSurgeon"]


class YOLOTRTGraphSurgeon:
    """
    YOLOv5 Graph Surgeon for TensorRT inference.

    Because TensorRT treat the ``torchvision::ops::nms`` as plugin, we use the a simple post-processing
    module named ``LogitsDecoder`` to connect to ``EfficientNMS_TRT`` plugin in TensorRT.

    And the ``EfficientNMS_TRT`` plays the same role of following computation.
    https://github.com/zhiqwang/yolov5-rt-stack/blob/ba00833/yolort/models/box_head.py#L410-L418

    Args:
        checkpoint_path (string): The path pointing to the PyTorch saved model to load.
        version (str): upstream version released by the ultralytics/yolov5, Possible
            values are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
        input_sample (Tensor, optional): Specify the input shape to export ONNX, and the
            default shape for the sample is (1, 3, 640, 640).
        score_thresh (float): Score threshold used for postprocessing the detections.
        enable_dynamic (bool): Whether to specify axes of tensors as dynamic. Default: False.
        device (torch.device): The device to be used for importing ONNX. Default: torch.device("cpu").
        precision (string): The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        version: str = "r6.0",
        input_sample: Optional[Tensor] = None,
        enable_dynamic: bool = False,
        device: torch.device = torch.device("cpu"),
        precision: str = "fp32",
    ):
        checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists()

        # Use YOLOTRTInference to convert saved model to an initial ONNX graph.
        model = YOLOTRTInference(checkpoint_path, version=version)
        model = model.eval()
        model = model.to(device=device)
        logger.info(f"Loaded saved model from {checkpoint_path}")

        onnx_model_path = checkpoint_path.with_suffix(".onnx")
        if input_sample is not None:
            input_sample = input_sample.to(device=device)
        model.to_onnx(onnx_model_path, input_sample=input_sample, enable_dynamic=enable_dynamic)
        self.graph = gs.import_onnx(onnx.load(onnx_model_path))
        assert self.graph
        logger.info("PyTorch2ONNX graph created successfully")

        # Fold constants via ONNX-GS that PyTorch2ONNX may have missed
        self.graph.fold_constants()
        self.num_classes = model.num_classes
        self.batch_size = 1
        self.precision = precision

    def infer(self):
        """
        Sanitize the graph by cleaning any unconnected nodes, do a topological resort,
        and fold constant inputs values. When possible, run shape inference on the
        ONNX graph to determine tensor shapes.
        """
        for _ in range(3):
            count_before = len(self.graph.nodes)

            self.graph.cleanup().toposort()
            try:
                for node in self.graph.nodes:
                    for o in node.outputs:
                        o.shape = None
                model = gs.export_onnx(self.graph)
                model = shape_inference.infer_shapes(model)
                self.graph = gs.import_onnx(model)
            except Exception as e:
                logger.info(f"Shape inference could not be performed at this time:\n{e}")
            try:
                self.graph.fold_constants(fold_shapes=True)
            except TypeError as e:
                logger.error(
                    "This version of ONNX GraphSurgeon does not support folding shapes, "
                    f"please upgrade your onnx_graphsurgeon module. Error:\n{e}"
                )
                raise

            count_after = len(self.graph.nodes)
            if count_before == count_after:
                # No new folding occurred in this iteration, so we can stop for now.
                break

    def save(self, output_path):
        """
        Save the ONNX model to the given location.

        Args:
            output_path: Path pointing to the location where to write
                out the updated ONNX model.
        """
        self.graph.cleanup().toposort()
        model = gs.export_onnx(self.graph)
        onnx.save(model, output_path)
        logger.info(f"Saved ONNX model to {output_path}")

    def register_nms(
        self,
        *,
        score_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        detections_per_img: int = 100,
    ):
        """
        Register the ``EfficientNMS_TRT`` plugin node.

        NMS expects these shapes for its input tensors:
            - box_net: [batch_size, number_boxes, 4]
            - class_net: [batch_size, number_boxes, number_labels]

        Args:
            score_thresh (float): The scalar threshold for score (low scoring boxes are removed).
            nms_thresh (float): The scalar threshold for IOU (new boxes that have high IOU
                overlap with previously selected boxes are removed).
            detections_per_img (int): Number of best detections to keep after NMS.
        """

        self.infer()
        # Find the concat node at the end of the network
        nms_inputs = self.graph.outputs

        op = "EfficientNMS_TRT"
        attrs = {
            "plugin_version": "1",
            "background_class": -1,  # no background class
            "max_output_boxes": detections_per_img,
            "score_threshold": score_thresh,
            "iou_threshold": nms_thresh,
            "score_activation": False,
            "box_coding": 0,
        }

        if self.precision == "fp32":
            dtype_output = np.float32
        elif self.precision == "fp16":
            dtype_output = np.float16
        else:
            raise NotImplementedError(f"Currently not supports precision: {self.precision}")

        # NMS Outputs
        output_num_detections = gs.Variable(
            name="num_detections",
            dtype=np.int32,
            shape=[self.batch_size, 1],
        )  # A scalar indicating the number of valid detections per batch image.
        output_boxes = gs.Variable(
            name="detection_boxes",
            dtype=dtype_output,
            shape=[self.batch_size, detections_per_img, 4],
        )
        output_scores = gs.Variable(
            name="detection_scores",
            dtype=dtype_output,
            shape=[self.batch_size, detections_per_img],
        )
        output_labels = gs.Variable(
            name="detection_classes",
            dtype=np.int32,
            shape=[self.batch_size, detections_per_img],
        )

        nms_outputs = [output_num_detections, output_boxes, output_scores, output_labels]

        # Create the NMS Plugin node with the selected inputs. The outputs of the node will also
        # become the final outputs of the graph.
        self.graph.layer(
            op=op,
            name="batched_nms",
            inputs=nms_inputs,
            outputs=nms_outputs,
            attrs=attrs,
        )
        logger.info(f"Created NMS plugin '{op}' with attributes: {attrs}")

        self.graph.outputs = nms_outputs

        self.infer()
