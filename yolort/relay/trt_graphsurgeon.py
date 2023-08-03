# Copyright (c) 2021, yolort team. All rights reserved.

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import torch
from onnx import shape_inference
from torch import Tensor
from yolort.utils import is_module_available, requires_module

if is_module_available("onnx_graphsurgeon"):
    import onnx_graphsurgeon as gs

if is_module_available("onnxsim"):
    import onnxsim

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
    https://github.com/zhiqwang/yolort/blob/ba00833/yolort/models/box_head.py#L410-L418

    Args:
        model_path (string): The path pointing to the PyTorch saved model to load.
        version (str): upstream version released by the ultralytics/yolov5, Possible
            values are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
        input_sample (Tensor, optional): Specify the input shape to export ONNX, and the
            default shape for the sample is (1, 3, 640, 640).
        score_thresh (float): Score threshold used for postprocessing the detections.
        enable_dynamic (bool): Whether to specify axes of tensors as dynamic. Default: False.
        device (torch.device): The device to be used for importing ONNX. Default: torch.device("cpu").
        precision (string): The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        simplify (bool, optional): Whether to simplify the exported ONNX. Default to False.
    """

    @requires_module("onnx_graphsurgeon")
    def __init__(
        self,
        model_path: str,
        *,
        version: str = "r6.0",
        input_sample: Optional[Tensor] = None,
        enable_dynamic: bool = False,
        device: torch.device = torch.device("cpu"),
        precision: str = "fp32",
        simplify: bool = False,
    ):
        model_path = Path(model_path)
        self.suffix = model_path.suffix
        assert model_path.exists() and self.suffix in (".onnx", ".pt", ".pth")

        # Use YOLOTRTInference to convert saved model to an initial ONNX graph.
        if model_path.suffix in (".pt", ".pth"):
            model = YOLOTRTInference(model_path, version=version)
            model = model.eval()
            model = model.to(device=device)
            self.num_classes = model.num_classes
            logger.info(f"Loaded saved model from {model_path}")

            if input_sample is not None:
                input_sample = input_sample.to(device=device)
            model_path = model_path.with_suffix(".onnx")
            model.to_onnx(model_path, input_sample=input_sample, enable_dynamic=enable_dynamic)
            logger.info("PyTorch2ONNX graph created successfully")
        # Use YOLOTRTInference to modify an existed ONNX graph.
        self.graph = gs.import_onnx(onnx.load(model_path))
        assert self.graph

        # Fold constants via ONNX-GS that PyTorch2ONNX may have missed
        self.graph.fold_constants()
        self.batch_size = 1
        self.precision = precision
        self.simplify = simplify

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
                        if o in self.graph.outputs:
                            continue
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

    def _process(self, dtype):
        Matrix = np.array(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [-0.5, 0.0, 0.5, 0.0], [0.0, -0.5, 0.0, 0.5]],
            dtype=dtype,
        )
        Attrs = [[5, 9999, 2, 1], [4, 5, 2, 1], [0, 4, 2, 1]]
        lastNode = [
            node for node in self.graph.nodes if node.outputs and node.outputs[0] in self.graph.outputs
        ][0]
        out_shape = lastNode.outputs[0].shape
        mul_inputs = []
        matmul_inputs = [None, gs.Constant(name="AddMatMul", values=Matrix)]
        for i, attr in enumerate(Attrs):
            Slice_inp = [lastNode.outputs[0]] + [
                gs.Constant(name=f"AddSlice_{i}_inp_{j}", values=np.array([val]))
                for j, val in enumerate(attr)
            ]
            Slice_out = gs.Variable(name=f"AddSlice_{i}_out")
            Slice = gs.Node(name=f"AddSlice_{i}", op="Slice", inputs=Slice_inp, outputs=[Slice_out])
            self.graph.nodes.append(Slice)
            if i < 2:
                mul_inputs.append(Slice_out)
            elif i == 2:
                matmul_inputs[0] = Slice_out
        mut_output = gs.Variable(name="NMS_Scores", shape=out_shape[:2] + [out_shape[2] - 5], dtype=dtype)
        matmut_output = gs.Variable(name="NMS_Boxes", shape=out_shape[:2] + [4], dtype=dtype)
        self.graph.layer(name="AddMul_0", op="Mul", inputs=mul_inputs, outputs=[mut_output])
        self.graph.layer(name="AddMatMul_0", op="MatMul", inputs=matmul_inputs, outputs=[matmut_output])
        self.graph.outputs = [matmut_output, mut_output]

    @requires_module("onnxsim")
    def save(self, output_path):
        """
        Save the ONNX model to the given location.

        Args:
            output_path: Path pointing to the location where to write
                out the updated ONNX model.
        """
        self.graph.cleanup().toposort()
        model = gs.export_onnx(self.graph)
        if self.simplify:
            try:
                model, check = onnxsim.simplify(model)
                assert check, "assert check failed, save origin onnx"
            except Exception as e:
                logger.info(f"Simplifier failure: {e}")
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

        if self.precision == "fp32":
            dtype_output = np.float32
        elif self.precision == "fp16":
            dtype_output = np.float16
        else:
            raise NotImplementedError(f"Currently not supports precision: {self.precision}")

        if self.suffix == ".onnx":
            self._process(dtype_output)

        op = "EfficientNMS_TRT"
        attrs = OrderedDict(
            plugin_version="1",
            background_class=-1,  # no background class
            max_output_boxes=detections_per_img,
            score_threshold=score_thresh,
            iou_threshold=nms_thresh,
            score_activation=False,
            box_coding=0,
        )

        op_outputs = [
            gs.Variable(
                name="num_detections",
                dtype=np.int32,
                shape=[self.batch_size, 1],
            ),
            gs.Variable(
                name="detection_boxes",
                dtype=dtype_output,
                shape=[self.batch_size, detections_per_img, 4],
            ),
            gs.Variable(
                name="detection_scores",
                dtype=dtype_output,
                shape=[self.batch_size, detections_per_img],
            ),
            gs.Variable(
                name="detection_classes",
                dtype=np.int32,
                shape=[self.batch_size, detections_per_img],
            ),
        ]

        # Create the NMS Plugin node with the selected inputs. The outputs of the node will also
        # become the final outputs of the graph.
        self.graph.layer(
            op=op, name="batched_nms", inputs=self.graph.outputs, outputs=op_outputs, attrs=attrs
        )
        logger.info(f"Created NMS plugin '{op}' with attributes: {attrs}")

        self.graph.outputs = op_outputs

        self.infer()
