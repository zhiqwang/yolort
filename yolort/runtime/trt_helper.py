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

import numpy as np
import onnx
from onnx import shape_inference

try:
    import onnx_graphsurgeon as gs
except ImportError:
    gs = None

from .yolo_tensorrt_model import YOLOTRTModule

logging.basicConfig(level=logging.INFO)
logging.getLogger("YOLOv5GraphSurgeon").setLevel(logging.INFO)
log = logging.getLogger("YOLOv5GraphSurgeon")


class YOLOv5GraphSurgeon:
    def __init__(
        self,
        checkpoint_path: str,
        score_thresh: float = 0.25,
        version: str = "r6.0",
    ):
        """
        Constructor of the YOLOv5 Graph Surgeon object, to do the conversion
        of a YOLOv5 saved onnx model to an ONNX-TensorRT parsable model.

        Args:
            checkpoint_path: The path pointing to the PyTorch saved model to load.
            score_thresh (float): Score threshold used for postprocessing the detections.
            version (str): upstream version released by the ultralytics/yolov5, Possible
                values are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
        """
        checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists()

        # Use YOLOTRTModule to convert saved model to an initial ONNX graph.
        model = YOLOTRTModule(checkpoint_path, score_thresh=score_thresh, version=version)
        model = model.eval()

        log.info(f"Loaded saved model from {checkpoint_path}")
        onnx_model_path = checkpoint_path.with_suffix(".onnx")
        model.to_onnx(onnx_model_path)
        self.graph = gs.import_onnx(onnx.load(onnx_model_path))
        assert self.graph
        log.info("PyTorch2ONNX graph created successfully")

        # Fold constants via ONNX-GS that PyTorch2ONNX may have missed
        self.graph.fold_constants()

        self.batch_size = None

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
                log.info(f"Shape inference could not be performed at this time:\n{e}")
            try:
                self.graph.fold_constants(fold_shapes=True)
            except TypeError as e:
                log.error(
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
        log.info(f"Saved ONNX model to {output_path}")

    def update_preprocessor(self, input_shape):
        """
        Remove all the pre-processing nodes in the ONNX graph and leave only
        the image normalization essentials.

        Args:
            input_shape: The input tensor shape to use for the ONNX graph.
        """
        # Update the input and output tensors shape
        input_shape = input_shape.split(",")
        assert len(input_shape) == 4
        for i in range(len(input_shape)):
            input_shape[i] = int(input_shape[i])
            assert input_shape[i] >= 1
        input_format = "NCHW"

        self.batch_size = input_shape[0]
        self.graph.inputs[0].shape = input_shape
        self.graph.inputs[0].dtype = np.float32

        self.infer()
        log.info(f"ONNX graph input shape: {self.graph.inputs[0].shape} [{input_format} format detected]")

        # Find the initial nodes of the graph, whatever the input
        # is first connected to, and disconnect them
        for node in [node for node in self.graph.nodes if self.graph.inputs[0] in node.inputs]:
            node.inputs.clear()

        # Reshape nodes tend to update the batch dimension to a fixed value of 1,
        # they should use the batch size instead
        for node in [node for node in self.graph.nodes if node.op == "Reshape"]:
            if type(node.inputs[1]) == gs.Constant and node.inputs[1].values[0] == 1:
                node.inputs[1].values[0] = self.batch_size

        self.infer()

    def update_nms(self, threshold=None, detections=None):
        """
        Updates the graph to replace the NMS op by BatchedNMS_TRT TensorRT plugin node.

        Args:
            threshold: Override the score threshold attribute. If set to None,
                use the value in the graph.
            detections: Override the max detections attribute. If set to None,
                use the value in the graph.
        """

        self.infer()

        head_names = []

        # There are five nodes at the bottom of the graph that provide important connection points:

        # 1. Find the concat node at the end of the class net (multi-scale class predictor)
        class_net = self.find_head_concat(head_names[0])
        class_net_tensor = class_net.outputs[0]

        # 2. Find the concat node at the end of the box net (multi-scale localization predictor)
        box_net = self.find_head_concat(head_names[1])
        box_net_tensor = box_net.outputs[0]

        # 3. Find the split node that separates the box net coordinates and
        # feeds them into the box decoder.
        box_net_split = self.graph.find_descendant_by_op(box_net, "Split")
        assert box_net_split and len(box_net_split.outputs) == 4

        # 4. Find the concat node at the end of the box decoder.
        box_decoder = self.graph.find_descendant_by_op(box_net_split, "Concat")
        assert box_decoder and len(box_decoder.inputs) == 4

        # 5. Find the NMS node.
        nms_node = self.graph.find_node_by_op("NonMaxSuppression")

        # Extract NMS Configuration
        num_detections = int(nms_node.inputs[2].values) if detections is None else detections
        iou_threshold = float(nms_node.inputs[3].values)
        score_thresh = float(nms_node.inputs[4].values) if threshold is None else threshold

        # NMS Inputs and Attributes
        # NMS expects these shapes for its input tensors:
        # box_net: [batch_size, number_boxes, 4]
        # class_net: [batch_size, number_boxes, number_labels]
        # anchors: [1, number_boxes, 4] (if used)
        nms_op = None
        nms_attrs = None
        nms_inputs = None

        # EfficientNMS TensorRT Plugin
        # Fusing the decoder will always be faster, so this is the default NMS method supported.
        # In this case, three inputs are given to the NMS TensorRT node:
        # - The box predictions (from the Box Net node found above)
        # - The class predictions (from the Class Net node found above)
        # As the original tensors from YOLOv5 will be used, the NMS code type is set to 0 (Corners),
        # because this is the internal box coding format used by the network.

        nms_inputs = [box_net_tensor, class_net_tensor]
        nms_op = "EfficientNMS_TRT"
        nms_attrs = {
            "plugin_version": "1",
            "background_class": -1,
            "max_output_boxes": num_detections,
            "score_thresh": max(0.01, score_thresh),
            "iou_threshold": iou_threshold,
            "score_activation": True,
            "box_coding": 0,
        }
        nms_output_labels_dtype = np.int32

        # NMS Outputs
        nms_output_num_detections = gs.Variable(
            name="num_detections", dtype=np.int32, shape=[self.batch_size, 1]
        )
        nms_output_boxes = gs.Variable(
            name="detection_boxes",
            dtype=np.float32,
            shape=[self.batch_size, num_detections, 4],
        )
        nms_output_scores = gs.Variable(
            name="detection_scores",
            dtype=np.float32,
            shape=[self.batch_size, num_detections],
        )
        nms_output_labels = gs.Variable(
            name="detection_labels",
            dtype=nms_output_labels_dtype,
            shape=[self.batch_size, num_detections],
        )

        nms_outputs = [nms_output_num_detections, nms_output_boxes, nms_output_scores, nms_output_labels]

        # Create the NMS Plugin node with the selected inputs. The outputs of the node will also
        # become the final outputs of the graph.
        self.graph.plugin(
            op=nms_op,
            name="nms/non_maximum_suppression",
            inputs=nms_inputs,
            outputs=nms_outputs,
            attrs=nms_attrs,
        )
        log.info(f"Created NMS plugin '{nms_op}' with attributes: {nms_attrs}")

        self.graph.outputs = nms_outputs

        self.infer()

    def find_head_concat(self, name_scope):
        # This will find the concatenation node at the end of either Class Net or Box Net.
        # These concatenation nodes bring together prediction data for each of 5 scales.
        # The concatenated Class Net node will have shape [batch_size, num_anchors, num_labels],
        # and the concatenated Box Net node has the shape [batch_size, num_anchors, 4].
        # These concatenation nodes can be be found by searching for all Concat's and checking
        # if the node two steps above in the graph has a name that begins with either
        # "box_net/..." or "class_net/...".
        for node in [node for node in self.graph.nodes if node.op == "Transpose" and name_scope in node.name]:
            concat = self.graph.find_descendant_by_op(node, "Concat")
            assert concat and len(concat.inputs) == 5
            log.info(f"Found {concat.op} node '{concat.name}' as the tip of {name_scope}")
            return concat
