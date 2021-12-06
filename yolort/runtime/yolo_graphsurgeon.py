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
    """
    Constructor of the YOLOv5 Graph Surgeon object, TensorRT treat ``nms`` as
    plugin, especially ``EfficientNMS_TRT`` in our yolort PostProcess module.

    Args:
        checkpoint_path: The path pointing to the PyTorch saved model to load.
        score_thresh (float): Score threshold used for postprocessing the detections.
        version (str): upstream version released by the ultralytics/yolov5, Possible
            values are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
        enable_dynamic: Whether to specify axes of tensors as dynamic. Default: True.
    """

    def __init__(
        self,
        checkpoint_path: str,
        score_thresh: float = 0.25,
        version: str = "r6.0",
        enable_dynamic: bool = True,
    ):
        checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists()

        # Use YOLOTRTModule to convert saved model to an initial ONNX graph.
        model = YOLOTRTModule(checkpoint_path, score_thresh=score_thresh, version=version)
        model = model.eval()

        log.info(f"Loaded saved model from {checkpoint_path}")
        onnx_model_path = checkpoint_path.with_suffix(".onnx")
        model.to_onnx(onnx_model_path, enable_dynamic=enable_dynamic)
        self.graph = gs.import_onnx(onnx.load(onnx_model_path))
        assert self.graph
        log.info("PyTorch2ONNX graph created successfully")

        # Fold constants via ONNX-GS that PyTorch2ONNX may have missed
        self.graph.fold_constants()

        self.batch_size = 1

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

    def register_nms(
        self,
        score_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        detections_per_img: int = 100,
    ):
        """
        Register the ``EfficientNMS_TRT`` plugin node.

        NMS expects these shapes for its input tensors:
        - box_net: [batch_size, number_boxes, 4]
        - class_net: [batch_size, number_boxes, number_labels]

        EfficientNMS TensorRT Plugin
        Fusing the decoder will always be faster, so this is the default NMS method supported.
        In this case, three inputs are given to the NMS TensorRT node:
        - The box predictions (from the Box Net node found above)
        - The class predictions (from the Class Net node found above)

        As the original tensors from YOLOv5 will be used, the NMS code type is set to 0 (Corners),
        because this is the internal box coding format used by the network.

        Args:
            threshold: Override the score threshold attribute. If set to None,
                use the value in the graph.
            detections: Override the max detections attribute. If set to None,
                use the value in the graph.
        """

        self.infer()
        # Find the concat node at the end of the network
        nms_inputs = self.graph.outputs
        op = "EfficientNMS_TRT"
        attrs = {
            "plugin_version": "1",
            "background_class": -1,  # no background class
            "max_output_boxes": detections_per_img,
            "score_threshold": max(0.01, score_thresh),
            "iou_threshold": nms_thresh,
            "score_activation": True,
            "box_coding": 0,
        }

        # NMS Outputs
        output_num_detections = gs.Variable(
            name="num_detections",
            dtype=np.int32,
            shape=[self.batch_size, 1],
        )  # A scalar indicating the number of valid detections per batch image.
        output_boxes = gs.Variable(
            name="detection_boxes",
            dtype=np.float32,
            shape=[self.batch_size, detections_per_img, 4],
        )
        output_scores = gs.Variable(
            name="detection_scores",
            dtype=np.float32,
            shape=[self.batch_size, detections_per_img],
        )
        output_labels = gs.Variable(
            name="detection_labels",
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
        log.info(f"Created NMS plugin '{op}' with attributes: {attrs}")

        self.graph.outputs = nms_outputs

        self.infer()
