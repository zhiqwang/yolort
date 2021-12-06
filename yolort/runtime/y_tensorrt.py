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

import numpy as np

try:
    import pycuda.driver as cuda
    import tensorrt as trt
except ImportError:
    trt, cuda = None, None

from .image_batcher import ImageBatcher


class PredictorTRT:
    """
    Implements inference for the EfficientDet TensorRT engine.

    Examples:
        >>> from yolort.runtime import PredictorTRT
        >>>
        >>> checkpoint_path = "yolort.onnx"
        >>> detector = PredictorTRT(checkpoint_path)
        >>>
        >>> img_path = "bus.jpg"
        >>> scores, class_ids, boxes = detector.run(img_path)
    """

    def __init__(self, engine_path: str):
        """
        Args:
            engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def _input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare
        memory allocations.

        Return:
            Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def _output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare
        memory allocations.

        Return:
            A list with two items per element, the shape and (numpy) datatype
                of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o["shape"], o["dtype"]))
        return specs

    def infer(self, batch, scales=None, nms_threshold=None):
        """
        Execute inference on a batch of images. The images should already be batched and
        preprocessed, as prepared by the ImageBatcher class. Memory copying to and from
        the GPU device will be performed here.

        Args:
            batch: A numpy array holding the image batch.
            scales: The image resize scales for each image in this batch.
                Default: No scale postprocessing applied.

        Return:
            A nested list for each image in the batch and each detection in the list.
        """
        # Prepare the output data
        outputs = []
        for shape, dtype in self._output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]["allocation"], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]["allocation"])

        # Process the results
        nums = outputs[0]
        boxes = outputs[1]
        scores = outputs[2]
        classes = outputs[3]
        detections = []
        normalized = np.max(boxes) < 2.0
        for i in range(self.batch_size):
            detections.append([])
            for n in range(int(nums[i])):
                scale = self.inputs[0]["shape"][2] if normalized else 1.0
                if scales and i < len(scales):
                    scale /= scales[i]
                if nms_threshold and scores[i][n] < nms_threshold:
                    continue
                detections[i].append(
                    {
                        "ymin": boxes[i][n][0] * scale,
                        "xmin": boxes[i][n][1] * scale,
                        "ymax": boxes[i][n][2] * scale,
                        "xmax": boxes[i][n][3] * scale,
                        "score": scores[i][n],
                        "class": int(classes[i][n]),
                    }
                )
        return detections

    def run(self, input, nms_threshold: float = 0.45):
        batcher = ImageBatcher(input, *self._input_spec())
        for batch, images, scales in batcher.get_batch():
            print(f"Processing Image {batcher.image_index} / {batcher.num_images}", end="\r")
            detections = self.infer(batch, scales, nms_threshold)
            for i in range(len(images)):
                # Text Results
                output_results = ""
                for d in detections[i]:
                    line = [d["xmin"], d["ymin"], d["xmax"], d["ymax"], d["score"], d["class"]]
                    output_results += "\t".join([str(f) for f in line]) + "\n"
