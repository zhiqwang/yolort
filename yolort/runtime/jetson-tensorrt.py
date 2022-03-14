import os
import subprocess
from collections import OrderedDict, namedtuple
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch


class jetsonTrt:
    def __init__(
        self,
        onnx_path="yolov5n.onnx",
        engine_path="yolov5n.engine",
        imgsz=(1, 3, 320, 320),
        workspace=8192,
        device=torch.device("cuda"),
        fp16=False,
    ):
        self.device = device
        self.fp16 = fp16
        if not Path(engine_path).exists():
            cmd = f"/usr/src/tensorrt/bin/trtexec --onnx={onnx_path} --saveEngine={engine_path} --workspace={workspace}"
            subprocess.check_output(cmd + " --fp16" if fp16 else cmd, shell=True)
        self.engine_path = engine_path
        self.read_engine()
        self.warmup(imgsz)

    def read_engine(self):
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            if model.binding_is_input(index) and dtype == np.float16:
                self.fp16 = True
        self.bindings = bindings
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        self.context = model.create_execution_context()

    def infer(self, im):
        self.binding_addrs["images"] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        num_dets = self.bindings["num_detections"].data
        boxes = self.bindings["detection_boxes"].data
        scores = self.bindings["detection_scores"].data
        labels = self.bindings["detection_classes"].data
        results = self.parse_output(boxes, scores, labels, num_dets)
        return results

    def warmup(self, imgsz=(1, 3, 320, 320)):
        # Warmup model by running inference once and only warmup GPU models
        if isinstance(self.device, torch.device) and self.device.type != "cpu":
            image = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)
            self.infer(image)

    @staticmethod
    def parse_output(all_boxes, all_scores, all_labels, all_num_dets):
        detections = []
        for boxes, scores, labels, num_dets in zip(all_boxes, all_scores, all_labels, all_num_dets):
            keep = num_dets.item()
            boxes, scores, labels = boxes[:keep], scores[:keep], labels[:keep]
            detections.append({"scores": scores, "labels": labels, "boxes": boxes})

        return detections


if __name__ == "__main__":
    cuda_visible = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f"We're using TensorRT: {trt.__version__} on {device} device: {cuda_visible}.")

    onnx_path = "../../yolov5n.onnx"
    engine_path = "../../yolov5n.engine"
    batchsize = 1
    img_size = 320

    trtexport = jetsonTrt(
        onnx_path=onnx_path,
        engine_path=engine_path,
        imgsz=(batchsize, 3, img_size, img_size),
        workspace=8192,
        device=device,
    )

    img_path = "../../test/assets/bus.jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32)
    img /= 255
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device)
    out = trtexport.infer(img_tensor)
    print(out)
