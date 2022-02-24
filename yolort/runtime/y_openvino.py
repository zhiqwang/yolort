import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops.boxes import batched_nms

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import openvino as ovo
    from openvino.inference_engine import IECore
except ImportError:
    ovo = None
    IECore = None


logger = logging.getLogger(__name__)


class PredictorOVO:
    def __init__(self, engine_path: str, device: str = "cpu") -> None:
        self.engine_path = Path(engine_path)
        self.device = device
        self.ovo_net = self._build_ovonet()

    def _build_ovonet(self):
        logger.info("Openvino inference engine was initialized.")
        if ovo is not None:
            if self.device != "cpu":
                logger.info("Openvino only support CPU.")
                self.device = "CPU"
            else:
                self.device = self.device.upper()
        else:
            raise ImportError(
                'openvino is not installed, please use command "pip install openvino-dev" firstly.'
            )
        ie = IECore()
        net_ir = ie.read_network(model=self.engine_path)
        exec_net_ir = ie.load_network(network=net_ir, device_name=self.device)
        return exec_net_ir

    def name_loader_simple_resize(self, img_path: str, new_shape=(640, 640)):
        image = cv2.imread(img_path)
        image = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, axes=(2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image

    def array_loader_simple_resize(self, img_array, new_shape=(640, 640)):
        for i, array in enumerate(img_array):
            if array.shape[-2:] != new_shape:
                img_array[i] = cv2.resize(array, new_shape, interpolation=cv2.INTER_LINEAR)
                img_array[i] = cv2.cvtColor(img_array[i], cv2.COLOR_BGR2RGB)
                img_array[i] = np.transpose(img_array[i], axes=(2, 0, 1))
        return np.ascontiguousarray(np.stack(img_array, axis=0).astype(np.float32) / 255.0)

    def __call__(self, inputs, nms_scores=0.45, cls_score=0.2):
        inputs = self.check_input(inputs)
        outputs = self.ovo_net.infer(inputs=inputs)
        outputs = self.nms(outputs, nms_scores, cls_score)
        return outputs

    def check_input(self, inputs):
        if isinstance(inputs, str):
            inputs = self.name_loader_simple_resize(inputs)
        elif isinstance(inputs, (list, tuple)):
            inputs = self.array_loader_simple_resize(inputs)
        elif isinstance(inputs, np.ndarray):
            if len(inputs.shape) < 3 or len(inputs.shape) > 4:
                logger.error("Wrong inputs shape!")
                exit()
            elif len(inputs.shape) == 3:
                inputs = np.expand_dims(inputs, axis=0)
                inputs = np.ascontiguousarray(inputs)
            elif len(inputs.shape) == 4:
                inputs = np.ascontiguousarray(inputs)
        return {"images": inputs}

    def nms(self, outputs, iou_score=0.45, cls_score=0.2, shape=(640, 640)):
        result_list = []
        batch = outputs["boxes"].shape[0]
        for i in range(batch):
            boxes, cls, scores = (
                torch.from_numpy(outputs["boxes"][i]),
                torch.from_numpy(outputs["scores"][i].argmax(1)),
                torch.from_numpy(outputs["scores"][i].max(1)),
            )
            selected = scores >= cls_score
            index = batched_nms(boxes[selected], scores[selected], cls[selected], iou_score)
            boxes = boxes[index]
            boxes[0].clamp_(0, shape[1])  # x1
            boxes[1].clamp_(0, shape[0])  # y1
            boxes[2].clamp_(0, shape[1])  # x2
            boxes[3].clamp_(0, shape[0])  # y2
            result_list.append([boxes, scores[index], cls[index]])

        return result_list


if __name__ == "__main__":
    ovo = PredictorOVO("../../yolov5s_openvino/yolov5s.xml")
    inputs = "../../test/assets/zidane.jpg"
    out = ovo(inputs)
    print(out)
