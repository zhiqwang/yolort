import torch

from models.yolo import Detect, Model
from models.yolo_wrapped import Body, YOLO


def yolov5(cfg_path='./models/yolov5s.yaml', checkpoint_path=None):
    backbone = Model(cfg=cfg_path)
    layer_body = Body(body=backbone, return_layers_body={'17': '0', '20': '1', '23': '2'})

    args_detect = [
        80,
        [8., 16., 32.],
        [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
        [128, 256, 512],
    ]
    layer_box_head = Detect(*args_detect)

    model = YOLO(layer_body, layer_box_head, [8., 16., 32.])

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint)

    return model
