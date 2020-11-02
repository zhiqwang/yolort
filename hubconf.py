import torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from models.yolo import Model
from models.box_head import YoloHead, PostProcess
from models.yolo_wrapped import YoloBody, YOLO


def yolov5(cfg_path='./models/yolov5s.yaml', checkpoint_path=None):
    min_size, max_size, image_mean, image_std = 352, 416, [0, 0, 0], [1, 1, 1]
    transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
    backbone = Model(cfg=cfg_path)
    layer_body = YoloBody(yolo_body=backbone, return_layers={'17': '0', '20': '1', '23': '2'})

    args_detect = [
        80,
        [8., 16., 32.],
        [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
        [128, 256, 512],
    ]
    layer_box_head = YoloHead(*args_detect)
    post_process = PostProcess(conf_thres=0.4, iou_thres=0.5)

    model = YOLO(layer_body, layer_box_head, post_process, transform, [8., 16., 32.])

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint)

    return model
