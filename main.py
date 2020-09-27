# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
"""
Implements the Wrapped YOLOV5 framework
"""

import torch


if __name__ == "__main__":

    model_yolo = torch.hub.load(
        'ultralytics/yolov5', 'yolov5s', pretrained=False, channels=3, classes=80,
    )

    from yolov5.models.yolo import Model

    model_test = Model(cfg="./yolov5/models/yolov5s.yaml")

    print(f'model: {model_test}')
