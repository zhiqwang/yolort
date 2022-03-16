# Copyright (c) 2020, yolort team. All rights reserved.

import torch
from yolort.models._utils import load_from_ultralytics


def convert_yolov5_to_yolort(
    checkpoint_path: str,
    output_path: str,
    version: str = "r6.0",
    prefix: str = "yolov5_darknet_pan",
    postfix: str = "custom.pt",
):
    """
    Convert model checkpoint trained with ultralytics/yolov5 to yolort.

    Args:
        checkpoint_path (str): Path of the YOLOv5 checkpoint model.
        output_path (str): Path of the converted yolort checkpoint model.
        version (str): upstream version released by the ultralytics/yolov5, Possible
            values are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
        prefix (str): The prefix string of the saved model. Default: "yolov5_darknet_pan".
        postfix (str): The postfix string of the saved model. Default: "custom.pt".
    """

    model_info = load_from_ultralytics(checkpoint_path, version=version)
    model_state_dict = model_info["state_dict"]

    size = model_info["size"]
    use_p6 = "6" if model_info["use_p6"] else ""
    output_postfix = f"{prefix}_{size}{use_p6}_{version.replace('.', '')}_{postfix}"
    torch.save(model_state_dict, output_path / output_postfix)
