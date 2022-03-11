# Copyright (c) 2020, yolort team. All rights reserved.

from functools import reduce
from typing import Dict, List, Optional

import torch
from torch import nn
from yolort.models import yolo
from yolort.v5 import load_yolov5_model, get_yolov5_size


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


def load_from_ultralytics(checkpoint_path: str, version: str = "r6.0"):
    """
    Allows the user to load model state file from the checkpoint trained from
    the ultralytics/yolov5.

    Args:
        checkpoint_path (str): Path of the YOLOv5 checkpoint model.
        version (str): upstream version released by the ultralytics/yolov5, Possible
            values are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
    """

    assert version in ["r3.1", "r4.0", "r6.0"], "Currently does not support this version."

    checkpoint_yolov5 = load_yolov5_model(checkpoint_path)
    num_classes = checkpoint_yolov5.yaml["nc"]
    strides = checkpoint_yolov5.stride
    # YOLOv5 will change the anchors setting when using the auto-anchor mechanism. So we
    # use the following formula to compute the anchor_grids instead of attaching it via
    # checkpoint_yolov5.yaml["anchors"]
    num_anchors = checkpoint_yolov5.model[-1].anchors.shape[1]
    anchor_grids = (
        (checkpoint_yolov5.model[-1].anchors * checkpoint_yolov5.model[-1].stride.view(-1, 1, 1))
        .reshape(1, -1, 2 * num_anchors)
        .tolist()[0]
    )

    depth_multiple = checkpoint_yolov5.yaml["depth_multiple"]
    width_multiple = checkpoint_yolov5.yaml["width_multiple"]

    use_p6 = False
    if len(strides) == 4:
        use_p6 = True

    if use_p6:
        inner_block_maps = {"0": "11", "1": "12", "3": "15", "4": "16", "6": "19", "7": "20"}
        layer_block_maps = {"0": "23", "1": "24", "2": "26", "3": "27", "4": "29", "5": "30", "6": "32"}
        p6_block_maps = {"0": "9", "1": "10"}
        head_ind = 33
        head_name = "m"
    else:
        inner_block_maps = {"0": "9", "1": "10", "3": "13", "4": "14"}
        layer_block_maps = {"0": "17", "1": "18", "2": "20", "3": "21", "4": "23"}
        p6_block_maps = None
        head_ind = 24
        head_name = "m"

    module_state_updater = ModuleStateUpdate(
        depth_multiple,
        width_multiple,
        inner_block_maps=inner_block_maps,
        layer_block_maps=layer_block_maps,
        p6_block_maps=p6_block_maps,
        strides=strides,
        anchor_grids=anchor_grids,
        head_ind=head_ind,
        head_name=head_name,
        num_classes=num_classes,
        version=version,
        use_p6=use_p6,
    )
    module_state_updater.updating(checkpoint_yolov5)
    state_dict = module_state_updater.model.half().state_dict()

    size = get_yolov5_size(depth_multiple, width_multiple)

    return {
        "num_classes": num_classes,
        "depth_multiple": depth_multiple,
        "width_multiple": width_multiple,
        "strides": strides,
        "anchor_grids": anchor_grids,
        "use_p6": use_p6,
        "size": size,
        "state_dict": state_dict,
    }


class ModuleStateUpdate:
    """
    Update checkpoint from ultralytics yolov5.
    """

    def __init__(
        self,
        depth_multiple: float,
        width_multiple: float,
        inner_block_maps: Optional[Dict[str, str]] = None,
        layer_block_maps: Optional[Dict[str, str]] = None,
        p6_block_maps: Optional[Dict[str, str]] = None,
        strides: Optional[List[int]] = None,
        anchor_grids: Optional[List[List[float]]] = None,
        head_ind: int = 24,
        head_name: str = "m",
        num_classes: int = 80,
        version: str = "r6.0",
        use_p6: bool = False,
    ) -> None:

        # Configuration for making the keys consistent
        if inner_block_maps is None:
            inner_block_maps = {"0": "9", "1": "10", "3": "13", "4": "14"}
        self.inner_block_maps = inner_block_maps
        if layer_block_maps is None:
            layer_block_maps = {"0": "17", "1": "18", "2": "20", "3": "21", "4": "23"}
        self.layer_block_maps = layer_block_maps
        self.p6_block_maps = p6_block_maps
        self.head_ind = head_ind
        self.head_name = head_name

        # Set model
        yolov5_size = get_yolov5_size(depth_multiple, width_multiple)
        backbone_name = f"darknet_{yolov5_size}_{version.replace('.', '_')}"
        self.model = yolo.build_model(
            backbone_name,
            depth_multiple,
            width_multiple,
            version,
            num_classes=num_classes,
            use_p6=use_p6,
            strides=strides,
            anchor_grids=anchor_grids,
        )

    def updating(self, state_dict):
        # Obtain module state
        state_dict = obtain_module_sequential(state_dict)

        # Update backbone weights
        for name, params in self.model.backbone.body.named_parameters():
            params.data.copy_(self.attach_parameters_block(state_dict, name, None))

        for name, buffers in self.model.backbone.body.named_buffers():
            buffers.copy_(self.attach_parameters_block(state_dict, name, None))

        # Update PAN weights
        # Updating P6 weights
        if self.p6_block_maps is not None:
            for name, params in self.model.backbone.pan.intermediate_blocks.p6.named_parameters():
                params.data.copy_(self.attach_parameters_block(state_dict, name, self.p6_block_maps))

            for name, buffers in self.model.backbone.pan.intermediate_blocks.p6.named_buffers():
                buffers.copy_(self.attach_parameters_block(state_dict, name, self.p6_block_maps))

        # Updating inner_block weights
        for name, params in self.model.backbone.pan.inner_blocks.named_parameters():
            params.data.copy_(self.attach_parameters_block(state_dict, name, self.inner_block_maps))

        for name, buffers in self.model.backbone.pan.inner_blocks.named_buffers():
            buffers.copy_(self.attach_parameters_block(state_dict, name, self.inner_block_maps))

        # Updating layer_block weights
        for name, params in self.model.backbone.pan.layer_blocks.named_parameters():
            params.data.copy_(self.attach_parameters_block(state_dict, name, self.layer_block_maps))

        for name, buffers in self.model.backbone.pan.layer_blocks.named_buffers():
            buffers.copy_(self.attach_parameters_block(state_dict, name, self.layer_block_maps))

        # Update YOLOHead weights
        for name, params in self.model.head.named_parameters():
            params.data.copy_(self.attach_parameters_heads(state_dict, name))

        for name, buffers in self.model.head.named_buffers():
            buffers.copy_(self.attach_parameters_heads(state_dict, name))

    @staticmethod
    def attach_parameters_block(state_dict, name, block_maps=None):
        keys = name.split(".")
        ind = int(block_maps[keys[0]]) if block_maps else int(keys[0])
        return rgetattr(state_dict[ind], keys[1:])

    def attach_parameters_heads(self, state_dict, name):
        keys = name.split(".")
        ind = int(keys[1])
        return rgetattr(getattr(state_dict[self.head_ind], self.head_name)[ind], keys[2:])


def rgetattr(obj, attr, *args):
    """
    Nested version of getattr.
    Ref: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr)


def obtain_module_sequential(state_dict):
    if isinstance(state_dict, nn.Sequential):
        return state_dict
    else:
        return obtain_module_sequential(state_dict.model)
