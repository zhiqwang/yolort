# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
from functools import reduce
from typing import Dict, Optional

from torch import nn

from yolort.models import yolo
from yolort.v5 import get_yolov5_size


class ModuleStateUpdate:
    """
    Update checkpoint from ultralytics yolov5.
    """

    def __init__(
        self,
        depth_multiple: Optional[float] = None,
        width_multiple: Optional[float] = None,
        version: str = "r6.0",
        num_classes: int = 80,
        inner_block_maps: Optional[Dict[str, str]] = None,
        layer_block_maps: Optional[Dict[str, str]] = None,
        head_ind: int = 24,
        head_name: str = "m",
        use_p6: bool = False,
    ) -> None:

        assert depth_multiple is not None, "depth_multiple must be set."
        assert width_multiple is not None, "width_multiple must be set."

        # Configuration for making the keys consistent
        if inner_block_maps is None:
            inner_block_maps = {
                "0": "9",
                "1": "10",
                "3": "13",
                "4": "14",
            }
        self.inner_block_maps = inner_block_maps
        if layer_block_maps is None:
            layer_block_maps = {
                "0": "17",
                "1": "18",
                "2": "20",
                "3": "21",
                "4": "23",
            }
        self.layer_block_maps = layer_block_maps
        self.head_ind = head_ind
        self.head_name = head_name

        # Set model
        yolov5_size = get_yolov5_size(depth_multiple, width_multiple)
        backbone_name = f"darknet_{yolov5_size}_{version.replace('.', '_')}"
        weights_name = (
            f"yolov5_darknet_pan_{yolov5_size}_{version.replace('.', '')}_coco"
        )
        model = yolo.build_model(
            backbone_name,
            depth_multiple,
            width_multiple,
            version,
            weights_name,
            num_classes=num_classes,
            use_p6=use_p6,
        )
        self.model = model

    def updating(self, state_dict):
        # Obtain module state
        state_dict = obtain_module_sequential(state_dict)

        # Update backbone features
        for name, params in self.model.backbone.body.named_parameters():
            params.data.copy_(self.attach_parameters_block(state_dict, name, None))

        for name, buffers in self.model.backbone.body.named_buffers():
            buffers.copy_(self.attach_parameters_block(state_dict, name, None))

        # Update PAN features
        for name, params in self.model.backbone.pan.inner_blocks.named_parameters():
            params.data.copy_(
                self.attach_parameters_block(state_dict, name, self.inner_block_maps)
            )

        for name, buffers in self.model.backbone.pan.inner_blocks.named_buffers():
            buffers.copy_(
                self.attach_parameters_block(state_dict, name, self.inner_block_maps)
            )

        for name, params in self.model.backbone.pan.layer_blocks.named_parameters():
            params.data.copy_(
                self.attach_parameters_block(state_dict, name, self.layer_block_maps)
            )

        for name, buffers in self.model.backbone.pan.layer_blocks.named_buffers():
            buffers.copy_(
                self.attach_parameters_block(state_dict, name, self.layer_block_maps)
            )

        # Update box heads
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
        return rgetattr(
            getattr(state_dict[self.head_ind], self.head_name)[ind], keys[2:]
        )


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
