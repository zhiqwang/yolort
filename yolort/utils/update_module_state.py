# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
from functools import reduce
import torch
from torch import nn

from ..models import yolo

from typing import Any


def update_module_state_from_ultralytics(
    arch: str = 'yolov5s',
    version: str = 'v4.0',
    num_classes: int = 80,
    **kwargs: Any,
):
    architecture_maps = {
        'yolov5s_v3.1': 'yolov5_darknet_pan_s_r31',
        'yolov5m_v3.1': 'yolov5_darknet_pan_m_r31',
        'yolov5l_v3.1': 'yolov5_darknet_pan_l_r31',
        'yolov5s_v4.0': 'yolov5_darknet_pan_s_r40',
        'yolov5m_v4.0': 'yolov5_darknet_pan_m_r40',
        'yolov5l_v4.0': 'yolov5_darknet_pan_l_r40',
    }

    model = torch.hub.load(f'ultralytics/yolov5:{version}', arch, pretrained=True)

    module_state_updater = ModuleStateUpdate(arch=architecture_maps[f'{arch}_{version}'],
                                             num_classes=num_classes, **kwargs)

    module_state_updater.updating(model)

    return module_state_updater.model.half()


class ModuleStateUpdate:
    """
    Update checkpoint from ultralytics yolov5
    """
    def __init__(
        self,
        arch: str = 'yolov5_darknet_pan_s_r31',
        num_classes: int = 80,
        inner_block_maps: dict = {'0': '9', '1': '10', '3': '13', '4': '14'},
        layer_block_maps: dict = {'0': '17', '1': '18', '2': '20', '3': '21', '4': '23'},
        head_ind: int = 24,
        head_name: str = 'm',
    ) -> None:
        # Configuration for making the keys consistent
        self.inner_block_maps = inner_block_maps
        self.layer_block_maps = layer_block_maps
        self.head_ind = head_ind
        self.head_name = head_name
        # Set model
        self.model = yolo.__dict__[arch](num_classes=num_classes)

    def updating(self, state_dict):
        # Obtain module state
        state_dict = obtain_module_sequential(state_dict)

        # Update backbone features
        for name, params in self.model.backbone.body.named_parameters():
            params.data.copy_(
                self.attach_parameters_block(state_dict, name, None))

        for name, buffers in self.model.backbone.body.named_buffers():
            buffers.copy_(
                self.attach_parameters_block(state_dict, name, None))

        # Update PAN features
        for name, params in self.model.backbone.pan.inner_blocks.named_parameters():
            params.data.copy_(
                self.attach_parameters_block(state_dict, name, self.inner_block_maps))

        for name, buffers in self.model.backbone.pan.inner_blocks.named_buffers():
            buffers.copy_(
                self.attach_parameters_block(state_dict, name, self.inner_block_maps))

        for name, params in self.model.backbone.pan.layer_blocks.named_parameters():
            params.data.copy_(
                self.attach_parameters_block(state_dict, name, self.layer_block_maps))

        for name, buffers in self.model.backbone.pan.layer_blocks.named_buffers():
            buffers.copy_(
                self.attach_parameters_block(state_dict, name, self.layer_block_maps))

        # Update box heads
        for name, params in self.model.head.named_parameters():
            params.data.copy_(
                self.attach_parameters_heads(state_dict, name))

        for name, buffers in self.model.head.named_buffers():
            buffers.copy_(
                self.attach_parameters_heads(state_dict, name))

    @staticmethod
    def attach_parameters_block(state_dict, name, block_maps=None):
        keys = name.split('.')
        ind = int(block_maps[keys[0]]) if block_maps else int(keys[0])
        return rgetattr(state_dict[ind], keys[1:])

    def attach_parameters_heads(self, state_dict, name):
        keys = name.split('.')
        ind = int(keys[1])
        return rgetattr(getattr(state_dict[self.head_ind], self.head_name)[ind], keys[2:])


def rgetattr(obj, attr, *args):
    """
    Nested version of getattr.
    See <https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects>
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return reduce(_getattr, [obj] + attr)


def obtain_module_sequential(state_dict):
    if isinstance(state_dict, nn.Sequential):
        return state_dict
    else:
        return obtain_module_sequential(state_dict.model)
