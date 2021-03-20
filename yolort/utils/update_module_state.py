# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
from functools import reduce
import torch
from torch import nn

from ..models import yolo

from typing import Any, Union, Optional


def update_module_state_from_ultralytics(
    arch: str = 'yolov5s',
    version: str = 'v4.0',
    feature_fusion_type: str = 'PAN',
    num_classes: int = 80,
    custom_path_or_model: Optional[Union[str, dict, nn.Module]] = None,
    set_fp16: bool = True,
    **kwargs: Any,
):
    """
    Allows the user to specify a file to use when loading an ultralytics model for conversion.
    This is valuable for users who have already trained their models using ultralytics and don't
    wish to re-train.

    Args:
        arch (str): yolo architecture. Possible values are 'yolov5s', 'yolov5m' and 'yolov5l'.
            Default: 'yolov5s'.
        version (str): the released version of ultralytics. Possible values are 'v3.1' and 'v4.0'.
            Default: 'v4.0'.
        feature_fusion_type (str): the type of fature fusion. Possible values are PAN and TAN.
            Default: 'PAN'.
        num_classes (int): number of detection classes (doesn't including background).
            Default: 80.
        custom_path_or_model (Optional[Union[str, dict, nn.Module]]): custom path or model.
            Possible arguments are str, dict, nn.Module, None, to respond to different scenarios.
            Ref: usage details for calling parameters:
            <https://github.com/ultralytics/yolov5/blob/ed2c742/hubconf.py#L112>
            and tutorial of loading ultralytics/YOLOv5 from PyTorch Hub:
            <https://github.com/ultralytics/yolov5/issues/36>
            - None: use model trained from COCO datasets as ultralytics
            - str: string of path to model
            - dict: torch.load('path/to/model.pt')
            - nn.Module: torch.load('path/to/model.pt')['model']
            Default: None.
        set_fp16 (bool): allow selective conversion to fp16 or not.
            Default: True.
    """
    architecture_maps = {
        'yolov5s_pan_v3.1': 'yolov5_darknet_pan_s_r31',
        'yolov5m_pan_v3.1': 'yolov5_darknet_pan_m_r31',
        'yolov5l_pan_v3.1': 'yolov5_darknet_pan_l_r31',
        'yolov5s_pan_v4.0': 'yolov5_darknet_pan_s_r40',
        'yolov5m_pan_v4.0': 'yolov5_darknet_pan_m_r40',
        'yolov5l_pan_v4.0': 'yolov5_darknet_pan_l_r40',
        'yolov5s_tan_v4.0': 'yolov5_darknet_tan_s_r40',
    }

    if custom_path_or_model is None:
        model = torch.hub.load(f'ultralytics/yolov5:{version}', arch, pretrained=True)
    else:
        model = torch.hub.load(f'ultralytics/yolov5:{version}', 'custom', path_or_model=custom_path_or_model)

    key_arch = f'{arch}_{feature_fusion_type.lower()}_{version}'
    assert key_arch in architecture_maps, (
        "Currently does't support your architecture, fell free to report a issue to yolort")

    module_state_updater = ModuleStateUpdate(
        arch=architecture_maps[key_arch],
        num_classes=num_classes,
        **kwargs,
    )

    module_state_updater.updating(model)

    if set_fp16:
        module_state_updater.model.half()

    return module_state_updater.model


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
