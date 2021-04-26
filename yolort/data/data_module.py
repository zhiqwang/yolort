# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from torch.nn import Module

from flash.data.process import Preprocess
from flash.data.data_module import DataModule

from .process import ObjectDetectionPreprocess

from typing import Dict, Optional


class ObjectDetectionData(DataModule):

    preprocess_cls = ObjectDetectionPreprocess

    @classmethod
    def from_coco(
        cls,
        train_folder: Optional[str] = None,
        train_ann_file: Optional[str] = None,
        train_transform: Optional[Dict[str, Module]] = None,
        val_folder: Optional[str] = None,
        val_ann_file: Optional[str] = None,
        val_transform: Optional[Dict[str, Module]] = None,
        test_folder: Optional[str] = None,
        test_ann_file: Optional[str] = None,
        test_transform: Optional[Dict[str, Module]] = None,
        predict_transform: Optional[Dict[str, Module]] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        preprocess: Preprocess = None,
        **kwargs
    ):
        preprocess = preprocess or cls.preprocess_cls(
            train_transform,
            val_transform,
            test_transform,
            predict_transform,
        )

        return cls.from_load_data_inputs(
            train_load_data_input=(train_folder, train_ann_file, train_transform),
            val_load_data_input=(val_folder, val_ann_file, val_transform) if val_folder else None,
            test_load_data_input=(test_folder, test_ann_file, test_transform) if test_folder else None,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=preprocess,
            **kwargs
        )
