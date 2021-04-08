# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from pathlib import PosixPath
from zipfile import ZipFile

import torch


def prepare_coco128(
    data_path: PosixPath,
    dirname: str = 'coco128',
) -> None:
    """
    Prepare coco128 dataset to test.

    Args:
        data_path (PosixPath): root path of coco128 dataset.
        dirname (str): the directory name of coco128 dataset. Default: 'coco128'.
    """
    if not data_path.is_dir():
        print(f'Create a new directory: {data_path}')
        data_path.mkdir(parents=True, exist_ok=True)

    zip_path = data_path / 'coco128.zip'
    coco128_url = 'https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip'
    if not zip_path.is_file():
        print(f'Downloading coco128 datasets form {coco128_url}')
        torch.hub.download_url_to_file(coco128_url, zip_path, hash_prefix='a67d2887')

    coco128_path = data_path / dirname
    if not coco128_path.is_dir():
        print(f'Unzipping dataset to {coco128_path}')
        with ZipFile(zip_path, 'r') as zip_obj:
            zip_obj.extractall(data_path)
