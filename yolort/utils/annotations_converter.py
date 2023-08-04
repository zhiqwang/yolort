# Copyright (c) 2020, yolort team. All rights reserved.

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image


class AnnotationsConverter:
    """
    Make a MSCOCO JSON format annotations from YOLO format. We first find all the images in that
    directory and then match their corresponding labels. The default setting places labels and
    images in the same directory. You can set the name of label_dir to substitute the path for the
    desired labels if the labels and images are not in the same directory. Our replacement strategy
    is to replace image_dir with label_dir, so please also set the image_dir carefully.

    Args:
        root (string): Root directory of the dataset
        metalabels (string | List[string]): Concrete label names of different classes
        image_dir (string, optional): Name of the path to be replaced if it isn't None,
            default is None
        label_dir (string, optional): Name of the replaced path for the desired labels,
            default is None
        split (string, optional): The dataset split, either 'train' (default) or 'test'
        year (int, optional): The year of the dataset, default is 2017
        set_license (bool, optional): Determine whether to set license, default is False
        image_posix (string, optional): Posix of the image, default is 'jpg'
    """

    def __init__(
        self,
        root: str,
        metalabels: Union[str, List[str]],
        image_dir: Optional[str] = None,
        label_dir: Optional[str] = None,
        split: str = "train",
        year: int = 2017,
        set_license: bool = False,
        image_posix: str = "jpg",
    ) -> None:

        self._year = year
        self.type = "instances"
        self.split = f"{split}{year}"
        self.root_path = Path(root)
        self.image_posix = image_posix
        self.image_dir = image_dir or ""
        self.label_dir = label_dir or ""
        self.annotation_root = self._set_annotation_path()
        self.metadata = self._get_metadata(metalabels)
        self.metainfo = self._set_metainfo()
        self.licenses = self._get_licenses(set_license)
        self.categories = self._set_categories()

    def _set_annotation_path(self):
        annotation_root = self.root_path / "annotations"
        Path(annotation_root).mkdir(parents=True, exist_ok=True)
        return annotation_root

    def _set_metainfo(self):
        return {
            "year": self._year,
            "version": "1.0",
            "description": "For object detection",
            "date_created": f"{self._year}",
        }

    @staticmethod
    def _get_metadata(metalabels: Union[str, List[str]]):
        if isinstance(metalabels, list):
            return metalabels

        if isinstance(metalabels, str):
            return np.loadtxt(metalabels, dtype="str", delimiter="\n")

        raise TypeError(f"path of metalabels of list of strings expected, got {type(metalabels)}")

    @staticmethod
    def _get_licenses(set_license):
        if set_license:
            licenses = [
                {
                    "id": 1,
                    "name": "GNU General Public License v3.0",
                    "url": "https://github.com/zhiqwang/yolort/blob/main/LICENSE",
                }
            ]
            return licenses

        return None

    def _set_categories(self):
        if isinstance(self.metadata[0], dict):
            return [
                {"id": coco_category["id"], "name": coco_category["name"]} for coco_category in self.metadata
            ]
        elif isinstance(self.metadata[0], str):
            return [{"id": label_id, "name": label_name} for label_id, label_name in enumerate(self.metadata)]
        else:
            raise NotImplementedError("Currently doesn't support this methods.")

    def generate(self, coco_type="instances", annotation_format="bbox"):
        image_paths = sorted(self.root_path.rglob(f"*.{self.image_posix}"))
        images, annotations = self._get_image_annotation_pairs(
            image_paths,
            annotation_format=annotation_format,
        )
        json_data = {
            "info": self.metainfo,
            "images": images,
            "type": self.type,
            "annotations": annotations,
            "categories": self.categories,
        }
        if self.licenses is not None:
            json_data["licenses"] = self.licenses
        output_path = self.annotation_root / f"{coco_type}_{self.split}.json"
        with open(output_path, "w") as json_file:
            json.dump(json_data, json_file, sort_keys=True)

    def _get_image_annotation_pairs(self, image_paths, annotation_format="bbox"):
        images = []
        annotations = []
        annotation_id = 0
        for img_id, img_path in enumerate(image_paths, 1):
            label_path = str(img_path).replace(f"{self.image_posix}", "txt")
            label_path = label_path.replace(self.image_dir, self.label_dir)
            width, height = Image.open(img_path).size

            images.append(
                {
                    "date_captured": f"{self._year}",
                    "file_name": str(Path(img_path).relative_to(self.root_path)),
                    "id": img_id,
                    "license": 1,
                    "url": "",
                    "height": height,
                    "width": width,
                }
            )

            with open(label_path, "r") as f:
                for line in f:
                    label_info = line.strip().split()
                    assert len(label_info) == 5
                    annotation_id += 1

                    category_id, vertex_info = label_info[0], label_info[1:]
                    category_id = self.categories[int(category_id)]["id"]
                    if annotation_format == "bbox":
                        segmentation, bbox, area = self._get_annotation(vertex_info, height, width)
                    else:
                        raise NotImplementedError

                    annotations.append(
                        {
                            "segmentation": segmentation,
                            "area": area,
                            "iscrowd": 0,
                            "image_id": img_id,
                            "bbox": bbox,
                            "category_id": category_id,
                            "id": annotation_id,
                        }
                    )

        return images, annotations

    @staticmethod
    def _get_annotation(vertex_info, height, width):

        cx, cy, w, h = [float(i) for i in vertex_info]
        cx = cx * width
        cy = cy * height
        w = w * width
        h = h * height
        x = cx - w / 2
        y = cy - h / 2

        segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        area = w * h

        bbox = [x, y, w, h]
        return segmentation, bbox, area
