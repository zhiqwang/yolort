from pathlib import Path
import json

import numpy as np
from PIL import Image
import cv2

from .builtin_meta import COCO_CATEGORIES


class Yolo2Coco:

    def __init__(self, root, split):
        self.info = {
            'year': 2021,
            'version': '1.0',
            'description': 'For object detection',
            'date_created': '2021',
        }
        self.licenses = [{
            'id': 1,
            'name': 'Attribution-NonCommercial',
            'url': 'http://creativecommons.org/licenses/by-nc-sa/4.0/',
        }]
        self.type = 'instances'
        self.split = split
        self.root_path = Path(root)

        self.annotation_root = self.root_path.joinpath('annotations')
        Path(self.annotation_root).mkdir(parents=True, exist_ok=True)

        self.categories = [{
            'id': coco_category['id'],
            'name': coco_category['name'],
            'supercategory': coco_category['supercategory'],
        } for coco_category in COCO_CATEGORIES]

    def generate(self, coco_type='instances', annotation_format='bbox'):
        label_paths = sorted(self.root_path.rglob('*.txt'))
        images, annotations = self._get_image_annotation_pairs(
            label_paths,
            annotation_format=annotation_format,
        )
        json_data = {
            'info': self.info,
            'images': images,
            'licenses': self.licenses,
            'type': self.type,
            'annotations': annotations,
            'categories': self.categories,
        }
        output_path = self.annotation_root.joinpath(f'{coco_type}_{self.split}.json')
        with open(output_path, 'w') as json_file:
            json.dump(json_data, json_file, sort_keys=True)

    def _get_image_annotation_pairs(self, label_paths, annotation_format='bbox'):
        images = []
        annotations = []
        annotation_id = 0
        for img_id, label_path in enumerate(label_paths, 1):
            img_path = label_path.parents[1].joinpath('images').joinpath(f'{label_path.stem}.tif')
            img = Image.open(img_path)
            width, height = img.size

            images.append({
                'date_captured': '2021',
                'file_name': str(img_path.relative_to(self.root_path)),
                'id': img_id,
                'license': 1,
                'url': '',
                'height': height,
                'width': width,
            })

            with open(label_path, 'r') as f:
                for line in f:
                    label_info = line.strip().split()
                    assert len(label_info) == 5
                    annotation_id += 1

                    label_info = [int(i) for i in label_info]
                    category_id, vertex_info = label_info[0], label_info[1:]
                    if annotation_format == 'bbox':
                        segmentation, bbox, area = self._get_annotation(vertex_info, height, width)
                    else:
                        raise NotImplementedError

                    annotations.append({
                        'segmentation': segmentation,
                        'area': np.float(area),
                        'iscrowd': 0,
                        'image_id': img_id,
                        'bbox': bbox,
                        'category_id': category_id,
                        'id': annotation_id,
                    })

        return images, annotations

    @staticmethod
    def _get_annotation(vertex_info, height, width):

        polygons = np.array(vertex_info).reshape(4, 1, -1)
        x, y, w, h = cv2.boundingRect(polygons)

        segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]],

        assert len(polygons) > 0, "COCOAPI does not support empty polygons"

        return segmentation, [x, y, w, h], area
