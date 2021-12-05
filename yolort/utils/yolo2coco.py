# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import argparse
import cv2
import json
from pathlib import Path
from PIL import Image
try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None

from .builtin_meta import COCO_CATEGORIES

class YOLO2COCO:
    def __init__(self, root, split):
        self.info = {
            "year": 2021,
            "version": "1.0",
            "description": "For object detection",
            "date_created": "2021",
        }
        self.licenses = [
            {
                "id": 1,
                "name": "GNU General Public License v3.0",
                "url": "https://github.com/zhiqwang/yolov5-rt-stack/blob/master/LICENSE",
            }
        ]
        self.type = "instances"
        self.split = split
        self.root_path = Path(root)
        self.label_path = self.root_path / "labels"
        self.annotation_root = self.root_path / "annotations"
        Path(self.annotation_root).mkdir(parents=True, exist_ok=True)

        self.categories = [
            {
                "id": coco_category["id"],
                "name": coco_category["name"],
                "supercategory": coco_category["supercategory"],
            }
            for coco_category in COCO_CATEGORIES
        ]

    def generate(self, coco_type="instances", annotation_format="bbox"):
        label_paths = sorted(self.label_path.rglob("*.txt"))
        images, annotations = self._get_image_annotation_pairs(
            label_paths,
            annotation_format=annotation_format,
        )
        json_data = {
            "info": self.info,
            "images": images,
            "licenses": self.licenses,
            "type": self.type,
            "annotations": annotations,
            "categories": self.categories,
        }
        output_path = self.annotation_root / f"{coco_type}_{self.split}.json"
        with open(output_path, "w") as json_file:
            json.dump(json_data, json_file, sort_keys=True)

    def _get_image_annotation_pairs(self, label_paths, annotation_format="bbox"):
        images = []
        annotations = []
        annotation_id = 0
        for img_id, label_path in enumerate(label_paths, 1):
            img_path = str(label_path).replace("labels", "images").replace("txt", "jpg")
            img = Image.open(img_path)
            width, height = img.size

            images.append(
                {
                    "date_captured": "2021",
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

    @staticmethod
    def _get_ellipse_mask(img_shape, ellipse_info, mask_id=1):
        assert len(img_shape) == 2
        img_mask = np.zeros(img_shape, dtype="uint8")
        center = tuple([int(round(i)) for i in ellipse_info[:2]])
        axes = tuple([int(np.ceil(i / 2)) for i in ellipse_info[2:4]])

        cv2.ellipse(img_mask, center, axes, ellipse_info[4], 0, 360, (mask_id,), -1)
        return img_mask

    def _get_seg_annotation(self, img_mask, img_vis=None):

        _, contours, _ = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        segmentation = []
        for contour in contours:
            # Valid polygons have >= 6 coordinates (3 points)
            if contour.size >= 6:
                segmentation.append(contour.flatten().tolist())

        RLEs = coco_mask.frPyObjects(segmentation, img_mask.shape[0], img_mask.shape[1])
        RLE = coco_mask.merge(RLEs)
        # RLE = coco_mask.encode(np.asfortranarray(img_mask))
        area = coco_mask.area(RLE)
        [x, y, w, h] = cv2.boundingRect(img_mask)

        if img_vis is not None:
            img_vis = img_vis.copy()
            cv2.drawContours(img_vis, contours, -1, (0, 255, 0), 1)
            cv2.rectangle(img_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            img_vis = None

        return segmentation, [x, y, w, h], area, img_vis

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Datasets converter from yolo to coco", add_help=False)

    parser.add_argument("--data_path", default="../coco128", help="Dataset root path")
    parser.add_argument(
        "--split",
        default="train2017",
        help="Dataset split part, optional: [train2017, val2017]",
    )

    args = parser.parse_args()

    converter = YOLO2COCO(args.data_path, args.split)
    converter.generate()
