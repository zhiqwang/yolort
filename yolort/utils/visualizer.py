# Copyright (c) 2022, yolort team. All rights reserved.

from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor


class Visualizer:
    """
    Visualizer that draws data about detection on images.

    It contains methods like `draw_{text,box}` that draw primitive objects to images, as well as
    high-level wrappers like `draw_{instance_predictions,dataset_dict}` that draw composite data
    in some pre-defined style.

    This visualizer focuses on high rendering quality rather than performance. It is not
    designed to be used for real-time applications.

    Reference:
        We have followed most of the interfaces of detectron2 here, but the implementation will be
        a bit different. Check out the following for more details.
        https://github.com/facebookresearch/detectron2/blob/9258799/detectron2/utils/visualizer.py

    Args:
        image (torch.Tensor or numpy.ndarray): Tensor of shape (C x H x W) or ndarray of
            shape (H x W x C) with dtype uint8.
        metalabels (string, optional): Concrete label names of different classes. Default: None
        instance_mode (int, optional): defines one of the pre-defined style for drawing
            instances on an image. Default: None
    """

    def __init__(
        self,
        image: Union[Tensor, np.ndarray],
        *,
        metalabels: Optional[str] = None,
        scale: float = 1.0,
        line_width: Optional[int] = None,
    ) -> None:

        if isinstance(image, torch.Tensor):
            if image.dtype != torch.uint8:
                raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
            if image.dim() != 3:
                raise ValueError("Pass individual images, not batches")
            if image.size(0) not in {1, 3}:
                raise ValueError("Only grayscale and RGB images are supported")
            # Handle Grayscale images
            if image.size(0) == 1:
                image = torch.tile(image, (3, 1, 1))
            self.img = image.permute(1, 2, 0).cpu().numpy()
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                raise ValueError(f"Numpy uint8 expected, got {image.dtype}")
            if image.ndim != 3:
                raise ValueError("Currently only RGB images are supported")
            self.img = image
        else:
            raise TypeError(f"Tensor or numpy.ndarray expected, got {type(image)}")

        # Set dataset metadata (e.g. class names)
        self.metadata = None
        if metalabels is not None:
            self.metadata = np.loadtxt(metalabels, dtype='str', delimiter='\n')

        self.scale = scale
        self.cpu_device = torch.device("cpu")
        self.line_width = line_width or max(round(sum(self.img.shape) / 2 * 0.003), 2)
        self.output = self.img

    def draw_instance_predictions(self, predictions: Dict):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (dict): the output of an instance detection model. Following
                fields will be used to draw: "boxes", "labels", "scores".

        Returns:
            np.ndarray: image object with visualizations.
        """
        boxes = predictions.get("boxes", None)
        scores = predictions.get("scores", None)
        labels = predictions.get("labels", None)
        labels = self._create_text_labels(labels, scores)

        self.overlay_instances(boxes=boxes, labels=labels, assigned_colors=None)
        return self.output

    def overlay_instances(
        self,
        *,
        boxes: Optional[Union[Tensor, np.ndarray]] = None,
        labels: Optional[List[str]] = None,
        assigned_colors: Optional[List[str]] = None,
    ):
        """
        Overlay bounding boxes and labels on input image.

        Args:
            boxes (Tensor or ndarray, optional): Tensor or numpy array of size (N, 4) containing
                bounding boxes in (xmin, ymin, xmax, ymax) format for the N objects in
                a single image. Note that the boxes are absolute coordinates with respect
                to the image. In other words: `0 <= xmin < xmax < W` and `0 <= ymin < ymax < H`.
                Default: None
            labels (List[string], optional): List containing the text to be displayed for each
                instance. Default: None
            colors (color or list of colors, optional): List containing the colors
                of the boxes or single color for all boxes. The color can be represented as
                PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
                By default, random colors are generated for boxes. Default: None

        Returns:
            np.ndarray: image object with visualizations.
        """
        num_instances = 0
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if labels is not None:
            assert len(labels) == num_instances
        if num_instances == 0:
            return self.output

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.

                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                self.draw_text(labels[i], text_pos, color=lighter_color)

        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        """
        Draws bounding boxes on given image.
        The values of the input image should be uint8 between 0 and 255.

        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            np.ndarray: image object with box drawn.
        """
        p1, p2 = (int(box_coord[0]), int(box_coord[1])), (int(box_coord[2]), int(box_coord[3]))
        cv2.rectangle(self.output, p1, p2, edge_color, thickness=self.line_width, lineType=cv2.LINE_AA)
        return self.output

    def draw_text(
        self,
        text: str,
        position: Tuple,
        *,
        font_size: Optional[int] = None,
        color: str = "g",
        txt_colors: Tuple[int, int, int] = (255, 255, 255),
    ):
        """
        Draws text on given image.

        Args:
            text (string): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used. Default: None
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.

        Returns:
            np.ndarray: image object with text drawn.
        """
        p1, p2 = (int(position[0]), int(position[1])), (int(position[2]), int(position[3]))

        if font_size is None:
            font_size = max(self.line_width - 1, 1)  # font thickness
        w, h = cv2.getTextSize(text, 0, fontScale=self.line_width / 3, thickness=font_size)[0]
        outside = p1[1] - h - 3 >= 0  # text fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(self.output, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            self.output,
            text,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            self.line_width / 3,
            txt_colors,
            thickness=font_size,
            lineType=cv2.LINE_AA,
        )
        return self.output

    @staticmethod
    def _create_text_labels(
        classes: Optional[List[int]] = None,
        scores: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        is_crowd: Optional[List[bool]] = None,
    ):
        """
        Generate labels that classes and scores can match, and set class back to its original
        name if concrete class names are provided.
        """
        labels = None
        if classes is not None:
            if class_names is not None and len(class_names) > 0:
                labels = [class_names[i] for i in classes]
            else:
                labels = [str(i) for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
        if labels is not None and is_crowd is not None:
            labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
        return labels

    def _change_color_brightness(self, color: Tuple[int, int, int], brightness_factor: float):
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.

        Args:
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
                0 will correspond to no change, a factor in [-1.0, 0) range will result in
                a darker color and a factor in (0, 1.0] range will result in a lighter color.

        Returns:
            modified_color (tuple[double]): a tuple containing the RGB values of the
                modified color. Each value in the tuple is in the [0.0, 1.0] range.
        """
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        return color

    def _convert_boxes(self, boxes: Tensor):
        """
        Convert different format of boxes to an Nx4 array.
        """
        return boxes.cpu().detach().numpy()
