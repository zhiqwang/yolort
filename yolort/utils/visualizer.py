# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2022, yolort team. All rights reserved.

from enum import Enum, unique
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.figure as mplfigure
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torch import Tensor


@unique
class ColorMode(Enum):
    """
    Enum of different color modes to use for instance visualizations.
    """

    IMAGE = 0
    """
    Picks a random color for every instance and overlay segmentations with low opacity.
    """
    SEGMENTATION = 1
    """
    Let instances of the same category have similar colors
    (from metadata.thing_colors), and overlay them with
    high opacity. This provides more attention on the quality of segmentation.
    """
    IMAGE_BW = 2
    """
    Same as IMAGE, but convert all areas without masks to gray-scale.
    Only available for drawing per-instance mask predictions.
    """


class VisImage:
    def __init__(self, img: np.ndarray, scale: float = 1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3) in range [0, 255].
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        """
        Args:
            img: same as in __init__
        """
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


class Visualizer:
    """
    Visualizer that draws data about detection/segmentation on images.

    It contains methods like `draw_{text,box,circle,line,binary_mask,polygon}`
    that draw primitive objects to images, as well as high-level wrappers like
    `draw_{instance_predictions,sem_seg,panoptic_seg_predictions,dataset_dict}`
    that draw composite data in some pre-defined style.

    Note that the exact visualization style for the high-level wrappers are subject to change.
    Style such as color, opacity, label contents, visibility of labels, or even the visibility
    of objects themselves (e.g. when the object is too small) may change according to different
    heuristics, as long as the results still look visually reasonable.

    To obtain a consistent style, you can implement custom drawing functions with the
    abovementioned primitive methods instead. If you need more customized visualization
    styles, you can process the data yourself following their format documented in
    tutorials (:doc:`/tutorials/models`, :doc:`/tutorials/datasets`). This class does not
    intend to satisfy everyone's preference on drawing styles.

    This visualizer focuses on high rendering quality rather than performance. It is not
    designed to be used for real-time applications.

    Args:
        image (Union[torch.Tensor, numpy.ndarray]): Tensor of shape (C x H x W) or ndarray of
            shape (H x W x C) with dtype uint8.
        instance_mode (ColorMode): defines one of the pre-defined style for drawing
            instances on an image.
    """

    def __init__(
        self,
        image: Union[Tensor, np.ndarray],
        scale: float = 1.0,
        line_width: Optional[int] = None,
    ):

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

        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")
        self.line_width = line_width or max(round(sum(self.img.shape) / 2 * 0.003), 2)

    @torch.no_grad()
    def draw_bounding_boxes(
        self,
        boxes: torch.Tensor,
        labels: Optional[List[str]] = None,
        colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
        txt_colors: Tuple[int, int, int] = (255, 255, 255),
    ) -> torch.Tensor:
        """
        Draws bounding boxes on given image.
        The values of the input image should be uint8 between 0 and 255.
        If fill is True, Resulting Tensor should be saved as PNG image.

        Args:
            boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax)
                format. Note that the boxes are absolute coordinates with respect to the image. In other
                words: `0 <= xmin < xmax < W` and `0 <= ymin < ymax < H`.
            labels (List[str]): List containing the labels of bounding boxes.
            colors (color or list of colors, optional): List containing the colors
                of the boxes or single color for all boxes. The color can be represented as
                PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
                By default, random colors are generated for boxes.
            fill (bool): If `True` fills the bounding box with specified color.
            width (int): Width of bounding box.
            font (str): A filename containing a TrueType font. If the file is not found in this filename,
                the loader may also search in other directories, such as the `fonts/` directory on Windows
                or `/Library/Fonts/`, `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
            font_size (int): The requested font size in points.

        Returns:
            img (Tensor[C, H, W]): Image Tensor of dtype uint8 with bounding boxes plotted.
        """

        p1, p2 = (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3]))
        cv2.rectangle(self.img, p1, p2, colors, thickness=self.line_width, lineType=cv2.LINE_AA)
        if labels:
            tf = max(self.line_width - 1, 1)  # font thickness
            w, h = cv2.getTextSize(labels, 0, fontScale=self.line_width / 3, thickness=tf)[0]
            outside = p1[1] - h - 3 >= 0  # labels fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.img, p1, p2, colors, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                self.img,
                labels,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                self.line_width / 3,
                txt_colors,
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
        return self.img
