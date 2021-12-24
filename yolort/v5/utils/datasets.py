# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""
import glob
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .augmentations import letterbox


# Parameters

# acceptable image suffixes
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
# acceptable video suffixes
VID_FORMATS = ["mov", "avi", "mp4", "mpg", "mpeg", "m4v", "wmv", "mkv"]
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))  # DPP


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace exif_transpose() version of
    https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py

    Args:
        image: The image to transpose.

    Return:
        An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


class LoadImages:
    """
    YOLOv5 image/video dataloader. And we're using th CHW RGB format.
    """

    def __init__(self, path: str, img_size: int = 640, stride: int = 32, auto: bool = True):
        path_source = str(Path(path).resolve())  # os-agnostic absolute path
        if "*" in path_source:
            files = sorted(glob.glob(path_source, recursive=True))  # glob
        elif os.path.isdir(path_source):
            files = sorted(glob.glob(os.path.join(path_source, "*.*")))  # dir
        elif os.path.isfile(path_source):
            files = [path_source]  # files
        else:
            raise Exception(f"ERROR: {path_source} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        num_images, num_videos = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.num_files = num_images + num_videos  # number of files
        self.video_flag = [False] * num_images + [True] * num_videos
        self.mode = "image"
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.num_files > 0, (
            f"No images or videos found in {path_source}. Supported formats "
            f"are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.num_files:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            ret_val, img_origin = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.num_files:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img_origin = self.cap.read()

            self.frame += 1
            source_bar = f"video {self.count + 1}/{self.num_files} ({self.frame}/{self.frames}) {path}: "

        else:
            # Read image
            self.count += 1
            img_origin = cv2.imread(path)  # opencv set the BGR order as the default
            assert img_origin is not None, f"Not Found Image: {path}"
            source_bar = f"image {self.count}/{self.num_files} {path}: "

        # Padded resize
        img = letterbox(img_origin, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert HWC to CHW, BGR to RGB
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        return path, img, img_origin, self.cap, source_bar

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.num_files  # number of files
