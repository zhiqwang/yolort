# Copyright (c) 2021, yolort team. All rights reserved.
#
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from pathlib import Path
import cv2

from typing import List, Optional, Tuple
import torch
from torchvision.ops import box_convert

from yolort.runtime import PredictorTRT
from yolort.utils.image_utils import to_numpy
from yolort.v5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages
from yolort.v5.utils.general import (
    colorstr,
    set_logging,
    increment_path,
    check_img_size,
    check_file,
    scale_coords,
    strip_optimizer,
)
from yolort.v5.utils.plots import Annotator, colors, save_one_box
from yolort.v5.utils.torch_utils import select_device, time_sync

logger = set_logging(__name__)


@torch.no_grad()
def run(
    weights: str = "yolort.engine",
    source: str = "bus.jpg",
    img_size: Tuple[int, int] = (640, 640),
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 1000,
    device: str = "",
    view_img: bool = False,
    save_txt: bool = False,
    save_conf: bool = False,
    save_crop: bool = False,
    nosave: bool = False,
    classes: Optional[List] = None,
    visualize: bool = False,
    update: bool = False,
    project: str = "./runs/detect",
    name: str = "exp",
    exist_ok: bool = False,
    line_thickness=3,
    hide_labels: bool = False,
    hide_conf: bool = False,
    half: bool = False,
):
    """
    The core function for detecting source of image, path or directory.

    Args:
        weights: Path of the engine
        source: file/dir/URL/glob, 0 for webcam
        img_size: inference size (height, width)
        conf_thres: confidence threshold
        iou_thres: NMS IOU threshold
        max_det: maximum detections per image
        device: cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img: show results
        save_txt: save results to *.txt
        save_conf: save confidences in --save-txt labels
        save_crop: save cropped prediction boxes
        nosave: do not save images/videos
        classes: filter by class: --class 0, or --class 0 2 3
        visualize: visualize features
        update: update all models
        project: save results to project/name
        name: save results to project/name
        exist_ok: existing project/name ok, do not increment
        line_thickness: bounding box thickness (pixels)
        hide_labels: hide labels
        hide_conf: hide confidences
        half: use FP16 half-precision inference
    """
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    # increment run
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    # make dir
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = PredictorTRT(
        weights,
        device=device,
        score_thresh=conf_thres,
        iou_thresh=iou_thres,
        detections_per_img=max_det,
    )
    stride, names = model.stride, model.names
    img_size = check_img_size(img_size, stride=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=img_size, stride=stride, auto=False)

    # Run inference
    model.warmup(img_size=(1, 3, *img_size), half=half)
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, _, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred_logits = model(im)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        detections = model.postprocessing(pred_logits)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(detections):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                boxes = scale_coords(im.shape[2:], det["boxes"], im0.shape).round()
                scores = det["scores"]
                labels = det["labels"]

                # Print results
                for c in labels.unique():
                    n = (labels == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for box, score, class_idx in zip(boxes, scores, labels):
                    if save_txt:  # Write to file
                        # normalized xywh
                        xywh = box_convert(torch.tensor(box).view(1, 4), in_fmt="xyxy", out_fmt="cxcywh")
                        xywh = (xywh / gn).view(-1).tolist()
                        line = (class_idx, *xywh, score) if save_conf else (class_idx, *xywh)  # label format
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    xyxy = to_numpy(box)
                    if save_img or save_crop or view_img:  # Add bbox to image
                        cls = int(class_idx)  # integer class
                        label = None if hide_labels else (names[cls] if hide_conf else f"{names[cls]} {score:.2f}")
                        annotator.box_label(xyxy, label, color=colors(cls, True))
                        if save_crop:
                            save_path = save_dir / "crops" / names[cls] / f"{p.stem}.jpg"
                            save_one_box(xyxy, imc, file=save_path, BGR=True)

            # Print time (inference-only)
            logger.info(f"{s}Done. ({t3 - t2:.3f}s)")

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:
                    raise NotImplementedError("Currently this method hasn't implemented yet.")

    # Print results
    speeds_info = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    logger.info(
        f"Speed: {speeds_info[0]:.1f}ms pre-process, {speeds_info[1]:.1f}ms inference, "
        f"{speeds_info[2]:.1f}ms NMS per image at shape {(1, 3, *img_size)}",
    )
    if save_txt or save_img:
        saved_info = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels "
            f"saved to {save_dir / 'labels'}" if save_txt else "",
        )
        logger.info(f"Results saved to {colorstr('bold', save_dir)}{saved_info}")
    if update:
        # update model (to fix SourceChangeWarning)
        strip_optimizer(weights)


def get_parser():
    parser = argparse.ArgumentParser("CLI tool for detecting source.", add_help=True)
    parser.add_argument("--weights", type=str, default="yolov5s.pt", help="model path(s)")
    parser.add_argument("--source", type=str, default="data/images", help="file/dir/URL/glob, 0 for webcam")
    parser.add_argument("--img_size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max_det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view_img", action="store_true", help="show results")
    parser.add_argument("--save_txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save_conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save_crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default="./runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist_ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line_thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide_labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide_conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")

    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    logger.info(f"Command Line Args: {args}")
    run(**vars(args))


if __name__ == "__main__":
    cli_main()
