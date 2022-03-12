# Copyright (c) 2021, yolort team. All rights reserved.

import contextlib
import copy
import io
import itertools
import logging
from pathlib import PosixPath

import numpy as np
from tabulate import tabulate
from torchvision.ops import box_convert

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from torchmetrics import Metric
except ImportError:
    COCO, COCOeval, Metric = None, None, None

from typing import Any, List, Callable, Optional, Union

from ._helper import create_small_table
from .distributed import all_gather


class COCOEvaluator(Metric):
    """
    Evaluate AP for instance detection using COCO's metrics that works in distributed mode.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).
    """

    def __init__(
        self,
        coco_gt: Union[str, PosixPath, COCO],
        iou_type: str = "bbox",
        eval_type: str = "yolov5",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        """
        Args:
            coco_gt (Union[str, PosixPath, COCO]): a json file in COCO's format or a COCO api
                - str: a json file in COCO's result format.
                - PosixPath: a json file in COCO's result format, and is wrapped with Path.
                - COCO: COCO api
            iou_type (str): iou type to compute.
            eval_type (str): The categories predicted by yolov5 are continuous [1-80], which is
                different from torchvision's discrete 91 categories. Default: yolov5.
        """
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self._logger = logging.getLogger(__name__)
        if isinstance(coco_gt, str) or isinstance(coco_gt, PosixPath):
            with contextlib.redirect_stdout(io.StringIO()):
                coco_gt = COCO(coco_gt)
        elif isinstance(coco_gt, COCO):
            coco_gt = copy.deepcopy(coco_gt)
        else:
            raise NotImplementedError(f"Currently not supports type {type(coco_gt)}")

        self.coco_gt = coco_gt
        if eval_type == "yolov5":
            self.category_id_maps = coco_gt.getCatIds()
        elif eval_type == "torchvision":
            self.category_id_maps = list(range(coco_gt.getCatIds()[-1] + 1))
        else:
            raise NotImplementedError(f"Currently not supports eval type {eval_type}")

        self.iou_type = iou_type
        self.coco_eval = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = []

    def update(self, preds, targets):
        records = {target["image_id"].item(): prediction for target, prediction in zip(targets, preds)}
        img_ids = list(np.unique(list(records.keys())))
        self.img_ids.extend(img_ids)

        results = self.prepare(records, self.iou_type)

        # suppress pycocotools prints
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()

        coco_eval = self.coco_eval

        coco_eval.cocoDt = self.coco_dt
        coco_eval.params.imgIds = list(img_ids)
        img_ids, eval_imgs = evaluate(coco_eval)

        self.eval_imgs.append(eval_imgs)

    def compute(self):
        # suppress pycocotools prints
        with contextlib.redirect_stdout(io.StringIO()):
            # Synchronize between processes
            coco_eval = self.coco_eval
            img_ids = self.img_ids
            eval_imgs = np.concatenate(self.eval_imgs, 2)
            create_common_coco_eval(coco_eval, img_ids, eval_imgs)

            # Accumulate
            coco_eval.accumulate()
            # Summarize
            coco_eval.summarize()

        results = self.derive_coco_results()
        return results

    def derive_coco_results(self, class_names: Optional[List[str]] = None):
        """
        Derive the desired score numbers from summarized COCOeval. Modified from
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/coco_evaluation.py

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[self.iou_type]

        if self.coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(self.coco_eval.stats[idx] * 100 if self.coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(f"Evaluation results for {self.iou_type}:\n" + create_small_table(results))

        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        precisions = self.coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append((f"{name}", float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info(f"Per-category {self.iou_type} AP:\n" + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        else:
            raise ValueError(f"Unknown iou type {iou_type}, fell free to report on GitHub issues")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="xywh").tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": self.category_id_maps[labels[k]],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results


def merge(img_ids, eval_imgs):
    """
    Gather data, copy from
    https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py#L163-L182
    """
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    """
    Synchronize version of coco_eval. Copy from:
    https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py#L185-L192
    """
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(self):
    """
    From pycocotools, just removed the prints and fixed a Python3 bug about unicode
    not defined. Mostly copy-paste from
    https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py#L300

    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    """
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
        print(f"useSegm (deprecated) is not None. Running {p.iouType} evaluation")
    # print(f'Evaluate annotation type *{p.iouType}*')
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()  # bottleneck

    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == "segm" or p.iouType == "bbox":
        computeIoU = self.computeIoU
    elif p.iouType == "keypoints":
        computeIoU = self.computeOks

    self.ious = {
        (imgId, catId): computeIoU(imgId, catId) for imgId in p.imgIds for catId in catIds
    }  # bottleneck

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    return p.imgIds, evalImgs
