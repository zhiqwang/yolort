# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import os
import copy
import contextlib

import numpy as np

from torchvision.ops import box_convert

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from torchmetrics import Metric

from ._utils import all_gather

from typing import List, Any, Callable, Optional


class COCOEvaluator(Metric):
    """
    COCO evaluator that works in distributed mode.
    """
    def __init__(
        self,
        coco_gt: COCO,
        iou_types: List[str] = ['bbox'],
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            self.create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def update(self, preds, targets):
        records = {target['image_id'].item(): prediction for target, prediction in zip(targets, preds)}
        img_ids = list(np.unique(list(records.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(records, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    self.coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()

            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = self.coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def compute(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        else:
            raise ValueError(f"Unknown iou type {iou_type}, fell free to report on GitHub issues")

    def coco80_to_coco91_class(self):  # converts 80-index (val2014) to 91-index (paper)
        # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
        # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
        # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
        # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
             35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
             64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        return x

    def prepare_for_coco_detection(self, predictions):
        coco91class = self.coco80_to_coco91_class()

        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = box_convert(boxes, in_fmt='xyxy', out_fmt='xywh').tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": coco91class[labels[k]],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def merge(self, img_ids, eval_imgs):
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

    def create_common_coco_eval(self, coco_eval, img_ids, eval_imgs):
        img_ids, eval_imgs = self.merge(img_ids, eval_imgs)
        img_ids = list(img_ids)
        eval_imgs = list(eval_imgs.flatten())

        coco_eval.evalImgs = eval_imgs
        coco_eval.params.imgIds = img_ids
        coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(self):
    '''
    From pycocotools, just removed the prints and fixed a Python3 bug about unicode
    not defined. Mostly copy-paste from
    <https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py>

    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()  # bottleneck

    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
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
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs
