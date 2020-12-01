import xml.etree.ElementTree as ET
import os
import shutil

import numpy as np


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        # obj_struct['pose'] = obj.find('pose').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [
            int(bbox.find('xmin').text),
            int(bbox.find('ymin').text),
            int(bbox.find('xmax').text),
            int(bbox.find('ymax').text),
        ]
        objects.append(obj_struct)
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    """
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # unions
            area_bb = (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
            area_BBGT = (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)

            unions = area_bb + area_BBGT - inters

            overlaps = inters / unions
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def _write_voc_results_file(all_boxes, image_index, cls_names, output_dir):
    output_path = os.path.join(output_dir, 'results')
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    print('Writing results file, wait a moment...')
    for cls_ind, cls_name in enumerate(cls_names):
        # DistributeSampler happens to clone the inputs to make the task
        # lenghts even among the nodes:
        # https://github.com/pytorch/pytorch/issues/22584
        # Boxes can be duplicated in the process since multiple
        # evaluation of the same image can happen, multiple boxes in the
        # same location decrease the final mAP, later in the code we discard
        # repeated image_index thanks to the sorting
        new_image_index, all_boxes[cls_ind] = zip(
            *sorted(zip(image_index, all_boxes[cls_ind]), key=lambda x: x[0]),
        )
        if cls_name == '__background__':
            continue

        filename = os.path.join(output_path, 'det_test_{:s}.txt'.format(cls_name))
        with open(filename, 'wt') as f:
            prev_index = ''
            for im_ind, index in enumerate(new_image_index):
                # check for repeated input and discard
                if prev_index == index:
                    continue
                prev_index = index
                dets = all_boxes[cls_ind][im_ind]

                if dets == []:
                    continue

                # the VOCdevkit expects 1-based indices
                for k in range(len(dets)):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                        index,
                        dets[k][-1],
                        dets[k][0] + 1,
                        dets[k][1] + 1,
                        dets[k][2] + 1,
                        dets[k][3] + 1,
                    ))


def _do_python_eval(data_loader, output_dir, use_07=True):
    voc_root = os.path.join(data_loader.dataset.root, 'VOCdevkit', 'VOC2007')
    image_set = data_loader.dataset.image_set
    imagesetfile = os.path.join(voc_root, 'ImageSets', 'Main', '{}.txt'.format(image_set))
    annopath = os.path.join(voc_root, 'Annotations/{:s}.xml')

    cls_names = data_loader.dataset.prepare.CLASSES
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

    for cls_name in cls_names:
        if cls_name == '__background__':
            continue
        detpath = os.path.join(output_dir, 'results', 'det_test_{:s}.txt')
        rec, prec, ap = voc_eval(
            detpath, annopath, imagesetfile, cls_name,
            ovthresh=0.5, use_07_metric=use_07_metric,
        )
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls_name, ap))

    print('Mean AP = {:.4f}'.format(np.mean(aps)))
