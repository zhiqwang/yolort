from pathlib import Path
import requests
from io import BytesIO

import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image

import numpy as np
import cv2

import torch
from torch import Tensor

from torchvision.ops.boxes import box_convert, box_iou
import torchvision
import time

from typing import Optional

import logging

logger = logging.getLogger(__name__)


def plot_one_box(box, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    COLORS = color_list()  # list of COLORS
    color = color or COLORS[np.random.randint(0, len(COLORS))]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img, label, (c1[0], c1[1] - 2), 0, tl / 3,
            [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA,
        )

    return img


def cv2_imshow(
    image: np.ndarray,
    imshow_scale: Optional[float] = None,
    convert_bgr_to_rgb: bool = True,
) -> None:
    """
    A replacement of cv2.imshow() for using in Jupyter notebooks.

    Args:
        image (np.ndarray):. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape (N, M, 3)
            is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color image.
        imshow_scale (Optional[float]): zoom ratio to show the image
        convert_bgr_to_rgb (bool): switch to convert BGR to RGB channel.
    """
    image = image.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if convert_bgr_to_rgb and image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if imshow_scale is not None:
        image = cv2.resize(image, None, fx=imshow_scale, fy=imshow_scale)

    display(Image.fromarray(image))


def color_list():
    # Return first 10 plt colors as (r,g,b)
    # Refer to <https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb>
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in plt.rcParams['axes.prop_cycle'].by_key()['color']]


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
):
    # Resize image to a 32-pixel-multiple rectangle
    # Ref: <https://github.com/ultralytics/yolov3/issues/232>
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def scale_coords(coords, img_shape, img_shape_origin, ratio_pad=None):
    # Rescale coords (xyxy) from img_shape to img_shape_origin
    if ratio_pad is None:  # calculate from img_shape_origin
        gain = min(img_shape[0] / img_shape_origin[0],
                   img_shape[1] / img_shape_origin[1])  # gain  = old / new
        pad = ((img_shape[1] - img_shape_origin[1] * gain) / 2,
               (img_shape[0] - img_shape_origin[0] * gain) / 2)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    # Clip bounding xyxy bounding boxes to image shape (height, width)
    coords[:, 0].clamp_(0, img_shape_origin[1])  # x1
    coords[:, 1].clamp_(0, img_shape_origin[0])  # y1
    coords[:, 2].clamp_(0, img_shape_origin[1])  # x2
    coords[:, 3].clamp_(0, img_shape_origin[0])  # y2
    return coords


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None,
                        agnostic=False, multi_label=False, labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, (
        f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0')
    assert 0 <= iou_thres <= 1, (
        f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0')

    # Settings
    max_wh = 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            label = labels[xi]
            v = torch.zeros((len(label), nc + 5), device=x.device)
            v[:, :4] = label[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(label)), label[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = box_convert(x[:, :4], in_fmt="cxcywh", out_fmt="xyxy")

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            # merging boxes
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            logger.warning(f'NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def get_image_from_url(
    url: str,
    flags: int = cv2.IMREAD_COLOR,
) -> np.ndarray:
    """
    Generates an image directly from an URL

    Args:
        url (str): the url address of an image.
        flags (int): Flag that can take values of cv::ImreadModes, which has the following
            constants predefined in cv2. Default: cv2.IMREAD_COLOR.
            - cv2.IMREAD_COLOR, always convert image to the 3 channel BGR color image.
            - cv2.IMREAD_GRAYSCALE, always convert image to the single channel grayscale
                image (codec internal conversion).
    """
    data = requests.get(url)
    buffer = BytesIO(data.content)
    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
    image = cv2.imdecode(bytes_as_np_array, flags)
    return image


def read_image_to_tensor(
    image: np.ndarray,
    is_half: bool = False,
) -> Tensor:
    """
    Parse an image to Tensor.

    Args:
        image (np.ndarray): the candidate ndarray image to be parsed to Tensor.
        is_half (bool): whether to transfer image to half. Default: False.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.ascontiguousarray(image, dtype=np.float32)  # uint8 to float32
    image = np.transpose(image / 255.0, [2, 0, 1])

    image = torch.from_numpy(image)
    image = image.half() if is_half else image.float()
    return image


def load_names(category_path):
    names = []
    with open(category_path, 'r') as f:
        for line in f:
            names.append(line.strip())
    return names


@torch.no_grad()
def overlay_boxes(detections, path, time_consume, args):

    img = cv2.imread(path) if args.save_img else None

    for i, pred in enumerate(detections):  # detections per image
        det_logs = ''
        save_path = Path(args.output_dir).joinpath(Path(path).name)
        txt_path = Path(args.output_dir).joinpath(Path(path).stem)

        if pred is not None and len(pred) > 0:
            # Rescale boxes from img_size to im0 size
            boxes, scores, labels = pred['boxes'].round(), pred['scores'], pred['labels']

            # Print results
            for c in labels.unique():
                n = (labels == c).sum()  # detections per class
                det_logs += '%g %ss, ' % (n, args.names[int(c)])  # add to string

            # Write results
            for xyxy, conf, cls_name in zip(boxes, scores, labels):
                if args.save_txt:  # Write to file
                    # normalized cxcywh
                    cxcywh = box_convert(xyxy, in_fmt="xyxy", out_fmt="cxcywh").tolist()
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * 5 + '\n') % (cls_name, *cxcywh))  # label format

                if args.save_img:  # Add bbox to image
                    label = '%s %.2f' % (args.names[int(cls_name)], conf)
                    plot_one_box(
                        xyxy,
                        img,
                        label=label,
                        color=args.colors[int(cls_name) % len(args.colors)],
                        line_thickness=3,
                    )

        # Print inference time
        logger.info('%sDone. (%.3fs)' % (det_logs, time_consume))

        # Save results (image with detections)
        if args.save_img:
            cv2.imwrite(str(save_path), img)

    return (boxes.tolist(), scores.tolist(), labels.tolist())


def box_cxcywh_to_xyxy(bbox):
    y = np.zeros_like(bbox)
    y[:, 0] = bbox[:, 0] - 0.5 * bbox[:, 2]  # top left x
    y[:, 1] = bbox[:, 1] - 0.5 * bbox[:, 3]  # top left y
    y[:, 2] = bbox[:, 0] + 0.5 * bbox[:, 2]  # bottom right x
    y[:, 3] = bbox[:, 1] + 0.5 * bbox[:, 3]  # bottom right y
    return y


def cast_image_tensor_to_numpy(images):
    """
    Cast image from torch.Tensor to opencv
    """
    images = to_numpy(images).copy()
    images = images * 255
    images = images.clip(0, 255).astype('uint8')
    return images


def parse_images(images):
    images = images.permute(0, 2, 3, 1)
    images = cast_image_tensor_to_numpy(images)

    return images


def parse_single_image(image):
    image = image.permute(1, 2, 0)
    image = cast_image_tensor_to_numpy(image)
    return image


def parse_single_target(target):
    boxes = box_convert(target['boxes'], in_fmt="cxcywh", out_fmt="xyxy")
    boxes = to_numpy(boxes)
    sizes = np.tile(to_numpy(target['size'])[1::-1], 2)
    boxes = boxes * sizes
    return boxes


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def restore_label(target, feature_map_sizes, image_sizes):
    h, w = image_sizes
    target = target / np.array(feature_map_sizes)[[3, 2, 3, 2]]
    target = target * np.array([w, h, w, h], np.float32)
    target = box_cxcywh_to_xyxy(target)
    return target


def restore_anchor(anchor, grid_x, grid_y, stride, feature_map_size, image_sizes):
    anchor *= stride
    h, w = image_sizes
    anchor_restored = np.stack([grid_y, grid_x], axis=1)

    anchor_restored = anchor_restored / np.array(feature_map_size, np.float)[[3, 2]]
    anchor_restored = anchor_restored * np.array([w, h], np.float32)
    anchor_restored = np.concatenate([anchor_restored, anchor], axis=1)
    anchor_restored = box_cxcywh_to_xyxy(anchor_restored)

    return anchor_restored


def anchor_match_visualize(images, targets, indices, anchors, pred):
    # Modified from <https://github.com/hhaAndroid/yolov5-comment/blob/e018889b/utils/general.py#L714>
    image_sizes = images.shape[-2:]
    images = parse_images(images)

    strdie = [8, 16, 32]

    images_with_anchor = []
    # for loop all images
    for j in range(images.shape[0]):
        image = images[j].astype(np.uint8)[..., ::-1]

        # Visualize each prediction scale individually
        vis_all_scale_images = []
        # i = 0 is associate with small objects,
        # i = 1 is associate with intermediate objects,
        # i = 2 is associate with large objects
        for i in range(3):
            image_per_scale = image.copy()
            stride = strdie[i]
            # anchor scale
            b, _, grid_x, grid_y = indices[i]

            b, grid_x, grid_y, anchor, target = map(to_numpy, [b, grid_x, grid_y, anchors[i], targets[i]])

            # Find out the corresponding branch of one image
            idx = b == j
            grid_x, grid_y, anchor, target = grid_x[idx], grid_y[idx], anchor[idx], target[idx]

            # Restore to the original image scale for visualization
            target = restore_label(target, pred[i].shape, image_sizes)

            # visualize labels
            image_per_scale = overlay_bbox(
                image_per_scale,
                target,
                color=(0, 0, 255),
                thickness=3,
                with_mask=False,
            )

            # The anchors need to restore the offset.
            # In eacy layer there has at most 3x3=9 anchors for matching.
            anchor_restored = restore_anchor(anchor, grid_x, grid_y, stride, pred[i].shape, image_sizes)

            # visualize positive anchor
            image_per_scale = overlay_bbox(image_per_scale, anchor_restored)
            vis_all_scale_images.append(image_per_scale)

        per_image_with_anchor = merge_images_with_boundary(vis_all_scale_images)
        images_with_anchor.append(per_image_with_anchor)

    return images_with_anchor


def overlay_bbox(image, bboxs_list, color=None, thickness=2, font_scale=0.3, with_mask=False):
    """
    Visualize bbox in object detection by drawing rectangle.

    Args:
        image: numpy.ndarray.
        bboxs_list: list: [pts_xyxy, prob, id]: label or prediction.
        color: tuple.
        thickness: int.
        font_scale: float.
        with_mask: draw as mask.

    Return:
        numpy.ndarray
    """
    assert image is not None
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt = ''
    COLORS = color_list()  # list of COLORS

    for bbox in bboxs_list:
        if len(bbox) == 5:
            txt = '{:.3f}'.format(bbox[4])
        elif len(bbox) == 6:
            txt = 'p={:.3f}, id={:.3f}'.format(bbox[4], bbox[5])
        bbox_f = np.array(bbox[:4], np.int32)

        mask = np.zeros_like(image, np.uint8)
        mask = cv2.rectangle(
            mask if with_mask else image,
            (bbox_f[0], bbox_f[1]),
            (bbox_f[2], bbox_f[3]),
            color=(color or COLORS[np.random.randint(0, len(COLORS))]),
            thickness=(-1 if with_mask else thickness),
        )
        if with_mask:
            image = cv2.addWeighted(image, 1.0, mask, 0.6, 0)

        if len(bbox) == 5 or len(bbox) == 6:
            cv2.putText(
                image,
                txt,
                (bbox_f[0], bbox_f[1] - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    return image


def merge_images_with_boundary(images_list, row_col_num=(1, -1)):
    """
    To merge a list of images.

    Additionally, parameter ``row_col_num`` supports user specified merge format.
    Notice, specified format must be greater than or equal to images number.

    Args:
        image_lists: list of np.ndarray.
        row_col_num: merge format. default is (1, -1), image will line up to show.
            when set example=(2, 5), images will display in two rows and five columns.
    """
    if not isinstance(images_list, list):
        images_list = [images_list]

    images_merged = merge_images(images_list, row_col_num)
    return images_merged


def merge_images(images_list, row_col_num):
    """
    Merges all input images as an image with specified merge format.

    Args:
        images_list: images list
        row_col_num: number of rows and columns displayed
    Return:
        image: merges image
    """

    num_images = len(images_list)
    row, col = row_col_num

    assert row > 0 or col > 0, 'row and col cannot be negative at same time!'

    for image in images_list:
        cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (255, 0, 255))

    if row_col_num[1] < 0 or num_images < row:
        images_merged = np.hstack(images_list)
    elif row_col_num[0] < 0 or num_images < col:
        images_merged = np.vstack(images_list)
    else:
        assert row * col >= num_images, 'Images overboundary, not enough windows to display all images!'

        fill_img_list = [np.zeros(images_list[0].shape, dtype=np.uint8)] * (row * col - num_images)
        images_list.extend(fill_img_list)
        merge_imgs_col = []
        for i in range(row):
            start = col * i
            end = col * (i + 1)
            merge_col = np.hstack(images_list[start: end])
            merge_imgs_col.append(merge_col)

        images_merged = np.vstack(merge_imgs_col)

    return images_merged
