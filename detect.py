import os
import time
from pathlib import Path

from numpy import random
import cv2
import torch

from utils.datasets import LoadImages
from utils.general import check_img_size, scale_coords, box_xyxy_to_cxcywh, plot_one_box, set_logging
from utils.torch_utils import time_synchronized

from hubconf import yolov5


def get_coco_names(category_path):
    names = []
    with open(category_path, 'r') as f:
        for line in f:
            names.append(line.strip())
    return names


@torch.no_grad()
def inference(model, img, is_half):
    model.eval()

    img = img.half() if is_half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    model_out = model(img)
    t2 = time_synchronized()
    time_consume = t2 - t1
    return model_out, time_consume


def main(args):
    print(args)

    device = torch.device(args.device)

    model = yolov5(cfg_path=args.model_cfg, checkpoint_path=args.model_checkpoint)
    model.eval()
    model = model.to(device)

    webcam = (args.image_source.isnumeric() or args.image_source.startswith(
        ('rtsp://', 'rtmp://', 'http://')) or args.image_source.endswith('.txt'))

    # Initialize
    set_logging()

    # half = device.type != 'cpu'  # half precision only supported on CUDA
    is_half = False

    # Load model
    imgsz = check_img_size(args.img_size, s=model.box_head.stride.max())  # check img_size
    if is_half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(args.image_source, img_size=imgsz)

    # Get names and colors
    names = get_coco_names(Path(args.coco_category_path))
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if is_half else img) if device.type != 'cpu' else None  # run once

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        model_out, time_consume = inference(model, img, is_half)

        # Process detections
        for i, det in enumerate(model_out):  # detections per image

            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = os.path.join(args.output_dir, Path(p).name)
            txt_postfix = f"{Path(p).stem}_{dataset.fram if dataset.mode == 'video' else ''}"
            txt_path = os.path.join(args.output_dir, txt_postfix)
            s += '%gx%g ' % img.shape[-2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[-2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if args.save_txt:  # Write to file
                        # normalized xywh
                        xywh = (box_xyxy_to_cxcywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if args.save_img or args.view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print inference time
            print('%sDone. (%.3fs)' % (s, time_consume))

            # Save results (image with detections)
            if args.save_img and dataset.mode == 'images':
                cv2.imwrite(save_path, im0)

    if args.save_txt or args.save_img:
        print(f'Results saved to {Path(args.output_dir)}')

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--model-cfg', type=str, default='./models/yolov5s.yaml',
                        help='path where the model cfg in')
    parser.add_argument('--model-checkpoint', type=str, default='./checkpoints/yolov5/yolov5s.pt',
                        help='path where the model checkpoint in')
    parser.add_argument('--coco-category-path', type=str, default='./libtorch_inference/weights/coco.names',
                        help='path where the coco category in')
    parser.add_argument('--image-source', type=str, default='./.github/',
                        help='path where the source images in')
    parser.add_argument('--output-dir', type=str, default='./data-bin/output',
                        help='path where to save')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4,
                        help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5,
                        help='IOU threshold for NMS')
    parser.add_argument('--device', default='cuda',
                        help='device')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-img', action='store_true',
                        help='save image inference results')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
