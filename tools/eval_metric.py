from pathlib import Path
import io
import contextlib
import argparse
import torch

import yolort

from yolort.data import COCOEvaluator
from yolort.data.coco import COCODetection
from yolort.data.transforms import default_val_transforms, collate_fn
from yolort.data import _helper as data_helper


def get_parser():
    parser = argparse.ArgumentParser('Evaluation CLI for yolort', add_help=True)

    parser.add_argument('--num_gpus', default=0, type=int, metavar='N',
                        help='Number of gpu utilizing (default: 0)')
    # Model architecture
    parser.add_argument('--arch', default='yolov5s',
                        help='Model structure to train')
    parser.add_argument('--num_classes', default=80, type=int,
                        help='Classes number of the model')
    parser.add_argument('--image_size', default=640, type=int,
                        help='Image size for evaluation (default: 640)')
    parser.add_argument('--score_thresh', default=0.005, type=float,
                        help='score threshold for mAP evaluation')
    # Dataset Configuration
    parser.add_argument('--image_path', default='./data-bin/coco128/images/train2017',
                        help='Root path of the dataset containing images')
    parser.add_argument('--annotation_path', default=None,
                        help='Path of the annotation file')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N',
                        help='Number of data loading workers (default: 8)')

    parser.add_argument('--output_dir', default='.',
                        help='Path where to save')
    return parser


def prepare_data(image_root, annotation_file, batch_size, num_workers):
    """
    Setup the coco dataset and dataloader for validation
    """
    # Define the dataloader
    data_set = COCODetection(image_root, annotation_file, default_val_transforms())

    # We adopt the sequential sampler in order to repeat the experiment
    sampler = torch.utils.data.SequentialSampler(data_set)

    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return data_set, data_loader


def eval_metric(args):
    if args.num_gpus == 0:
        device = torch.device('cpu')
        print('Set CPU mode.')
    elif args.num_gpus == 1:
        device = torch.device('cuda')
        print('Set GPU mode.')
    else:
        raise NotImplementedError('Currently not supported multi-GPUs mode')

    # Prepare the dataset and dataloader for evaluation
    image_path = Path(args.image_path)
    annotation_path = Path(args.annotation_path)

    print('Loading annotations into memory...')
    with contextlib.redirect_stdout(io.StringIO()):
        data_set, data_loader = prepare_data(
            image_path,
            annotation_path,
            args.batch_size,
            args.num_workers,
        )

    coco_gt = data_helper.get_coco_api_from_dataset(data_set)
    coco_evaluator = COCOEvaluator(coco_gt)

    # Model Definition and Initialization
    model = yolort.models.__dict__[args.arch](
        pretrained=True,
        size=(args.image_size, args.image_size),
        num_classes=args.num_classes,
        score_thresh=args.score_thresh,
    )

    model = model.eval()
    model = model.to(device)

    # COCO evaluation
    print('Computing the mAP...')
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            preds = model(images)
            coco_evaluator.update(preds, targets)

    results = coco_evaluator.compute()

    # Format the results
    # coco_evaluator.derive_coco_results()

    # mAP results
    print(f"The evaluated mAP 0.5:095 is {results['AP']:0.3f}, "
          f"and mAP 0.5 is {results['AP50']:0.3f}.")


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    print(f'Command Line Args: {args}')
    eval_metric(args)


if __name__ == "__main__":
    cli_main()
