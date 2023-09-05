import argparse
import contextlib
import io
import time
from pathlib import Path

import torch
import torchvision
import yolort
from yolort.data import _helper as data_helper
from yolort.data.coco import COCODetection
from yolort.data.coco_eval import COCOEvaluator
from yolort.data.transforms import collate_fn, default_val_transforms
from yolort.utils.logger import MetricLogger


def get_parser():
    parser = argparse.ArgumentParser("Evaluation CLI for yolort", add_help=True)

    parser.add_argument(
        "--num_gpus",
        default=0,
        type=int,
        metavar="N",
        help="Number of gpu utilizing (default: 0)",
    )
    # Model architecture
    parser.add_argument("--arch", default="yolov5s", help="Model structure to train")
    parser.add_argument("--num_classes", default=80, type=int, help="Number of classes")
    parser.add_argument(
        "--image_size",
        default=640,
        type=int,
        help="Image size for evaluation (default: 640)",
    )
    # Dataset Configuration
    parser.add_argument(
        "--image_path",
        default="./data-bin/coco128/images/train2017",
        help="Root path of the dataset containing images",
    )
    parser.add_argument("--annotation_path", default=None, help="Path of the annotation file")
    parser.add_argument(
        "--eval_type",
        default="yolov5",
        help="Type of category id maps, yolov5 use continuous [1, 80], "
        "torchvision use discrete ids range in [1, 90]",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        metavar="N",
        help="Number of data loading workers (default: 8)",
    )

    parser.add_argument(
        "--print_freq",
        default=20,
        type=int,
        help="The frequency of printing the logging",
    )
    parser.add_argument("--output_dir", default=".", help="Path where to save")

    # Weights and Biases arguments
    parser.add_argument("--use_wandb", default=True, type=bool, help="Whether to use W&B for metric logging")
    parser.add_argument("--wandb_project", default="yolov5-rt", type=str, help="Name of the W&B Project")
    parser.add_argument("--wandb_entity", default=None, type=str, help="entity to use for W&B logging")

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
        device = torch.device("cpu")
        print("Set CPU mode.")
    elif args.num_gpus == 1:
        device = torch.device("cuda")
        print("Set GPU mode.")
    else:
        raise NotImplementedError("Currently not supported multi-GPUs mode")

    # Prepare the dataset and dataloader for evaluation
    image_path = Path(args.image_path)
    annotation_path = Path(args.annotation_path)

    print("Loading annotations into memory...")
    with contextlib.redirect_stdout(io.StringIO()):
        data_set, data_loader = prepare_data(
            image_path,
            annotation_path,
            args.batch_size,
            args.num_workers,
        )

    coco_gt = data_helper.get_coco_api_from_dataset(data_set)
    coco_evaluator = COCOEvaluator(coco_gt, eval_type=args.eval_type)

    # Model Definition and Initialization
    if args.eval_type == "yolov5":
        model = yolort.models.__dict__[args.arch](
            pretrained=True,
            num_classes=args.num_classes,
        )
    elif args.eval_type == "torchvision":
        model = torchvision.models.detection.__dict__[args.arch](
            pretrained=True,
            num_classes=args.num_classes,
        )
    else:
        raise NotImplementedError(f"Currently not supports eval type: {args.eval_type}")

    model = model.eval()
    model = model.to(device)

    print("Computing the mAP...")
    results = evaluate(
        model,
        data_loader,
        coco_evaluator,
        device,
        args.print_freq,
        args.use_wandb,
        args.wandb_project,
        args.wandb_entity,
    )

    # mAP results
    print(
        f"The evaluated mAP at 0.50:0.95 is {results['AP']:0.3f}, "
        f"and mAP at 0.50 is {results['AP50']:0.3f}."
    )


@torch.no_grad()
def evaluate(model, data_loader, coco_evaluator, device, print_freq, use_wandb, wandb_project, wandb_entity):
    # COCO evaluation
    metric_logger = MetricLogger(
        delimiter="  ", use_wandb=use_wandb, wandb_project=wandb_project, wandb_entity=wandb_entity
    )
    header = "Test:"
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_time = time.time()
        preds = model(images)
        model_time = time.time() - model_time

        evaluator_time = time.time()
        coco_evaluator.update(preds, targets)
        evaluator_time = time.time() - evaluator_time

        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    results = coco_evaluator.compute()
    return results


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    print(f"Command Line Args: {args}")
    eval_metric(args)


if __name__ == "__main__":
    cli_main()