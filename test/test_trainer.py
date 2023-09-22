# Copyright (c) 2021, yolort team. All rights reserved.

import argparse
import importlib

import sys

sys.path.append("../yolort/")


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default="yolov5n")
    parser.add_argument("-n", "--name", type=str, default="yolov5n", help="model name")

    # distributed
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics. \
        Implemented loggers include `tensorboard` and `wandb`.",
        default="tensorboard",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def test_training_step():
    args = make_parser().parse_args()
    module_name = ".".join(["yolort", "exp", "default", args.name])
    exp = importlib.import_module(module_name).Exp()
    exp.merge(args.opts)
    h, w = exp.input_size
    assert h % 32 == 0 and w % 32 == 0, "input size must be multiples of 32"

    from yolort.trainer import Trainer

    trainer = Trainer(exp, args)
    trainer.train()


def test_test_epoch_end():
    args = make_parser().parse_args()
    module_name = ".".join(["yolort", "exp", "default", args.name])
    exp = importlib.import_module(module_name).Exp()
    exp.merge(args.opts)

    main(exp, args)
