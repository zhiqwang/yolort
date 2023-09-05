import datetime
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist

import inspect
import os
import sys
from collections import defaultdict
from loguru import logger

import cv2
import numpy as np

# from yolort.utils import is_module_available
#
# if is_module_available("cv2"):
#     import wandb


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t", use_wandb=True, wandb_project="yolov5-rt", wandb_entity=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.use_wandb = use_wandb
        if is_main_process() and self.use_wandb:
            self.wandb_run = wandb.init(project=wandb_project, entity=wandb_entity)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
            if is_main_process() and self.wandb_run:
                wandb.log({k: v})

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth.
        Default value: 0.

    Returns:
        str: module name of the caller
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals["__name__"]


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """

    def __init__(self, level="INFO", caller_names=("apex", "pycocotools")):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
                Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        # flush is related with CPR(cursor position report) in terminal
        return sys.__stdout__.flush()

    def isatty(self):
        # when using colab, jax is installed by default and issue like
        # https://github.com/Megvii-BaseDetection/YOLOX/issues/1437 might be raised
        # due to missing attribute like`isatty`.
        # For more details, checked the following link:
        # https://github.com/google/jax/blob/10720258ea7fb5bde997dfa2f3f71135ab7a6733/jax/_src/pretty_printer.py#L54  # noqa
        return sys.__stdout__.isatty()

    def fileno(self):
        # To solve the issue when using debug tools like pdb
        return sys.__stdout__.fileno()


def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguru(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    # only keep logger in rank0 process
    if distributed_rank == 0:
        logger.add(
            sys.stderr,
            format=loguru_format,
            level="INFO",
            enqueue=True,
        )
        logger.add(save_file)

    # redirect stdout/stderr to loguru
    redirect_sys_output("INFO")


class WandbLogger(object):
    """
    Log training runs, datasets, models, and predictions to Weights & Biases.
    This logger sends information to W&B at wandb.ai.
    By default, this information includes hyperparameters,
    system configuration and metrics, model metrics,
    and basic data metrics and analyses.

    For more information, please refer to:
    https://docs.wandb.ai/guides/track
    https://docs.wandb.ai/guides/integrations/other/yolox
    """
    def __init__(self,
                 project=None,
                 name=None,
                 id=None,
                 entity=None,
                 save_dir=None,
                 config=None,
                 val_dataset=None,
                 num_eval_images=100,
                 log_checkpoints=False,
                 **kwargs):
        """
        Args:
            project (str): wandb project name.
            name (str): wandb run name.
            id (str): wandb run id.
            entity (str): wandb entity name.
            save_dir (str): save directory.
            config (dict): config dict.
            val_dataset (Dataset): validation dataset.
            num_eval_images (int): number of images from the validation set to log.
            log_checkpoints (bool): log checkpoints
            **kwargs: other kwargs.

        Usage:
            Any arguments for wandb.init can be provided on the command line using
            the prefix `wandb-`.
            Example
            ```
            python tools/train.py .... --logger wandb wandb-project <project-name> \
                wandb-name <run-name> \
                wandb-id <run-id> \
                wandb-save_dir <save-dir> \
                wandb-num_eval_imges <num-images> \
                wandb-log_checkpoints <bool>
            ```
            The val_dataset argument is not open to the command line.
        """
        try:
            import wandb
            self.wandb = wandb
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "wandb is not installed."
                "Please install wandb using pip install wandb"
                )

        from yolort.data.datasets import VOCDetection

        self.project = project
        self.name = name
        self.id = id
        self.save_dir = save_dir
        self.config = config
        self.kwargs = kwargs
        self.entity = entity
        self._run = None
        self.val_artifact = None
        if num_eval_images == -1:
            self.num_log_images = len(val_dataset)
        else:
            self.num_log_images = min(num_eval_images, len(val_dataset))
        self.log_checkpoints = (log_checkpoints == "True" or log_checkpoints == "true")
        self._wandb_init = dict(
            project=self.project,
            name=self.name,
            id=self.id,
            entity=self.entity,
            dir=self.save_dir,
            resume="allow"
        )
        self._wandb_init.update(**kwargs)

        _ = self.run

        if self.config:
            self.run.config.update(self.config)
        self.run.define_metric("train/epoch")
        self.run.define_metric("val/*", step_metric="train/epoch")
        self.run.define_metric("train/step")
        self.run.define_metric("train/*", step_metric="train/step")

        self.voc_dataset = VOCDetection

        if val_dataset and self.num_log_images != 0:
            self.val_dataset = val_dataset
            self.cats = val_dataset.cats
            self.id_to_class = {
                cls['id']: cls['name'] for cls in self.cats
            }
            self._log_validation_set(val_dataset)

    @property
    def run(self):
        if self._run is None:
            if self.wandb.run is not None:
                logger.info(
                    "There is a wandb run already in progress "
                    "and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()`"
                    "before instantiating `WandbLogger`."
                )
                self._run = self.wandb.run
            else:
                self._run = self.wandb.init(**self._wandb_init)
        return self._run

    def _log_validation_set(self, val_dataset):
        """
        Log validation set to wandb.

        Args:
            val_dataset (Dataset): validation dataset.
        """
        if self.val_artifact is None:
            self.val_artifact = self.wandb.Artifact(name="validation_images", type="dataset")
            self.val_table = self.wandb.Table(columns=["id", "input"])

            for i in range(self.num_log_images):
                data_point = val_dataset[i]
                img = data_point[0]
                id = data_point[3]
                img = np.transpose(img, (1, 2, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if isinstance(id, torch.Tensor):
                    id = id.item()

                self.val_table.add_data(
                    id,
                    self.wandb.Image(img)
                )

            self.val_artifact.add(self.val_table, "validation_images_table")
            self.run.use_artifact(self.val_artifact)
            self.val_artifact.wait()

    def _convert_prediction_format(self, predictions):
        image_wise_data = defaultdict(int)

        for key, val in predictions.items():
            img_id = key

            try:
                bboxes, cls, scores = val
            except KeyError:
                bboxes, cls, scores = val["bboxes"], val["categories"], val["scores"]

            # These store information of actual bounding boxes i.e. the ones which are not None
            act_box = []
            act_scores = []
            act_cls = []

            if bboxes is not None:
                for box, classes, score in zip(bboxes, cls, scores):
                    if box is None or score is None or classes is None:
                        continue
                    act_box.append(box)
                    act_scores.append(score)
                    act_cls.append(classes)

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in act_box],
                    "scores": [score.numpy().item() for score in act_scores],
                    "categories": [
                        self.val_dataset.class_ids[int(act_cls[ind])]
                        for ind in range(len(act_box))
                    ],
                }
            })

        return image_wise_data

    def log_metrics(self, metrics, step=None):
        """
        Args:
            metrics (dict): metrics dict.
            step (int): step number.
        """

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = v.item()

        if step is not None:
            metrics.update({"train/step": step})
            self.run.log(metrics)
        else:
            self.run.log(metrics)

    def log_images(self, predictions):
        if len(predictions) == 0 or self.val_artifact is None or self.num_log_images == 0:
            return

        table_ref = self.val_artifact.get("validation_images_table")

        columns = ["id", "predicted"]
        for cls in self.cats:
            columns.append(cls["name"])

        if isinstance(self.val_dataset, self.voc_dataset):
            predictions = self._convert_prediction_format(predictions)

        result_table = self.wandb.Table(columns=columns)

        for idx, val in table_ref.iterrows():

            avg_scores = defaultdict(int)
            num_occurrences = defaultdict(int)

            id = val[0]
            if isinstance(id, list):
                id = id[0]

            if id in predictions:
                prediction = predictions[id]
                boxes = []
                for i in range(len(prediction["bboxes"])):
                    bbox = prediction["bboxes"][i]
                    x0 = bbox[0]
                    y0 = bbox[1]
                    x1 = bbox[2]
                    y1 = bbox[3]
                    box = {
                        "position": {
                            "minX": min(x0, x1),
                            "minY": min(y0, y1),
                            "maxX": max(x0, x1),
                            "maxY": max(y0, y1)
                        },
                        "class_id": prediction["categories"][i],
                        "domain": "pixel"
                    }
                    avg_scores[
                        self.id_to_class[prediction["categories"][i]]
                    ] += prediction["scores"][i]
                    num_occurrences[self.id_to_class[prediction["categories"][i]]] += 1
                    boxes.append(box)
            else:
                boxes = []
            average_class_score = []
            for cls in self.cats:
                if cls["name"] not in num_occurrences:
                    score = 0
                else:
                    score = avg_scores[cls["name"]] / num_occurrences[cls["name"]]
                average_class_score.append(score)
            result_table.add_data(
                idx,
                self.wandb.Image(val[1], boxes={
                        "prediction": {
                            "box_data": boxes,
                            "class_labels": self.id_to_class
                        }
                    }
                ),
                *average_class_score
            )

        self.wandb.log({"val_results/result_table": result_table})

    def save_checkpoint(self, save_dir, model_name, is_best, metadata=None):
        """
        Args:
            save_dir (str): save directory.
            model_name (str): model name.
            is_best (bool): whether the model is the best model.
            metadata (dict): metadata to save corresponding to the checkpoint.
        """

        if not self.log_checkpoints:
            return

        if "epoch" in metadata:
            epoch = metadata["epoch"]
        else:
            epoch = None

        filename = os.path.join(save_dir, model_name + "_ckpt.pth")
        artifact = self.wandb.Artifact(
            name=f"run_{self.run.id}_model",
            type="model",
            metadata=metadata
        )
        artifact.add_file(filename, name="model_ckpt.pth")

        aliases = ["latest"]

        if is_best:
            aliases.append("best")

        if epoch:
            aliases.append(f"epoch-{epoch}")

        self.run.log_artifact(artifact, aliases=aliases)

    def finish(self):
        self.run.finish()

    @classmethod
    def initialize_wandb_logger(cls, args, exp, val_dataset):
        wandb_params = dict()
        prefix = "wandb-"
        for k, v in zip(args.opts[0::2], args.opts[1::2]):
            if k.startswith("wandb-"):
                try:
                    wandb_params.update({k[len(prefix):]: int(v)})
                except ValueError:
                    wandb_params.update({k[len(prefix):]: v})

        return cls(config=vars(exp), val_dataset=val_dataset, **wandb_params)
