# 🔦 Yolov5 Runtime Stack

[![Stable](https://github.com/zhiqwang/yolov5-rt-stack/workflows/Stable/badge.svg)](https://github.com/zhiqwang/yolov5-rt-stack/actions?query=workflow%3AStable) [![Nightly](https://github.com/zhiqwang/yolov5-rt-stack/workflows/Nightly/badge.svg)](https://github.com/zhiqwang/yolov5-rt-stack/actions?query=workflow%3ANightly)

Yet another implementation of Ultralytics's [yolov5](https://github.com/ultralytics/yolov5), and with modules refactoring to make it available in the `libtorch`, `onnxruntime` and other backends. *Currently work in process, very pleasure for suggestion and cooperation.*

<a href=".github/zidane.jpg"><img src=".github/zidane.jpg" alt="YOLO inferencing" width="500"/></a>

## 🆕 What's New and Development Plans

- [x] Support exporting to `TorchScript` model. *Oct. 8, 2020*.
- [x] Support doing inference using `LibTorch` cpp interface. *Oct. 10, 2020*.
- [x] Add `TorchScript` cpp inference example, *Nov. 4, 2020*.
- [x] Refactor YOLO modules, *Nov. 16, 2020*.
- [x] Support exporting to `onnx`, and doing inference using `onnxruntime`. *Nov. 17, 2020*.
- [ ] Add more fetures ...

## 🛠 Usage

There are something different comparing to [ultralytics's](https://github.com/ultralytics/yolov5/blob/master/models/yolo.py) implementation. This repo can load ultralytics's trained model checkpoint with minor modifications, I have converted ultralytics's lastest released [v3.1](https://github.com/ultralytics/yolov5/releases/download/v3.1/yolov5s.pt) checkpoint [here](https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.2.1/yolov5s.pt).

You can also convert ultralytics's trained (or your own) model checkpoint with the following command:

```bash
python -m utils.updated_checkpoint [--checkpoint_path ./yolov5s.pt]
                                   [--cfg_path ./models/yolov5s.yaml]
                                   [--updated_checkpoint_path ./checkpoints/yolov5/yolov5s.pt]
```

### ✨ `PyTorch` Interface

The `detect.py` reads a image source and does inference, you can also check for the more details in [inference-pytorch-export-libtorch](notebooks/inference-pytorch-export-libtorch.ipynb) notebook.

```bash
python -m detect [--model_cfg ./models/yolov5s.yaml]
                 [--input_source YOUR_IMAGE_SOURCE_DIR]
                 [--checkpoint ./checkpoints/yolov5/yolov5s.pt]
                 [--labelmap ./checkpoints/yolov5/coco.names]
                 [--output_dir ./data-bin/output]
                 [--min_size 640]
                 [--max_size 640]
                 [--save_img]
                 [--gpu]  # GPU switch, Set False as default
```

### 🚀 `LibTorch` Interface

Here providing an [example](./deployment) of getting `LibTorch` inferencing to work. Also you can check the [CI](.github/workflows/stable.yml) for more details.

### ✏️ Model Visualization

Now, `yolov5-rt-stack` can draw the model graph directly, check for more details in [visualize-jit-models](notebooks/visualize-jit-models.ipynb) notebook.

<a href="notebooks/assets/yolov5.detail.svg"><img src="notebooks/assets/yolov5.detail.svg" alt="YOLO model visualize" width="500"/></a>

## 🎓 Acknowledgement

- The implementation of `yolov5` borrow the code from [ultralytics](https://github.com/ultralytics/yolov5).
- This repo borrows the architecture design and part of the code from [torchvision](https://github.com/pytorch/vision).
