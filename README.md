# 🔦 Yolov5 Runtime Stack

[![Stable](https://github.com/zhiqwang/yolov5-rt-stack/workflows/Stable/badge.svg)](https://github.com/zhiqwang/yolov5-rt-stack/actions?query=workflow%3AStable) [![Nightly](https://github.com/zhiqwang/yolov5-rt-stack/workflows/Nightly/badge.svg)](https://github.com/zhiqwang/yolov5-rt-stack/actions?query=workflow%3ANightly)

Yet another implementation of Ultralytics's [yolov5](https://github.com/ultralytics/yolov5), and with modules refactoring to make it available in deployment backends such as `libtorch`, `onnxruntime` and so on.

<a href=".github/zidane.jpg"><img src=".github/zidane.jpg" alt="YOLO inference demo" width="500"/></a>

## 🆕 What's New

- Support exporting to `TorchScript` model. *Oct. 8, 2020*.
- Support inferring with `LibTorch` cpp interface. *Oct. 10, 2020*.
- Add `TorchScript` cpp inference example, *Nov. 4, 2020*.
- Refactor YOLO modules and support *dynmaic batching* inference, *Nov. 16, 2020*.
- Support exporting to `onnx`, and inferring with `onnxruntime` interface. *Nov. 17, 2020*.
- Add graph visualization tools. *Nov. 21, 2020*.

## 🛠️ Usage

There are something different comparing to [ultralytics's](https://github.com/ultralytics/yolov5/blob/master/models/yolo.py) implementation. This repo can load ultralytics's trained model checkpoint with minor modifications, I have converted ultralytics's lastest released [v3.1](https://github.com/ultralytics/yolov5/releases/download/v3.1/yolov5s.pt) checkpoint [here](https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.2.1/yolov5s.pt).

You can also convert ultralytics's trained (or your own) model checkpoint with the following command:

```bash
python -m utils.updated_checkpoint [--checkpoint_path_ultralytics ./checkpoint/yolov5s_ultralytics.pt]
                                   [--checkpoint_path_rt_stack ./checkpoints/yolov5s_rt_stack.pt]
```

### 🔥 Loading via `torch.hub`

The models are also available via torch hub, to load `yolov5s` with pretrained weights simply do:

```python
model = torch.hub.load('zhiqwang/yolov5-rt-stack', 'yolov5s', pretrained=True)
```

### ✨ Inference on `PyTorch` backend

There are no extra compiled components in `yolov5-rt-stack` and package dependencies are minimal, so the code is very simple to use. We provide instructions how to install dependencies via conda. First, clone the repository locally:

```bash
git clone https://github.com/zhiqwang/yolov5-rt-stack.git
```

Then, install PyTorch 1.7.0+ and torchvision 0.8.1+:

```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

To read a source image and detect its objects run:

```bash
python -m detect [--input_source YOUR_IMAGE_SOURCE_DIR]
                 [--labelmap ./notebooks/assets/coco.names]
                 [--output_dir ./data-bin/output]
                 [--min_size 640]
                 [--max_size 640]
                 [--save_img]
                 [--gpu]  # GPU switch, Set False as default
```

You can also check the [inference-pytorch-export-libtorch](notebooks/inference-pytorch-export-libtorch.ipynb) notebook for more details.

### 🚀 Inference on `LibTorch` backend

Here provide an [example](./deployment) of getting `LibTorch` inference to work. Also you can check the [CI](.github/workflows/stable.yml) for more details.

## 🎨 Model Graph Visualization

Now, `yolov5-rt-stack` can draw the model graph directly, check for more details in [visualize-jit-models](notebooks/visualize-jit-models.ipynb) notebook.

<a href="notebooks/assets/yolov5.detail.svg"><img src="notebooks/assets/yolov5.detail.svg" alt="YOLO model visualize" width="500"/></a>

## 🌟 Contributing to `yolov5-rt-stack`

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us. BTW, leave a star if you liked it, this means a lot to me :)

## 🎓 Acknowledgement

- The implementation of `yolov5` borrow the code from [ultralytics](https://github.com/ultralytics/yolov5).
- This repo borrows the architecture design and part of the code from [torchvision](https://github.com/pytorch/vision).
