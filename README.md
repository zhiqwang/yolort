# 🔦 yolort - YOLOv5 Runtime Stack

[![CI testing](https://github.com/zhiqwang/yolov5-rt-stack/workflows/CI%20testing/badge.svg)](https://github.com/zhiqwang/yolov5-rt-stack/actions?query=workflow%3A%22CI+testing%22)
[![PyPI version](https://badge.fury.io/py/yolort.svg)](https://badge.fury.io/py/yolort)
[![codecov](https://codecov.io/gh/zhiqwang/yolov5-rt-stack/branch/master/graph/badge.svg?token=1GX96EA72Y)](https://codecov.io/gh/zhiqwang/yolov5-rt-stack)
[![Github Downloads](https://img.shields.io/github/downloads/zhiqwang/yolov5-rt-stack/total?color=blue&label=Downloads&logo=github&logoColor=lightgrey)](https://img.shields.io/github/downloads/zhiqwang/yolov5-rt-stack/total?color=blue&label=Downloads&logo=github&logoColor=lightgrey)
[![Downloads PyPI](https://static.pepy.tech/personalized-badge/yolort?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads%20PyPI)](https://pepy.tech/project/yolort)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/yolort/shared_invite/zt-mqwc7235-940aAh8IaKYeWclrJx10SA)

**What it is.** Yet another implementation of Ultralytics's [yolov5](https://github.com/ultralytics/yolov5), and with modules refactoring to make it available in deployment backends such as `libtorch`, `onnxruntime`, `tvm` and so on.

**About the code.** Follow the design principle of [detr](https://github.com/facebookresearch/detr):

> object detection should not be more difficult than classification, and should not require complex libraries for training and inference.

`yolort` is very simple to implement and experiment with. You like the implementation of torchvision's faster-rcnn, retinanet or detr? You like yolov5? You love `yolort`!

<a href=".github/zidane.jpg"><img src=".github/zidane.jpg" alt="YOLO inference demo" width="500"/></a>

## 🆕 What's New

- Support exporting to `TorchScript` model. *Oct. 8, 2020*.
- Support inferring with `LibTorch` cpp interface. *Oct. 10, 2020*.
- Add `TorchScript` cpp inference example. *Nov. 4, 2020*.
- Refactor YOLO modules and support *dynmaic batching* inference. *Nov. 16, 2020*.
- Support exporting to `ONNX`, and inferring with `ONNXRuntime` interface. *Nov. 17, 2020*.
- Add graph visualization tools. *Nov. 21, 2020*.
- Add `TVM` compile and inference notebooks. *Feb. 5, 2021*.

## 🛠️ Usage

There are no extra compiled components in `yolort` and package dependencies are minimal, so the code is very simple to use.

### Installation and Inference Examples

- Above all, follow the [official instructions](https://pytorch.org/get-started/locally/) to install PyTorch 1.7.0+ and torchvision 0.8.1+

- Installation via Pip

  Simple installation from [PyPI](https://pypi.org/project/yolort/)

  ```shell
  pip install -U yolort
  ```

  Or from Source

  ```shell
  # clone yolort repository locally
  git clone https://github.com/zhiqwang/yolov5-rt-stack.git
  cd yolov5-rt-stack
  # install in editable mode
  pip install -e .
  ```

- Install pycocotools (for evaluation on COCO):

  ```shell
  pip install -U 'git+https://github.com/ppwwyyxx/cocoapi.git#subdirectory=PythonAPI'
  ```

- To read a source of image(s) and detect its objects 🔥

  ```python
  from yolort.models import yolov5s

  # Load model
  model = yolov5s(pretrained=True, score_thresh=0.45)
  model.eval()

  # Perform inference on an image file
  predictions = model.predict('bus.jpg')
  # Perform inference on a list of image files
  predictions = model.predict(['bus.jpg', 'zidane.jpg'])
  ```

### Loading via `torch.hub`

The models are also available via torch hub, to load `yolov5s` with pretrained weights simply do:

```python
model = torch.hub.load('zhiqwang/yolov5-rt-stack', 'yolov5s', pretrained=True)
```

### Updating checkpoint from ultralytics/yolov5

The module state of `yolort` has some differences comparing to `ultralytics/yolov5`. We can load ultralytics's trained model checkpoint with minor changes, and we have converted ultralytics's release [v3.1](https://github.com/ultralytics/yolov5/releases/tag/v3.1) and [v4.0](https://github.com/ultralytics/yolov5/releases/tag/v4.0). For example, if you want to convert a `yolov5s` (release 4.0) model, you can just run the following script:

```python
from yolort.utils import update_module_state_from_ultralytics

# Update module state from ultralytics
model = update_module_state_from_ultralytics(arch='yolov5s', version='v4.0')
# Save updated module
torch.save(model.state_dict(), 'yolov5s_updated.pt')
```

### Inference on `LibTorch` backend 🚀

We provide a [notebook](notebooks/inference-pytorch-export-libtorch.ipynb) to demonstrate how the model is transformed into `torchscript`. And we provide an [C++ example](./deployment) of how to infer with the transformed `torchscript` model. For details see the [GitHub Actions](.github/workflows/ci_test.yml).

## 🎨 Model Graph Visualization

Now, `yolort` can draw the model graph directly, checkout our [visualize-jit-models](notebooks/visualize-jit-models.ipynb) notebook to see how to use and visualize the model graph.

<a href="notebooks/assets/yolov5.detail.svg"><img src="notebooks/assets/yolov5.detail.svg" alt="YOLO model visualize" width="500"/></a>

## 🎓 Acknowledgement

- The implementation of `yolov5` borrow the code from [ultralytics](https://github.com/ultralytics/yolov5).
- This repo borrows the architecture design and part of the code from [torchvision](https://github.com/pytorch/vision).

## 🤗 Contributing

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us. *BTW, leave a 🌟 if you liked it, this means a lot to us* :)
