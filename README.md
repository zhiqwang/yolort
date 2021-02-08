# 🔦 yolov5rt - YOLOv5 Runtime Stack

[![CI testing](https://github.com/zhiqwang/yolov5-rt-stack/workflows/CI%20testing/badge.svg)](https://github.com/zhiqwang/yolov5-rt-stack/actions?query=workflow%3A%22CI+testing%22) [![codecov](https://codecov.io/gh/zhiqwang/yolov5-rt-stack/branch/master/graph/badge.svg?token=1GX96EA72Y)](https://codecov.io/gh/zhiqwang/yolov5-rt-stack)

**Note:** The master branch is not stable, please use [release/0.2.0](https://github.com/zhiqwang/yolov5-rt-stack/tree/release/v0.2.0) now.

**What it is.** Yet another implementation of Ultralytics's [yolov5](https://github.com/ultralytics/yolov5), and with modules refactoring to make it available in deployment backends such as `libtorch`, `onnxruntime` and so on.

**About the code.** Follow the design principle of [detr](https://github.com/facebookresearch/detr):

> object detection should not be more difficult than classification, and should not require complex libraries for training and inference.

`yolov5rt` is very simple to implement and experiment with. You like the implementation of torchvision's faster-rcnn, retinanet or detr? You like yolov5? You love `yolov5rt`!

<a href=".github/zidane.jpg"><img src=".github/zidane.jpg" alt="YOLO inference demo" width="500"/></a>

## 🆕 What's New

- Support exporting to `TorchScript` model. *Oct. 8, 2020*.
- Support inferring with `LibTorch` cpp interface. *Oct. 10, 2020*.
- Add `TorchScript` cpp inference example, *Nov. 4, 2020*.
- Refactor YOLO modules and support *dynmaic batching* inference, *Nov. 16, 2020*.
- Support exporting to `onnx`, and inferring with `onnxruntime` interface. *Nov. 17, 2020*.
- Add graph visualization tools. *Nov. 21, 2020*.
- Add `TVM` compile and inference notebooks. *Feb 5, 2021*.

## 🛠️ Usage

There are no extra compiled components in `yolov5rt` and package dependencies are minimal, so the code is very simple to use.

<details><summary>Expand to see the instructions of how to install dependencies via conda.</summary><br/>

- First, clone the repository locally:

  ```bash
  git clone https://github.com/zhiqwang/yolov5-rt-stack.git
  ```

- Then, install PyTorch 1.7.0+ and torchvision 0.8.1+:

  ```bash
  conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
  ```

- Install pycocotools (for evaluation on COCO) and scipy (for training):

  ```bash
  conda install cython scipy
  pip install -U pycocotools>=2.0.2  # corresponds to https://github.com/ppwwyyxx/cocoapi
  ```

- That's it, should be good to train and evaluate detection models.

</details>

### Loading via `torch.hub`

The models are also available via torch hub, to load `yolov5s` with pretrained weights simply do:

```python
model = torch.hub.load('zhiqwang/yolov5-rt-stack', 'yolov5s', pretrained=True)
```

### Updating checkpoint from ultralytics/yolov5

The module state of `yolov5rt` has some differences comparing to `ultralytics/yolov5`. We can load ultralytics's trained model checkpoint with minor changes, and we have converted ultralytics's lastest release [v3.1](https://github.com/ultralytics/yolov5/releases/download/v3.1/yolov5s.pt) checkpoint [here](https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.2.1/yolov5s.pt).

<details><summary>Expand to see more information of how to update ultralytics's trained (or your own) model checkpoint.</summary><br/>

- If you train your model using ultralytics's repo, you should update the model checkpoint first. ultralytics's trained model has a limitation that their model must load in the root path of ultralytics, so a important thing is to desensitize the path dependence as follows:

  ```python
  # Noted that current path is the root of ultralytics/yolov5, and the checkpoint is
  # downloaded from <https://github.com/ultralytics/yolov5/releases/download/v3.1/yolov5s.pt>
  ultralytics_weights = 'https://github.com/ultralytics/yolov5/releases/download/v3.1/yolov5s.pt'
  checkpoints_ = torch.load(ultralytics_weights, map_location='cpu')['model']
  torch.save(checkpoints_.state_dict(), desensitize_ultralytics_weights)
  ```

- Load `yolov5rt` model as follows:

  ```python
  from hubconf import yolov5s

  model = yolov5s()
  model.eval()
  ```

- Now let's update ultralytics/yolov5 trained checkpoint, see the [conversion script](utils/updated_checkpoint.py) for more information:

  ```python
  from utils.updated_checkpoint import update_ultralytics_checkpoints

  model = update_ultralytics_checkpoints(model, desensitize_ultralytics_weights)
  # updated checkpint is saved to checkpoint_path_rt_stack
  torch.save(model.state_dict(), checkpoint_path_rt_stack)
  ```

</details>

### Inference on `PyTorch` backend 🔥

To read a source image and detect its objects run:

```bash
python -m detect [--input_source ./test/assets/zidane.jpg]
                 [--labelmap ./notebooks/assets/coco.names]
                 [--output_dir ./data-bin/output]
                 [--min_size 640]
                 [--max_size 640]
                 [--save_img]
                 [--gpu]  # GPU switch, Set False as default
```

You can also see the [inference-pytorch-export-libtorch](notebooks/inference-pytorch-export-libtorch.ipynb) notebook for more information.

### Inference on `LibTorch` backend 🚀

We provide an [example](./deployment) of getting `LibTorch` inference to work. For details see the [GitHub actions](.github/workflows/nightly.yml).

## 🎨 Model Graph Visualization

Now, `yolov5rt` can draw the model graph directly, checkout our [visualize-jit-models](notebooks/visualize-jit-models.ipynb) notebook to see how to use and visualize the model graph.

<a href="notebooks/assets/yolov5.detail.svg"><img src="notebooks/assets/yolov5.detail.svg" alt="YOLO model visualize" width="500"/></a>

## 🎓 Acknowledgement

- The implementation of `yolov5` borrow the code from [ultralytics](https://github.com/ultralytics/yolov5).
- This repo borrows the architecture design and part of the code from [torchvision](https://github.com/pytorch/vision).

## 🤗 Contributing

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us. *BTW, leave a 🌟 if you liked it, this means a lot to us* :)
