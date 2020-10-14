# 🔦 Yolov5 Runtime Stack

[![Stable](https://github.com/zhiqwang/yolov5-rt-stack/workflows/Stable/badge.svg)](https://github.com/zhiqwang/yolov5-rt-stack/actions?query=workflow%3AStable) [![Nightly](https://github.com/zhiqwang/yolov5-rt-stack/workflows/Nightly/badge.svg)](https://github.com/zhiqwang/yolov5-rt-stack/actions?query=workflow%3ANightly)

<p align="left"><a href=".github/bus_det.jpg"><img src=".github/bus_det.jpg" alt="YOLO inferencing" height="440"/></a></p>

## 🆕 What's New and Development Plans

- [x] Support exporting to `TorchScript` model. *Oct. 8, 2020.*
- [x] Support doing inference using `libtorch` cpp interface. *Oct. 10, 2020.*
- [ ] Support exporting to `onnx`, and doing inference using `onnxruntime`.
- [ ] Add more fetures ...

## 🛠 Usage

### PyTorch interface

The `detect.py` reads a directory and does inferencing with all contained images.

```bash
python -m detect [--model-cfg ./models/yolov5s.yaml]
                 [--model-checkpoint ./checkpoints/yolov5/yolov5s.pt]
                 [--coco-category-path ./libtorch_inference/weights/coco.names]
                 [--image-source YOUR_IMAGE_SOURCE_DIR]
                 [--output-dir ./data-bin/output]
                 [--img-size 416]
                 [--save-img]
```

### LibTorch interface

I have supplied a minimal [example](test/tracing/test_tracing.cpp) of getting `LibTorch` inferencing to work. You can check the [CI](.github/workflows/stable.yml) for more details.

- Setup the environment variables

```bash
export TORCH_PATH=$(dirname $(python -c "import torch; print(torch.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TORCH_PATH/lib/
```

- Navigate to the root directory of the examples

```bash
cd test/tracing
```

- Build examples

```bash
mkdir build && cd build
cmake .. -DTorch_DIR=$TORCH_PATH/share/cmake/Torch
make
```

## 🤗 Pretrained Checkpoints

Using following commands to update [ultralytics's](https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5s.pt) trained model weights.

```bash
python -m utils.updated_checkpoint [--checkpoint-path ./yolov5s.pt]
                                   [--cfg-path ./models/yolov5s.yaml]
                                   [--updated-checkpoint-path ./checkpoints/yolov5/yolov5s.pt]
```

Or you can download it from [here](https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.1.0/yolov5s.pt).

## 🎓 Acknowledgement

- <https://github.com/ultralytics/yolov5>
- <https://github.com/yasenh/libtorch-yolov5>
