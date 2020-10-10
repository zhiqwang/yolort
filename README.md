# 🔦 Yolov5 Runtime Stack

[![Stable](https://github.com/zhiqwang/yolov5-rt-stack/workflows/Stable/badge.svg)](https://github.com/zhiqwang/yolov5-rt-stack/actions?query=workflow%3AStable) [![Nightly](https://github.com/zhiqwang/yolov5-rt-stack/workflows/Nightly/badge.svg)](https://github.com/zhiqwang/yolov5-rt-stack/actions?query=workflow%3ANightly)

## 🆕 What's New and Development Plans

- [x] Support exporting to `TorchScript` model. *Oct. 8, 2020.*
- [x] Support doing inference using `libtorch` cpp interface. *Oct. 10, 2020.*
- [ ] Support exporting to `onnx`, and doing inference using `onnxruntime`.
- [ ] Add more fetures ...

## 🛠 Usage

## 🤗 Pretrained Models

Using below command to update [ultralytics's](https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5s.pt) trained model weights.

```bash
python -m utils.updated_checkpoint [--checkpoint-path ./yolov5s.pt] \
    [--cfg-path ./models/yolov5s.yaml] \
    [--updated-checkpoint-path ./checkpoints/yolov5/yolov5s.pt]
```

Or you can download it from [here](https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.1.0/yolov5s.pt).

## 🎓 Acknowledgement

- <https://github.com/ultralytics/yolov5>
- <https://github.com/yasenh/libtorch-yolov5>
