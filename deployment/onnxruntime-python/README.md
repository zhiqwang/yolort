# ONNXRuntime Inference

The ONNXRuntime inference for `yolort`, both GPU and CPU are supported.

## Dependencies

- Ubuntu 20.04 / Windows 10
- ONNXRuntime 1.7 +
- OpenCV 4.5 +
- CUDA 11 [Optional]

*We didn't impose too strong restrictions on the versions of dependencies.*

## Usage

1. Update your PyTorch model weights from ultralytics to yolort and export to ONNX following the [notebooks with tutorials](https://github.com/zhiqwang/yolov5-rt-stack/blob/master/notebooks/).

2. To infer your own images run the script as following.

    ```bash
    python example.py [--image ../../../test/assets/zidane.jpg]
                      [--model_path ../../../notebooks/yolov5s.onnx]
                      [--class_names ../../../notebooks/assets/coco.names]
                      [--gpu]  # GPU switch, which is optional, and set False as default
    ```