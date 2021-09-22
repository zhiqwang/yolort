# ONNXRuntime Inference

The ONNXRuntime inference for `yolort`, both GPU and CPU are supported.

## Dependencies

- Ubuntu 20.04 / Windows 10
- ONNXRuntime 1.7 +
- OpenCV 4.5 +
- CUDA 11 [Optional]

*We didn't impose too strong restrictions on the versions of dependecies.*

## Usage

1. First, Setup the environment variable.

    ```bash
    export ORT_DIR=YOUR_ONNXRUNTIME_DIR
    ```

2. Compile the source code.

    ```bash
    cd deployment/onnxruntime
    mkdir build && cd build
    cmake .. -DONNXRUNTIME_DIR=$ORT_DIR
    cmake --build .
    ```

3. Update your PyTorch model weights from ultralytics to yolort and export to ONNX following the [notebooks with tutorials](https://github.com/zhiqwang/yolov5-rt-stack/blob/master/notebooks/).

4. Now, you can infer your own images.

    ```bash
    ./yolort_onnx [--image ../../../test/assets/zidane.jpg]
                    [--model_path ../../../test/tracing/yolov5s.torchscript.pt]
                    [--class_names ../../../notebooks/assets/coco.names]
                    [--gpu]  # GPU switch, which is optional, and set False as default
    ```