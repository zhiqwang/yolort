# LibTorch Inference

The LibTorch inference of yolort. Both GPU and CPU are supported.

## Dependencies

- Ubuntu 18.04
- LibTorch 1.8.0 / 1.9.0
- TorchVision 0.9.0 / 0.10.0
- OpenCV 3.4+
- CUDA 10.2 [Optional]

## Usage

1. First, Setup the environment variables.

    ```bash
    export TORCH_PATH=$(dirname $(python -c "import torch; print(torch.__file__)"))
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TORCH_PATH/lib/
    ```

1. Don't forget to compile `TorchVision` using the following scripts.

    ```bash
    git clone https://github.com/pytorch/vision.git
    cd vision
    git checkout release/0.9  # replace to `nightly` branch instead if you are using the nightly version
    mkdir build && cd build
    cmake .. -DTorch_DIR=$TORCH_PATH/share/cmake/Torch  # Set `-DWITH_CUDA=ON` if you're using GPU
    make -j4
    sudo make install
    ```

1. Generate `TorchScript` model

    Unlike [ultralytics's](https://github.com/ultralytics/yolov5/blob/master/models/export.py) trace (`torch.jit.trace`) mechanism, We're using `torch.jit.script` to trace the YOLOv5 models which containing the whole pre-processing (especially using the `letterbox` ops) and post-processing (especially with the `nms` ops) procedures, so you don't need to rewrite manually the C++ codes of pre-processing and post-processing.

    ```bash
    git clone https://github.com/zhiqwang/yolov5-rt-stack.git
    cd yolov5-rt-stack
    python -m test.tracing.trace_model
    ```

1. Then compile the source code.

    ```bash
    cd deployment
    mkdir build && cd build
    cmake .. -DTorch_DIR=$TORCH_PATH/share/cmake/Torch
    make
    ```

1. Now, you can infer your own images.

    ```bash
    ./yolo_inference [--input_source YOUR_IMAGE_SOURCE_PATH]
                     [--checkpoint ../../checkpoints/yolov5/yolov5s.torchscript.pt]
                     [--labelmap ../../notebooks/assets/coco.names]
                     [--output_dir ../../data-bin/output]
                     [--gpu]  # GPU switch, Set False as default
    ```
