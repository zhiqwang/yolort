# LibTorch Inference

A LibTorch inference implementation of yolov5. Both GPU and CPU are supported.

## Dependencies

- Ubuntu 18.04
- CUDA 10.2
- LibTorch 1.7.0+
- TorchVision 0.8.1+
- OpenCV 3.4+

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
    git checkout release/0.8.0 # replace to `nightly` branch instead if you are using the nightly version
    mkdir build && cd build
    cmake .. -DTorch_DIR=$TORCH_PATH/share/cmake/Torch
    make -j4
    sudo make install
    ```

1. Generate `TorchScript` model

    Unlike [ultralytics's](https://github.com/ultralytics/yolov5/blob/master/models/export.py) trace (`torch.jit.trace`) mechanism, I'm using `torch.jit.script` to jit trace the YOLO models which containing the whole pre-processing (especially using the `GeneralizedRCNNTransform` ops) and post-processing (especially with the `nms` ops) procedures, so you don't need to rewrite manually the cpp codes of pre-processing and post-processing.

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
                     [--labelmap ../../checkpoints/yolov5/coco.names]
                     [--output_dir ../../data-bin/output]
                     [--gpu]  # GPU switch, Set False as default
    ```
