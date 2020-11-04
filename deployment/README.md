# LibTorch Inference

A LibTorch inference implementation of yolov5. Both GPU and CPU are supported.

## Dependencies

- Ubuntu 18.04
- CUDA 10.2
- LibTorch 1.7.0+
- TorchVision 0.8.1+
- OpenCV 3.4+

## Usage

First, Setup the environment variables.

```bash
export TORCH_PATH=$(dirname $(python -c "import torch; print(torch.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TORCH_PATH/lib/
```

Then compile the source code.

```bash
mkdir build && cd build
cmake .. -DTorch_DIR=$TORCH_PATH/share/cmake/Torch
make
```

Now, you can infer your own images.

```bash
./yolo_inference [--input_source YOUR_IMAGE_SOURCE_PATH]
                 [--checkpoint ../../checkpoints/yolov5/yolov5s.torchscript.pt]
                 [--labelmap ../../checkpoints/yolov5/coco.names]
                 [--output_dir ../../data-bin/output]
                 [--gpu]  # GPU switch, Set False as default
```
