# LibTorch Inference

![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white) ![LibTorch](https://img.shields.io/badge/LibTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white) ![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black) ![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white) ![macOS](https://img.shields.io/badge/macOS-000000?style=for-the-badge&logo=apple&logoColor=white)

The LibTorch inference for `yolort`, both GPU and CPU are supported.

## Dependencies

- LibTorch 1.8.0+ together with corresponding TorchVision 0.9.0+
- OpenCV
- CUDA 10.2+ \[Optional\]

*We didn't impose too strong restrictions on the version of CUDA.*

## Usage

1. First, Setup the LibTorch Environment variables.

   ```bash
   export TORCH_PATH=$(dirname $(python -c "import torch; print(torch.__file__)"))
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TORCH_PATH/lib/  # Optional
   ```

1. Don't forget to compile `LibTorchVision` using the following scripts.

   ```bash
   git clone https://github.com/pytorch/vision.git
   cd vision
   git checkout release/0.9  # Double check the version of TorchVision currently in use
   mkdir build && cd build
   # Add `-DWITH_CUDA=on` below if you're using GPU
   cmake .. -DTorch_DIR=$TORCH_PATH/share/cmake/Torch -DCMAKE_INSTALL_PREFIX=./install
   cmake --build .
   cmake --install .
   # Setup the LibTorchVision Environment variables
   export TORCHVISION_PATH=$PWD/install
   ```

1. Generate `TorchScript` model

   Unlike [ultralytics's](https://github.com/ultralytics/yolov5/blob/8ee9fd1/export.py) `torch.jit.trace` mechanism, We're using `torch.jit.script` to trace the YOLOv5 models which containing the whole pre-processing (especially with the [`letterbox`](https://github.com/ultralytics/yolov5/blob/8ee9fd1/utils/augmentations.py#L85-L115) ops) and post-processing (especially with the `nms` ops) procedures, as such you don't need to rewrite manually the C++ codes for pre-processing and post-processing.

   ```python
   from yolort.models import yolov5n

   model = yolov5n(pretrained=True)
   model.eval()

   traced_model = torch.jit.script(model)
   traced_model.save("yolov5n.torchscript.pt")
   ```

1. Then compile the source code.

   ```bash
   cd deployment/libtorch
   mkdir build && cd build
   cmake .. -DTorch_DIR=$TORCH_PATH/share/cmake/Torch -DTorchVision_DIR=$TORCHVISION_PATH/share/cmake/TorchVision
   make
   ```

1. Now, you can infer your own images.

   ```bash
   ./yolort_torch [--input_source ../../../test/assets/zidane.jpg]
                  [--checkpoint ../yolov5n.torchscript.pt]
                  [--labelmap ../../../notebooks/assets/coco.names]
                  [--gpu]  # GPU switch, which is optional, and set False as default
   ```
