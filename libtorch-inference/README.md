# LibTorch Inference

A LibTorch inference implementation of yolov5. Both GPU and CPU are supported.

## Dependencies

- Ubuntu 18.04
- CUDA 10.2
- LibTorch 1.6.0 <https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.6.0.zip>

## Usage

```bash
mkdir build && cd build
cmake -D CMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
make
# CPU
./libtorch-inference --source [demo.jpg] --weights [yolov5s.torchscript.pt] --view-img
```

## Acknowledgement

- This repo borrows the code from <https://github.com/yasenh/libtorch-yolov5>.
