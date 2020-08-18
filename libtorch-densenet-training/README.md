# A Demo of Classification Training Task in LibTorch

A LibTorch inference implementation of the classification object detection algorithm. Both GPU and CPU are supported. Obviously the libtorch is not suitable for training a deep learning model, this repo is just for presenting the utility of libtorch.

## Dependencies

- Ubuntu 18.04
- CUDA 10.2
- LibTorch 1.6.0 <https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.6.0.zip>
- TorchVisoin 0.7.0

## Usage

```bash
$ mkdir build $$ cd build
$ cmake -D CMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
$ cmake --build .
[ 25%] Building CXX object CMakeFiles/libtorch-classifier.dir/src/cifar10.cpp.o
[ 50%] Building CXX object CMakeFiles/libtorch-classifier.dir/src/main.cpp.o
[ 75%] Building CXX object CMakeFiles/libtorch-classifier.dir/src/transform.cpp.o
[100%] Linking CXX executable libtorch-classifier
[100%] Built target libtorch-classifier
$ ./libtorch-classifier
```

## Acknowledgement

- This repo borrows the code from <https://github.com/prabhuomkar/pytorch-cpp>.
