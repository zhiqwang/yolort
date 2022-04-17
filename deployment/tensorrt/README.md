# TensorRT Inference Example

![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white) ![Nvidia](https://img.shields.io/badge/NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=white) ![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black) ![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)

The TensorRT inference example of `yolort`.

## Dependencies

- TensorRT 8.2+
- OpenCV

## Usage

Here we will mainly discuss how to use the C++ interface, we recommend that you check out our [tutorial](https://zhiqwang.com/yolov5-rt-stack/notebooks/onnx-graphsurgeon-inference-tensorrt.html) first.

1. Export your custom model to TensorRT format

   We provide a CLI tool to export the custom model checkpoint trained from yolov5 to TensorRT serialized engine.

   ```bash
   python tools/export_model.py --checkpoint_path {path/to/your/best.pt} --include engine
   ```

   Note: This CLI will output a pair of ONNX model and TensorRT serialized engine if you have the full TensorRT's Python environment, otherwise it will only output an ONNX models with suffixes ".trt.onnx". And then you can also use the [`trtexct`](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec) provided by TensorRT to export the serialized engine as below:

   ```bash
   trtexec --onnx=best.trt.onnx --saveEngine=best.engine --workspace=8192
   ```

1. \[Optional\] Quick test with the TensorRT Python interface.

   ```python
   import torch
   from yolort.runtime import PredictorTRT

   # Load the serialized TensorRT engine
   engine_path = "best.engine"
   device = torch.device("cuda")
   y_runtime = PredictorTRT(engine_path, device=device)

   # Perform inference on an image file
   predictions = y_runtime.predict("bus.jpg")
   ```

1. Prepare the environment for OpenCV and TensorRT

   - Build OpenCV libraries
   - Download CUDA, cudnn and TensorRT

1. Create build directory and build `yolort_trt` project

   - Build yolort TensorRT executable files

     ```bash
     mkdir -p build && cd build
     # Add `-G "Visual Studio 16 2019"` below to specify the compile version of VS on Windows System
     cmake -DTENSORRT_DIR={path/to/your/TensorRT/install/directory} -DOpenCV_DIR={path/to/your/OpenCV_BUILD_DIR} ..
     cmake --build .  # Can also use the yolort_trt.sln to build on Windows System
     ```

   - \[Windows System Only\] Copy following dependent dynamic link libraries (xxx.dll) to Release/Debug directory

     - cudnn_cnn_infer64_8.dll, cudnn_ops_infer64_8.dll, cudnn64_8.dll, nvinfer.dll, nvinfer_plugin.dll, nvonnxparser.dll, zlibwapi.dll (On which CUDA and cudnn depend)
     - opencv_corexxx.dll opencv_imgcodecsxxx.dll opencv_imgprocxxx.dll (Subsequent dependencies by OpenCV or you can also use Static OpenCV Library)

1. Now, you can infer your own images.

   ```bash
   ./yolort_trt --image {path/to/your/image}
                --model_path {path/to/your/serialized/tensorrt/engine}
                --class_names {path/to/your/class/names}
   ```

   The above `yolort_trt` will determine if it needs to build the serialized engine file from ONNX based on the file suffix, and only do serialization when the argument `--model_path` given are with `.onnx` suffixes, all other suffixes are treated as the TensorRT serialized engine.
