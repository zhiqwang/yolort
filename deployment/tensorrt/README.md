# TensorRT Inference

The TensorRT inference for `yolort`, support CUDA only.

## Dependencies

- TensorRT 8.2 +

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

   # Load the exported TensorRT engine
   engine_path = "best.engine"
   device = torch.device("cuda")
   y_runtime = PredictorTRT(engine_path, device=device)

   # Perform inference on an image file
   predictions = y_runtime.predict("bus.jpg")
   ```

1. Create build directory and build project.

   ```bash
   mkdir -p build && cd build
   cmake .. -DTENSORRT_DIR={path/to/your/TensorRT/install/director}
   cmake --build .
   ```

1. Now, you can infer your own images.

   ```bash
   ./yolort_trt --image ../../../test/assets/zidane.jpg
                --model_path ../../../notebooks/best.engine
                --class_names ../../../notebooks/assets/coco.names
   ```

   The above `yolort_trt` will determine if it needs to build the serialized engine file from ONNX based on the file suffix, and only do serialization when the argument `--model_path` given are with `.onnx` suffixes, all other suffixes are treated as the TensorRT serialized engine.
