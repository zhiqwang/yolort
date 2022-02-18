# TensorRT Inference

The TensorRT inference for `yolort`, support CUDA only.

## Dependencies

- TensorRT 8.2 +

## Usage

1. Create build directory and cmake config.

   ```bash
   mkdir -p build/ && cd build/
   cmake .. -DTENSORRT_DIR={path/to/your/trt/install/director}
   ```

1. Build project

   ```bash
   cmake --build .
   ```

1. Export your custom model to TensorRT format

   Here is a small demo to surgeon the YOLOv5 ONNX model and then export to TensorRT engine. For details see out our [tutorial for deploying yolort on TensorRT](https://zhiqwang.com/yolov5-rt-stack/notebooks/onnx-graphsurgeon-inference-tensorrt.html).

   We provide a CLI tool to export the custom model checkpoint trained from yolov5 to TensorRT serialized engine.

   ```bash
   python tools/export_model.py --checkpoint_path [path/to/your/best.pt] --include engine
   ```

   Note: This CLI will output a pair of ONNX model and TensorRT serialized engine if you have the full TensorRT's Python environment, otherwise it will only output an ONNX models. And you can also use the [`trtexct`](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec) provided by TensorRT to export the serialized engine as below:

   ```bash
   trtexec --onnx=yolov5n6.trt.onnx --saveEngine=yolov5n6-trtexec.engine --workspace=8192
   ```

1. Now, you can infer your own images.

   ```bash
   ./yolort_trt [--image ../../../test/assets/zidane.jpg]
                [--model_path ../../../notebooks/yolov5s.trt.onnx]
                [--class_names ../../../notebooks/assets/coco.names]
   ```

   The above `yolort_trt` will determine if it needs to build the serialized engine file from ONNX based on the file suffix, and only do serialization when the argument `--model_path` given are with `.onnx` suffixes, all other suffixes are treated as the TensorRT serialized engine.
