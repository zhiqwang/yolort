# TensorRT Inference

The TensorRT inference for `yolort`, support GPU only.

## Dependencies

- TensorRT 8.x

## Usage

1. Create build director and cmake config.

   ```bash
   mkdir -p build/ && cd build/
   cmake .. -DTENSORRT_DIR=${your_trt_install_director}
   ```

1. Build project

   ```bash
   make
   ```

1. Export your custom model to ONNX(see [onnx-graphsurgeon-inference-tensorrt](https://github.com/zhiqwang/yolov5-rt-stack/blob/main/notebooks/onnx-graphsurgeon-inference-tensorrt.ipynb)).

1. Now, you can infer your own images.

   ```bash
   ./yolort_trt [--image ../../../test/assets/zidane.jpg]
                 [--model_path ../../../notebooks/yolov5s.onnx]
                 [--class_names ../../../notebooks/assets/coco.names]
                 [--fp16]  # Enable it if your GPU support fp16 inference
   ```
