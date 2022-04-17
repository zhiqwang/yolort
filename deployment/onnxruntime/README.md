# ONNX Runtime Inference

![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white) ![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black) ![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white) ![macOS](https://img.shields.io/badge/macOS-000000?style=for-the-badge&logo=apple&logoColor=white)

The ONNX Runtime inference for `yolort`, both CPU and GPU are supported.

## Dependencies

- ONNX Runtime 1.7+
- OpenCV
- CUDA \[Optional\]

*We didn't impose too strong restrictions on the versions of dependencies.*

## Features

The ONNX model exported by yolort differs from other pipeline in the following three ways.

- We embed the pre-processing into the graph (mainly composed of `letterbox`). and the exported model expects a `Tensor[C, H, W]`, which is in `RGB` channel and is rescaled to range `float32 [0-1]`.
- We embed the post-processing into the model graph with `torchvision.ops.batched_nms`. So the outputs of the exported model are straightforward `boxes`, `labels` and `scores` fields of this image.
- We adopt the dynamic shape mechanism to export the ONNX models.

## Usage

1. Export your custom model to ONNX.

   ```bash
   python tools/export_model.py --checkpoint_path {path/to/your/best.pt} --size_divisible 32/64
   ```

   And then, you can find that a ONNX model ("best.onnx") have been generated in the directory of "best.pt". Set the `size_divisible` here according to your model, 32 for P5 ("yolov5s.pt" for instance) and 64 for P6 ("yolov5s6.pt" for instance).

1. \[Optional\] Quick test with the ONNX Runtime Python interface.

   ```python
   from yolort.runtime import PredictorORT

   # Load the serialized ONNX model
   engine_path = "yolov5n6.onnx"
   device = "cpu"
   y_runtime = PredictorORT(engine_path, device=device)

   # Perform inference on an image file
   predictions = y_runtime.predict("bus.jpg")
   ```

1. Compile the source code.

   ```bash
   cd deployment/onnxruntime
   mkdir build && cd build
   cmake .. -DONNXRUNTIME_DIR={path/to/your/ONNXRUNTIME/install/director}
   cmake --build .
   ```

1. Now, you can infer your own images.

   ```bash
   ./yolort_onnx --image ../../../test/assets/zidane.jpg
                 --model_path ../../../notebooks/best.onnx
                 --class_names ../../../notebooks/assets/coco.names
                 [--gpu]  # GPU switch, which is optional, and set False as default
   ```
