# ONNXRuntime Inference

The ONNXRuntime inference for `yolort`, both GPU and CPU are supported.

## Dependencies

- Ubuntu 20.04 / Windows 10 / macOS
- ONNXRuntime 1.7 +
- OpenCV 4.5 +
- CUDA 11 \[Optional\]

*We didn't impose too strong restrictions on the versions of dependencies.*

## Features

The `ONNX` model exported with `yolort` differs from the official one in the following three ways.

- The exported `ONNX` graph now supports dynamic shapes, and we use `(3, H, W)` as the input shape (for example `(3, 640, 640)`).
- We embed the pre-processing ([`letterbox`](https://github.com/ultralytics/yolov5/blob/9ef94940aa5e9618e7e804f0758f9a6cebfc63a9/utils/augmentations.py#L88-L118)) into the graph as well. We only require the input image to be in the `RGB` channel, and to be rescaled to `float32 [0-1]` from general `uint [0-255]`. The main logic we use to implement this mechanism is below. (And [this](https://github.com/zhiqwang/yolov5-rt-stack/blob/b9c67205a61fa0e9d7e6696372c133ea0d36d9db/yolort/models/transform.py#L210-L234) plays the same role of the official `letterbox`, but there will be a little difference in accuracy now.)
- We embed the post-processing (`nms`) into the model graph, which performs the same task as [`non_max_suppression`](https://github.com/ultralytics/yolov5/blob/fad57c29cd27c0fcbc0038b7b7312b9b6ef922a8/utils/general.py#L532-L623) except for the format of the inputs. (And here the `ONNX` graph is required to be dynamic.)

## Usage

1. First, Setup the environment variable.

   ```bash
   export ORT_DIR=YOUR_ONNXRUNTIME_DIR
   ```

1. Compile the source code.

   ```bash
   cd deployment/onnxruntime
   mkdir build && cd build
   cmake .. -DONNXRUNTIME_DIR=$ORT_DIR
   cmake --build .
   ```

1. Export your custom model to ONNX.

   ```bash
   python tools/export_model.py [--checkpoint_path path/to/custom/best.pt]
   ```

   And then, you can find that a new pair of ONNX models ("best.onnx" and "best.sim.onnx") has been generated in the directory of "best.pt".

1. \[Optional\] Quick test with the ONNXRuntime Python interface.

   ```python
   from yolort.runtime import PredictorORT

   detector = PredictorORT("best.onnx")
   img_path = "bus.jpg"
   scores, class_ids, boxes = detector.run_on_image(img_path)
   ```

1. Now, you can infer your own images.

   ```bash
   ./yolort_onnx [--image ../../../test/assets/zidane.jpg]
                 [--model_path ../../../notebooks/yolov5s.onnx]
                 [--class_names ../../../notebooks/assets/coco.names]
                 [--gpu]  # GPU switch, which is optional, and set False as default
   ```
