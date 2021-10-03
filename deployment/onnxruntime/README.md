# ONNXRuntime Inference

The ONNXRuntime inference for `yolort`, both GPU and CPU are supported.

## Dependencies

- Ubuntu 20.04 / Windows 10
- ONNXRuntime 1.7 +
- OpenCV 4.5 +
- CUDA 11 \[Optional\]

*We didn't impose too strong restrictions on the versions of dependencies.*

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
                                [--simplify]
   ```

   Afterwards, you can see that a new pair of ONNX models ("best.onnx" and "best.sim.onnx") has been generated in the directory of "best.pt".

1. \[Optional\] Quick test with the ONNXRuntime Python interface.

   ```python
   from yolort.runtime import PredictorORT

   detector = PredictorORT('best.sim.onnx')
   img_path = 'bus.jpg'
   scores, class_ids, boxes = detector.run_on_image(img_path)
   ```

1. Now, you can infer your own images.

   ```bash
   ./yolort_onnx [--image ../../../test/assets/zidane.jpg]
                 [--model_path ../../../notebooks/yolov5s.onnx]
                 [--class_names ../../../notebooks/assets/coco.names]
                 [--gpu]  # GPU switch, which is optional, and set False as default
   ```
