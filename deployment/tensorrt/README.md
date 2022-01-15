# TensorRT Inference

The TensorRT inference for `yolort`, support CUDA only.

## Dependencies

- TensorRT 8.0 +

## Usage

1. Create build directory and cmake config.

   ```bash
   mkdir -p build/ && cd build/
   cmake .. -DTENSORRT_DIR={path/to/your/trt/install/director}
   ```

1. Build project

   ```bash
   cmake --build . -j4
   ```

1. Export your custom model to ONNX

   Here is a small demo to surgeon the YOLOv5 ONNX model and then export to TensorRT engine. For details see out our [tutorial for deploying yolort on TensorRT](https://zhiqwang.com/yolov5-rt-stack/notebooks/onnx-graphsurgeon-inference-tensorrt.html).

   - Set the super parameters

     ```python
     model_path = "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n6.pt"
     checkpoint_path = attempt_download(model_path)
     onnx_path = "yolov5n6.onnx"
     engine_path = "yolov5n6.engine"

     score_thresh = 0.4
     iou_thresh = 0.45
     detections_per_img = 100
     ```

   - Surgeon the yolov5 ONNX models

     ```python
     from yolort.runtime.yolo_graphsurgeon import YOLOGraphSurgeon

     yolo_gs = YOLOGraphSurgeon(
         checkpoint_path,
         version="r6.0",
         enable_dynamic=False,
     )

     yolo_gs.register_nms(
         score_thresh=score_thresh,
         nms_thresh=iou_thresh,
         detections_per_img=detections_per_img,
     )

     # Export the ONNX model
     yolo_gs.save(onnx_path)
     ```

   - Build the TensorRT engine

     ```python
     from yolort.runtime.trt_helper import EngineBuilder

     engine_builder = EngineBuilder()
     engine_builder.create_network(onnx_path)
     engine_builder.create_engine(engine_path, precision="fp32")
     ```

1. Now, you can infer your own images.

   ```bash
   ./yolort_trt [--image ../../../test/assets/zidane.jpg]
                [--model_path ../../../notebooks/yolov5s.onnx]
                [--class_names ../../../notebooks/assets/coco.names]
                [--fp16]  # Enable it if your GPU support fp16 inference
   ```
