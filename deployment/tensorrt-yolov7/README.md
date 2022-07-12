# YOLOv7 TensorRT Inference Example

`YOLOv7` 的 TensorRT 推理

## Usage

1. 导出YOLOv7 ONNX模型

   ```sh
   python models/export.py --weights ../yolov7-tiny.pt --grid
   ```

1. 下载 yolort 源码:

   ```sh
   git clone https://github.com/zhiqwang/yolov5-rt-stack.git
   python setup.py install 
   ```

1. 安装依赖

   ```sh
   pip install nvidia-pyindex
   pip install onnx_graphsurgeon
   ```

1. 为ONNX文件添加 EfficientNMS 算子，同时导出TenorRT序列化文件

   ```python
   from yolort.runtime.trt_helper import export_tensorrt_engine

   export_tensorrt_engine(
       checkpoint_path="yolov7-tiny.onnx",
       onnx_path="yolov7-tiny-nms.onnx",
       engine_path="yolov7-tiny.engine",
   )
   ```

1. 编译 YOLOv7 的 TensorRT 可执行文件:

   ```sh
   cd ../..
   mkdir build && cd build
   cmake ..
   make -j8
   ```

1. 现在您可以使用上面的可执行文件来做推理啦！

   ```sh
   ./yolov7 -model_path yolov7-tiny.engine -image_path ./demo1.jpg
   ```

## Reference

[triple-Mu](https://github.com/triple-Mu)

[Wulingtian](https://github.com/Wulingtian)
