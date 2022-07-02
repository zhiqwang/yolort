# YOLOv6 TensorRT Inference Example

`YOLOv6` 的 TensorRT 推理，目前仅支持 Windows 环境的使用。

## Dependencies

- TensorRT-8.4.0.6
- OpenCV4.1.1

## Usage

~图文版请查看 [Windows 上基于 TensorRT 的 YOLOv6 部署保姆级教程](https://mp.weixin.qq.com/s/oxWodmYtULp5KznSYI19wQ).~

1. 首先需要安装 TensorRT 和 OpenCV, 我们使用的是 TensorRT-8.4.0.6 和 OpenCV-4.1.1, 如果您安装的 TensorRT 和 OpenCV 版本不一致，需要在 CmakeLists 里面修改版本和路径。

1. 下载 yolort 源码:

   ```sh
   git clone https://github.com/zhiqwang/yolov5-rt-stack.git
   cd yolov5-rt-stack/deployment/tensorrt-yolov6
   ```

1. 导出 ONNX 的时候请务必包含 EfficientNMS 算子, 稍后我们会加入一个更清晰的流程. ~从 YOLOv6 官方地址下载 ONNX 模型，如 [yolov6n.onnx](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6n.onnx)~.

1. 生成序列化的 TensorRT 文件:

   ```sh
   cd build_model
   mkdir build && cd build
   cmake .. -G "Visual Studio 16 2019"
   cmake --build .
   ```

   并执行如下命令生成序列化的 TensorRT 文件:

   ```sh
   build_model.exe yolov6n.onnx yolov6n.plan
   ```

   注: 您也可以直接使用 `trtexec` 来生成。

1. 编译 YOLOv6 的 TensorRT 可执行文件:

   ```sh
   cd ../..
   mkdir build && cd build
   cmake .. -G "Visual Studio 16 2019"
   cmake --build .
   ```

1. 现在您可以使用上面的可执行文件来做推理啦！

   ```sh
   ./yolov6.exe -model_path ./yolov6n.plan -image_path ./demo1.jpg
   ```
