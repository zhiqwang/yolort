#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <sys/time.h>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cmdline.h"

using namespace nvonnxparser;
using namespace nvinfer1;

#define CHECK(status)                                                \
  do {                                                               \
    auto ret = (status);                                             \
    if (ret != 0) {                                                  \
      std::cerr << __LINE__ << "Cuda failure: " << ret << std::endl; \
      abort();                                                       \
    }                                                                \
  } while (0)

class MyLogger : public ILogger {
 public:
  MyLogger() = default;
  virtual ~MyLogger() = default;
  void log(Severity severity, AsciiChar const* msg) noexcept override {
    // suppress info-level messages
    if (severity <= Severity::kWARNING)
      std::cout << msg << std::endl;
  }
};

struct Detection {
  cv::Rect box;
  float conf;
  int classId;
};

inline size_t getElementSize(nvinfer1::DataType t) noexcept {
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  return 0;
}

ICudaEngine* CreateCudaEngineFromOnnx(
    MyLogger& logger,
    const char* onnx_path,
    const int max_batch_size = 1,
    bool enable_int8 = false,
    bool enable_fp16 = false) {
  std::unique_ptr<IBuilder> builder{createInferBuilder(logger)};
  if (!builder) {
    std::cerr << "Create builder fault!" << std::endl;
    return nullptr;
  }
  builder->setMaxBatchSize(max_batch_size);

  std::cout << "Platform:" << std::endl;
  std::cout << "  DLACores: " << builder->getNbDLACores() << std::endl;
  std::cout << "  INT8: " << (builder->platformHasFastInt8() ? "YES" : "NO") << std::endl;
  std::cout << "  FP16: " << (builder->platformHasFastFp16() ? "YES" : "NO") << std::endl;

  uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  std::unique_ptr<INetworkDefinition> network{builder->createNetworkV2(flag)};
  if (!network) {
    return nullptr;
  }

  std::unique_ptr<IParser> parser{createParser(*network.get(), logger)};
  if (!parser) {
    return nullptr;
  }
  parser->parseFromFile(onnx_path, static_cast<int>(ILogger::Severity::kWARNING));
  if (parser->getNbErrors() > 0) {
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
      std::cout << parser->getError(i)->desc() << std::endl;
    }
  }

  std::unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
  if (!config) {
    return nullptr;
  }
  config->setMaxWorkspaceSize(20 * (1U << 20));
  config->setFlag(BuilderFlag::kGPU_FALLBACK);
  if (enable_int8) {
    if (builder->platformHasFastInt8()) {
      config->setFlag(BuilderFlag::kINT8);
      std::cout << "Inference data type: INT8." << std::endl;
    } else {
      std::cout << "Your platfrom DO NOT support INT8 inference." << std::endl;
    }
  } else if (enable_fp16) {
    if (builder->platformHasFastFp16()) {
      config->setFlag(BuilderFlag::kFP16);
      std::cout << "Inference data type: FP16." << std::endl;
    } else {
      std::cout << "Your platfrom DO NOT support FP16 inference." << std::endl;
    }
  } else {
    std::cout << "Inference data type: FP32." << std::endl;
  }

  if (builder->getNbDLACores() > 0) {
    config->setDefaultDeviceType(DeviceType::kDLA);
    config->setDLACore(builder->getNbDLACores());
  }

  // TODO: dynamic input，还没想好怎么搞
  // IOptimizationProfile* profile = builder->createOptimizationProfile();
  // if (!profile) {
  //   return nullptr;
  // }
  //
  // {
  //   Dims dim = network->getInput(0)->getDimensions();
  //   const char* name = network->getInput(0)->getName();
  //   profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, dim.d[1], dim.d[2],
  //   dim.d[3])); profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(1, dim.d[1],
  //   dim.d[2], dim.d[3])); profile->setDimensions(
  //       name,
  //       OptProfileSelector::kMAX,
  //       Dims4(builder->getMaxBatchSize(), dim.d[1], dim.d[2], dim.d[3]));
  //   // profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, 3, 192, 320));
  //   // profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(1, 3, 256, 416));
  //   // profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(1, 3, 640, 640));
  //   if (profile->isValid()) {
  //     config->addOptimizationProfile(profile);
  //   } else {
  //     std::cout << "profile is invalid!\n" << std::endl;
  //     exit(-1);
  //   }
  // }

  std::unique_ptr<IHostMemory> serializedModel{
      builder->buildSerializedNetwork(*network.get(), *config.get())};
  if (!serializedModel) {
    std::cerr << "buildSerializedNetwork fail!" << std::endl;
    return nullptr;
  }

  std::unique_ptr<IRuntime> runtime{createInferRuntime(logger)};
  if (!runtime) {
    std::cerr << "createInferRuntime fail!" << std::endl;
    return nullptr;
  }

  {
    std::cout << "Network:" << std::endl;
    for (int i = 0; i < network->getNbInputs(); i++) {
      ITensor* input = network->getInput(i);
      std::cout << "  input_name[" << i << "]:" << input->getName() << std::endl;
    }
    for (int i = 0; i < network->getNbOutputs(); i++) {
      ITensor* output = network->getOutput(i);
      std::cout << "  output_name[" << i << "]:" << output->getName() << std::endl;
    }
  }

  return runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());
}

class YOLOv5Detector {
 public:
  YOLOv5Detector(
      const char* model_path,
      int max_batch_size = 1,
      bool enable_int8 = false,
      bool enable_fp16 = false);
  virtual ~YOLOv5Detector() = default;
  YOLOv5Detector(const YOLOv5Detector&) = delete;
  YOLOv5Detector& operator=(const YOLOv5Detector&) = delete;

  std::vector<Detection> detect(cv::Mat& image);
  std::vector<std::vector<Detection>> detect(std::vector<cv::Mat>& images);

 private:
  MyLogger logger;
  std::unique_ptr<ICudaEngine> engine;
  std::unique_ptr<IExecutionContext> context;
}; /* class YOLOv5Detector */

YOLOv5Detector::YOLOv5Detector(
    const char* model_path,
    int max_batch_size,
    bool enable_int8,
    bool enable_fp16)
    : engine(
          {CreateCudaEngineFromOnnx(logger, model_path, max_batch_size, enable_int8, enable_fp16)}),
      context({engine->createExecutionContext()}) {}

std::vector<Detection> YOLOv5Detector::detect(cv::Mat& image) {
  std::vector<Detection> result;
  std::size_t batch_size = 1;
  void* buffers[engine->getNbBindings()];
  int num_detections_index = engine->getBindingIndex("num_detections");
  int detection_boxes_index = engine->getBindingIndex("detection_boxes");
  int detection_scores_index = engine->getBindingIndex("detection_scores");
  int detection_labels_index = engine->getBindingIndex("detection_labels");

  int32_t num_detections = 0;
  float* detection_boxes = nullptr;
  float* detection_scores = nullptr;
  int32_t* detection_labels = nullptr;

  for (int32_t i = 0; i < engine->getNbBindings(); i++) {
    {
      /* Debug output */
      std::cout << "  Bind[" << i << "] {"
                << "Name:" << engine->getBindingName(i)
                << ", Datatype:" << static_cast<int>(engine->getBindingDataType(i)) << ", Shape:(";
      for (int j = 0; j < engine->getBindingDimensions(i).nbDims; j++) {
        std::cout << engine->getBindingDimensions(i).d[j] << ",";
      }
      std::cout << ")"
                << "}" << std::endl;
    }
    Dims dim = engine->getBindingDimensions(i);
    size_t buffer_size = batch_size;
    // FIXME: 此处如果为 dynamic input，部分形状为 -1
    for (int j = 1; j < engine->getBindingDimensions(i).nbDims; j++) {
      buffer_size *= engine->getBindingDimensions(i).d[j];
    }
    CHECK(cudaMalloc(&buffers[i], buffer_size * getElementSize(engine->getBindingDataType(i))));
    if (i == detection_boxes_index) {
      detection_boxes = new float[buffer_size];
    } else if (i == detection_scores_index) {
      detection_scores = new float[buffer_size];
    } else if (i == detection_labels_index) {
      detection_labels = new int32_t[buffer_size];
    }
  }

  /* Dims == > NCHW */
  int32_t input_h = engine->getBindingDimensions(0).d[2];
  int32_t input_w = engine->getBindingDimensions(0).d[3];
  cudaStream_t stream; /* XXX: 此处应该可以直接声明为类成员变量？ */
  CHECK(cudaStreamCreate(&stream));
  cv::resize(image, image, cv::Size(input_w, input_h));
  image.convertTo(image, CV_32FC3);
  CHECK(cudaMemcpyAsync(buffers[0], image.data, image.total(), cudaMemcpyHostToDevice, stream));
  context->enqueueV2(buffers, stream, nullptr);

  for (int32_t i = 1; i < engine->getNbBindings(); i++) {
    size_t buffer_size = batch_size;
    // FIXME: 此处如果为 dynamic input，部分形状为 -1
    for (int j = 1; j < engine->getBindingDimensions(i).nbDims; j++) {
      buffer_size *= engine->getBindingDimensions(i).d[j];
    }
    if (i == detection_boxes_index) {
      CHECK(cudaMemcpyAsync(
          detection_boxes,
          buffers[detection_boxes_index],
          buffer_size * getElementSize(engine->getBindingDataType(i)),
          cudaMemcpyDeviceToHost,
          stream));
    } else if (i == detection_scores_index) {
      CHECK(cudaMemcpyAsync(
          detection_scores,
          buffers[detection_scores_index],
          buffer_size * getElementSize(engine->getBindingDataType(i)),
          cudaMemcpyDeviceToHost,
          stream));
    } else if (i == detection_labels_index) {
      CHECK(cudaMemcpyAsync(
          detection_labels,
          buffers[detection_labels_index],
          buffer_size * getElementSize(engine->getBindingDataType(i)),
          cudaMemcpyDeviceToHost,
          stream));
    } else if (i == num_detections_index) {
      CHECK(cudaMemcpyAsync(
          &num_detections,
          buffers[num_detections_index],
          buffer_size * getElementSize(engine->getBindingDataType(i)),
          cudaMemcpyDeviceToHost,
          stream));
    }
  }

  cudaStreamDestroy(stream);

  for (int i = 0; i < engine->getNbBindings(); ++i) {
    CHECK(cudaFree(buffers[i]));
  }

  for (int32_t i = 0; i < num_detections; i++) {
    Detection detection;
    detection.box.x = detection_boxes[4 * i];
    detection.box.y = detection_boxes[4 * i + 1];
    detection.box.width = detection_boxes[4 * i + 2];
    detection.box.height = detection_boxes[4 * i + 3];
    detection.classId = detection_labels[i];
    detection.conf = detection_scores[i];
    result.push_back(detection);
  }

  delete[] detection_boxes;
  delete[] detection_scores;
  delete[] detection_labels;

  return result;
}

int main(int argc, char* argv[]) {
  cmdline::parser cmd;
  cmd.add("int8", '\0', "Enable INT8 inference.");
  cmd.add("fp16", '\0', "Enable FP16 inference.");
  cmd.add<std::string>("model_path", 'm', "Path to onnx model.", true, "yolov5.onnx");
  cmd.add<std::string>("image", 'i', "Image source to be detected.", true, "bus.jpg");
  cmd.add<std::string>("class_names", 'c', "Path of dataset labels.", true, "coco.names");

  cmd.parse_check(argc, argv);
  std::string imagePath = cmd.get<std::string>("image");
  std::string modelPath = cmd.get<std::string>("model_path");

  cv::Mat image = cv::imread(modelPath);
  YOLOv5Detector yolo_detector(modelPath.c_str(), cmd.exist("int8"), cmd.exist("fp16"));
  std::vector<Detection> result = yolo_detector.detect(image);

  std::cout << "Detected " << result.size() << " objects." << std::endl;

  return 0;
}
