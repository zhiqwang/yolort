#include <cuda_runtime_api.h>
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
#include <opencv2/opencv.hpp>
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
  float conf{};
  int classId{};
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
  config->setMaxWorkspaceSize(1U << 18); // 1G
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

  IOptimizationProfile* profile = builder->createOptimizationProfile();
  if (!profile) {
    return nullptr;
  }
  {
    Dims dim = network->getInput(0)->getDimensions();
    const char* name = network->getInput(0)->getName();
    profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
    profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
    profile->setDimensions(
        name,
        OptProfileSelector::kMAX,
        Dims4(builder->getMaxBatchSize(), dim.d[1], dim.d[2], dim.d[3]));
    // profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, 3, 192, 320));
    // profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(1, 3, 256, 416));
    // profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(1, 3, 640, 640));
    if (profile->isValid()) {
      config->addOptimizationProfile(profile);
    } else {
      std::cout << "profile is invalid!\n" << std::endl;
      exit(-1);
    }
  }

  std::unique_ptr<IHostMemory> serializedModel{
      builder->buildSerializedNetwork(*network.get(), *config.get())};
  if (!serializedModel) {
    return nullptr;
  }

  std::unique_ptr<IRuntime> runtime{createInferRuntime(logger)};
  if (!runtime) {
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

  std::vector<Detection> detect(std::vector<cv::Mat>& image);

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
    : engine({CreateCudaEngineFromOnnx(logger, model_path, max_batch_size, enable_int8, enable_fp16)}),
      context({engine->createExecutionContext()}) {}

std::vector<Detection> detect(std::vector<cv::Mat>& image) {
  std::vector<Detection> result;
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

  YOLOv5Detector yolo_detector(modelPath.c_str(), cmd.exist("int8"), cmd.exist("fp16"));

  return 0;
}
