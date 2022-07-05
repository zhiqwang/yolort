#include <fstream>
#include <iostream>
#include <string>
#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
 public:
  Logger(Severity severity = Severity::kWARNING) : reportableSeverity(severity) {}

  void log(Severity severity, char const* msg) noexcept
  // void log(Severity severity, const char* msg) noexcept
  {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportableSeverity)
      return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }
    std::cerr << msg << std::endl;
  }

  Severity reportableSeverity;
};

static Logger g_logger_;

void onnxToTRTModel(
    std::string onnx_file,
    std::string trt_file,
    nvinfer1::IHostMemory*& trt_model_stream) {
  int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;

  // create the builder
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(g_logger_);
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  auto parser = nvonnxparser::createParser(*network, g_logger_);

  if (!parser->parseFromFile(onnx_file.c_str(), verbosity)) {
    std::string msg("failed to parse onnx file");
    g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    exit(EXIT_FAILURE);
  }

  // Build the engine
  builder->setMaxBatchSize(1);
  // 创建iBuilderConfig对象
  nvinfer1::IBuilderConfig* iBuilderConfig = builder->createBuilderConfig();
  // 设置engine可使用的最大GPU临时值
  iBuilderConfig->setMaxWorkspaceSize((1 << 20) * 12);
  iBuilderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
  nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *iBuilderConfig);

  // 将engine序列化，保存到文件中
  trt_model_stream = engine->serialize();
  // save engine
  std::ofstream p(trt_file, std::ios::binary);
  if (!p) {
    std::cerr << "could not open plan output file" << std::endl;
  }
  p.write(reinterpret_cast<const char*>(trt_model_stream->data()), trt_model_stream->size());
  parser->destroy();
  engine->destroy();
  network->destroy();
  builder->destroy();
  iBuilderConfig->destroy();
}

int main(int argc, char** argv) {
  std::string onnx_file = argv[1];
  std::string trt_file = argv[2];

  nvinfer1::IHostMemory* trt_model_stream;
  onnxToTRTModel(onnx_file, trt_file, trt_model_stream);
}
