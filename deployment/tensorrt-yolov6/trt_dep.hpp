#ifndef _TRT_DEP_HPP_
#define _TRT_DEP_HPP_

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::string;
using std::vector;

using nvinfer1::ICudaEngine;
using nvinfer1::ILogger;
using Severity = nvinfer1::ILogger::Severity;

class Logger : public ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity != Severity::kINFO) {
      std::cout << msg << std::endl;
    }
  }
};

struct TrtDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;
using TrtSharedEnginePtr = std::shared_ptr<ICudaEngine>;

extern Logger gLogger;

TrtSharedEnginePtr shared_engine_ptr(ICudaEngine* ptr);
TrtSharedEnginePtr parse_to_engine(string onnx_path, bool use_fp16);
void serialize(TrtSharedEnginePtr engine, string save_path);
TrtSharedEnginePtr deserialize(string serpth);
void infer_with_engine(
    TrtSharedEnginePtr engine,
    float* data,
    int in_size,
    int out_size,
    float* prob);

#endif
