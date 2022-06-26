#include "trt_dep.hpp"
#include <opencv2/opencv.hpp>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using nvinfer1::Dims2;
using nvinfer1::Dims3;
using nvinfer1::IBuilder;
using nvinfer1::IBuilderConfig;
using nvinfer1::ICudaEngine;
using nvinfer1::IExecutionContext;
using nvinfer1::IHostMemory;
using nvinfer1::IInt8Calibrator;
using nvinfer1::ILogger;
using nvinfer1::INetworkDefinition;
using nvinfer1::IRuntime;
using Severity = nvinfer1::ILogger::Severity;

using std::string;

using std::array;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios;
using std::ofstream;
using std::vector;

Logger gLogger;

TrtSharedEnginePtr shared_engine_ptr(ICudaEngine* ptr) {
  return TrtSharedEnginePtr(ptr, TrtDeleter());
}

TrtSharedEnginePtr parse_to_engine(string onnx_pth, bool use_fp16) {
  unsigned int maxBatchSize{1};
  int memory_limit = 1U << 30; // 1G

  auto builder = TrtUniquePtr<IBuilder>(nvinfer1::createInferBuilder(gLogger));
  if (!builder) {
    cout << "create builder failed\n";
    std::abort();
  }

  const auto explicitBatch =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = TrtUniquePtr<INetworkDefinition>(builder->createNetworkV2(explicitBatch));
  if (!network) {
    cout << "create network failed\n";
    std::abort();
  }

  auto config = TrtUniquePtr<IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    cout << "create builder config failed\n";
    std::abort();
  }

  auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
  if (!parser) {
    cout << "create parser failed\n";
    std::abort();
  }

  int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
  bool state = parser->parseFromFile(onnx_pth.c_str(), verbosity);
  if (!state) {
    cout << "parse model failed\n";
    std::abort();
  }

  config->setMaxWorkspaceSize(memory_limit);
  if (use_fp16 && builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16); // fp16
  }
  // TODO: see if use dla or int8

  auto output = network->getOutput(0);
  output->setType(nvinfer1::DataType::kINT32);

  TrtSharedEnginePtr engine = shared_engine_ptr(builder->buildEngineWithConfig(*network, *config));
  if (!engine) {
    cout << "create engine failed\n";
    std::abort();
  }

  return engine;
}

void serialize(TrtSharedEnginePtr engine, string save_path) {
  auto trt_stream = TrtUniquePtr<IHostMemory>(engine->serialize());
  if (!trt_stream) {
    cout << "serialize engine failed\n";
    std::abort();
  }

  ofstream ofile(save_path, ios::out | ios::binary);
  ofile.write((const char*)trt_stream->data(), trt_stream->size());

  ofile.close();
}

TrtSharedEnginePtr deserialize(string serpth) {
  ifstream ifile(serpth, ios::in | ios::binary);
  if (!ifile) {
    cout << "read serialized file failed\n";
    std::abort();
  }

  ifile.seekg(0, ios::end);
  const int mdsize = ifile.tellg();
  ifile.clear();
  ifile.seekg(0, ios::beg);
  vector<char> buf(mdsize);
  ifile.read(&buf[0], mdsize);
  ifile.close();
  cout << "model size: " << mdsize << endl;
  auto runtime = TrtUniquePtr<IRuntime>(nvinfer1::createInferRuntime(gLogger));
  TrtSharedEnginePtr engine =
      shared_engine_ptr(runtime->deserializeCudaEngine((void*)&buf[0], mdsize, nullptr));
  return engine;
}

void infer_with_engine(
    TrtSharedEnginePtr engine,
    float* data,
    int in_size,
    int out_size,
    float* prob) {
  Dims3 out_dims =
      static_cast<Dims3&&>(engine->getBindingDimensions(engine->getBindingIndex("outputs")));

  vector<void*> buffs(2);
  vector<float> res(out_size);

  auto context = TrtUniquePtr<IExecutionContext>(engine->createExecutionContext());
  if (!context) {
    cout << "create execution context failed\n";
    std::abort();
  }

  cudaError_t state;
  state = cudaMalloc(&buffs[0], in_size * sizeof(float));
  if (state) {
    cout << "allocate memory failed\n";
    std::abort();
  }
  state = cudaMalloc(&buffs[1], out_size * sizeof(float));
  if (state) {
    cout << "allocate memory failed\n";
    std::abort();
  }
  cudaStream_t stream;
  state = cudaStreamCreate(&stream);
  if (state) {
    cout << "create stream failed\n";
    std::abort();
  }

  state =
      cudaMemcpyAsync(buffs[0], &data[0], in_size * sizeof(float), cudaMemcpyHostToDevice, stream);
  if (state) {
    cout << "transmit to device failed\n";
    std::abort();
  }
  context->enqueueV2(&buffs[0], stream, nullptr);
  // context->enqueue(1, &buffs[0], stream, nullptr);
  state = cudaMemcpyAsync(prob, buffs[1], out_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
  if (state) {
    cout << "transmit to host failed \n";
    std::abort();
  }
  cudaStreamSynchronize(stream);

  cudaFree(buffs[0]);
  cudaFree(buffs[1]);
  cudaStreamDestroy(stream);
}
