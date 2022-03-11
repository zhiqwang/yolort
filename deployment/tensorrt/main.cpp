#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include "NvInfer.h"
#include "NvInferPlugin.h"
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

void visualizeDetection(
    cv::Mat& image,
    std::vector<Detection>& detections,
    const std::vector<std::string>& classNames) {
  for (const Detection& detection : detections) {
    cv::rectangle(image, detection.box, cv::Scalar(229, 160, 21), 2);

    int x = detection.box.x;
    int y = detection.box.y;

    int conf = (int)(detection.conf * 100);
    int classId = detection.classId;
    std::string label = classNames[classId] + " 0." + std::to_string(conf);

    int baseline = 0;
    cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.8, 2, &baseline);
    cv::rectangle(
        image, cv::Point(x, y - 25), cv::Point(x + size.width, y), cv::Scalar(229, 160, 21), -1);

    cv::putText(
        image, label, cv::Point(x, y - 3), cv::FONT_ITALIC, 0.8, cv::Scalar(255, 255, 255), 2);
  }
}

float letterbox(
    const cv::Mat& image,
    cv::Mat& out_image,
    const cv::Size& new_shape = cv::Size(640, 640),
    int stride = 32,
    const cv::Scalar& color = cv::Scalar(114, 114, 114),
    bool fixed_shape = false,
    bool scale_up = true) {
  cv::Size shape = image.size();
  float r = std::min(
      (float)new_shape.height / (float)shape.height, (float)new_shape.width / (float)shape.width);
  if (!scale_up) {
    r = std::min(r, 1.0f);
  }

  int newUnpad[2]{
      (int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};

  cv::Mat tmp;
  if (shape.width != newUnpad[0] || shape.height != newUnpad[1]) {
    cv::resize(image, tmp, cv::Size(newUnpad[0], newUnpad[1]));
  } else {
    tmp = image.clone();
  }

  float dw = new_shape.width - newUnpad[0];
  float dh = new_shape.height - newUnpad[1];

  if (!fixed_shape) {
    dw = (float)((int)dw % stride);
    dh = (float)((int)dh % stride);
  }

  dw /= 2.0f;
  dh /= 2.0f;

  int top = int(std::round(dh - 0.1f));
  int bottom = int(std::round(dh + 0.1f));
  int left = int(std::round(dw - 0.1f));
  int right = int(std::round(dw + 0.1f));
  cv::copyMakeBorder(tmp, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);

  return 1.0f / r;
}

std::vector<std::string> loadNames(const std::string& path) {
  // load class names
  std::vector<std::string> classNames;
  std::ifstream infile(path);
  if (infile.good()) {
    std::string line;
    while (getline(infile, line)) {
      classNames.emplace_back(line);
    }
    infile.close();
  } else {
    std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
  }

  return classNames;
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
  config->setMaxWorkspaceSize(100 * (1U << 20));
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

ICudaEngine* CreateCudaEngineFromSerializedModel(MyLogger& logger, const char* model_path) {
  std::cout << "Loading engine from file: " << model_path << std::endl;
  std::ifstream engineFile(model_path, std::ios::binary);
  if (!engineFile.is_open()) {
    std::cerr << "Open model file fail: " << model_path << std::endl;
    return nullptr;
  }
  engineFile.seekg(0, std::ifstream::end);
  int64_t fsize = engineFile.tellg();
  engineFile.seekg(0, std::ifstream::beg);

  std::vector<char> engineData(fsize);
  engineFile.read(engineData.data(), fsize);

  std::unique_ptr<IRuntime> runtime{createInferRuntime(logger)};
  if (!runtime) {
    std::cerr << "createInferRuntime fail!" << std::endl;
    return nullptr;
  }

  /* Must initLibNvInferPlugins if you use any plugin */
  initLibNvInferPlugins(&logger, "");

  return runtime->deserializeCudaEngine(engineData.data(), fsize);
}

ICudaEngine* CreateCudaEngineFromFile(
    MyLogger& logger,
    const std::string& file_path,
    const int max_batch_size,
    bool enable_int8,
    bool enable_fp16) {
  if (file_path.find_last_of(".onnx") == (file_path.size() - 1)) {
    return CreateCudaEngineFromOnnx(
        logger, file_path.c_str(), max_batch_size, enable_int8, enable_fp16);
  }
  /* All suffixes except .onnx will be treated as the TensorRT serialized engine. */
  return CreateCudaEngineFromSerializedModel(logger, file_path.c_str());
}

class YOLOv5Detector {
 public:
  YOLOv5Detector(
      const char* model_path,
      int max_batch_size = 1,
      bool enable_int8 = false,
      bool enable_fp16 = false);
  virtual ~YOLOv5Detector();
  YOLOv5Detector(const YOLOv5Detector&) = delete;
  YOLOv5Detector& operator=(const YOLOv5Detector&) = delete;

  std::vector<Detection> detect(cv::Mat& image);
  std::vector<std::vector<Detection>> detect(std::vector<cv::Mat>& images);

 private:
  MyLogger logger;
  std::unique_ptr<ICudaEngine> engine;
  std::unique_ptr<IExecutionContext> context;
  cudaStream_t stream;
}; /* class YOLOv5Detector */

YOLOv5Detector::YOLOv5Detector(
    const char* model_path,
    int max_batch_size,
    bool enable_int8,
    bool enable_fp16)
    : engine(
          {CreateCudaEngineFromFile(logger, model_path, max_batch_size, enable_int8, enable_fp16)}),
      context({engine->createExecutionContext()}) {
  CHECK(cudaStreamCreate(&stream));
}

YOLOv5Detector::~YOLOv5Detector() {
  if (stream) {
    CHECK(cudaStreamDestroy(stream));
  }
}

std::vector<Detection> YOLOv5Detector::detect(cv::Mat& image) {
  std::vector<Detection> result;
  std::vector<void*> buffers(engine->getNbBindings());
  std::size_t batch_size = 1;
  int num_detections_index = engine->getBindingIndex("num_detections");
  int detection_boxes_index = engine->getBindingIndex("detection_boxes");
  int detection_scores_index = engine->getBindingIndex("detection_scores");
  int detection_labels_index = engine->getBindingIndex("detection_classes");

  int32_t num_detections = 0;
  std::vector<float> detection_boxes;
  std::vector<float> detection_scores;
  std::vector<int> detection_labels;

  Dims dim = engine->getBindingDimensions(0);
  dim.d[0] = batch_size;
  context->setBindingDimensions(0, dim);

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

    size_t buffer_size = 1;
    for (int j = 0; j < engine->getBindingDimensions(i).nbDims; j++) {
      buffer_size *= engine->getBindingDimensions(i).d[j];
    }
    CHECK(cudaMalloc(&buffers[i], buffer_size * getElementSize(engine->getBindingDataType(i))));
    if (i == detection_boxes_index) {
      detection_boxes.resize(buffer_size);
    } else if (i == detection_scores_index) {
      detection_scores.resize(buffer_size);
    } else if (i == detection_labels_index) {
      detection_labels.resize(buffer_size);
    }
  }

  /* Dims == > NCHW */
  int32_t input_h = engine->getBindingDimensions(0).d[2];
  int32_t input_w = engine->getBindingDimensions(0).d[3];
  cv::Mat tmp;
  /* Fixed shape */
  float scale = letterbox(image, tmp, {input_w, input_h}, 32, {114, 114, 114}, true);
  cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
  tmp.convertTo(tmp, CV_32FC3, 1 / 255.0);
  {
    /* HWC ==> CHW */
    int offset = 0;
    std::vector<cv::Mat> split_images;
    cv::split(tmp, split_images);
    for (auto split_image : split_images) {
      CHECK(cudaMemcpyAsync(
          (float*)(buffers[0]) + offset,
          split_image.data,
          split_image.total() * sizeof(float),
          cudaMemcpyHostToDevice,
          stream));
      offset += split_image.total();
    }
  }
  context->enqueueV2(buffers.data(), stream, nullptr);

  for (int32_t i = 1; i < engine->getNbBindings(); i++) {
    if (i == detection_boxes_index) {
      CHECK(cudaMemcpy(
          detection_boxes.data(),
          buffers[detection_boxes_index],
          detection_boxes.size() * getElementSize(engine->getBindingDataType(i)),
          cudaMemcpyDeviceToHost));
    } else if (i == detection_scores_index) {
      CHECK(cudaMemcpy(
          detection_scores.data(),
          buffers[detection_scores_index],
          detection_scores.size() * getElementSize(engine->getBindingDataType(i)),
          cudaMemcpyDeviceToHost));
    } else if (i == detection_labels_index) {
      CHECK(cudaMemcpy(
          detection_labels.data(),
          buffers[detection_labels_index],
          detection_labels.size() * getElementSize(engine->getBindingDataType(i)),
          cudaMemcpyDeviceToHost));
    } else if (i == num_detections_index) {
      CHECK(cudaMemcpy(
          &num_detections,
          buffers[num_detections_index],
          getElementSize(engine->getBindingDataType(i)),
          cudaMemcpyDeviceToHost));
    }
  }

  for (int i = 0; i < engine->getNbBindings(); ++i) {
    CHECK(cudaFree(buffers[i]));
  }

  /* Convert box fromat from LTRB to LTWH */
  int x_offset = (input_w * scale - image.cols) / 2;
  int y_offset = (input_h * scale - image.rows) / 2;
  for (int32_t i = 0; i < num_detections; i++) {
    result.emplace_back();
    Detection& detection = result.back();
    detection.box.x = detection_boxes[4 * i] * scale - x_offset;
    detection.box.y = detection_boxes[4 * i + 1] * scale - y_offset;
    detection.box.width = detection_boxes[4 * i + 2] * scale - x_offset - detection.box.x;
    detection.box.height = detection_boxes[4 * i + 3] * scale - y_offset - detection.box.y;
    detection.classId = detection_labels[i];
    detection.conf = detection_scores[i];
  }

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
  std::string classNamesPath = cmd.get<std::string>("class_names");
  std::vector<std::string> classNames = loadNames(classNamesPath);

  cv::Mat image = cv::imread(imagePath);
  if (image.empty()) {
    std::cerr << "Read image file fail: " << imagePath << std::endl;
    return -1;
  }
  YOLOv5Detector yolo_detector(modelPath.c_str(), cmd.exist("int8"), cmd.exist("fp16"));
  std::vector<Detection> result = yolo_detector.detect(image);

  std::cout << "Detected " << result.size() << " objects." << std::endl;

  visualizeDetection(image, result, classNames);

  cv::imwrite("result.jpg", image);

  return 0;
}
