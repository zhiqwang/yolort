#include <cuda_runtime_api.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "cmdline.h"
#include "preprocess.h"

#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000
#define BATCH_SIZE 8

using namespace nvonnxparser;
using namespace nvinfer1;
double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
static inline int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names) {
  DIR* p_dir = opendir(p_dir_name);
  if (p_dir == nullptr) {
    return -1;
  }

  struct dirent* p_file = nullptr;
  while ((p_file = readdir(p_dir)) != nullptr) {
    if (strcmp(p_file->d_name, ".") != 0 && strcmp(p_file->d_name, "..") != 0) {
      // std::string cur_file_name(p_dir_name);
      // cur_file_name += "/";
      // cur_file_name += p_file->d_name;
      std::string cur_file_name(p_file->d_name);
      file_names.push_back(cur_file_name);
    }
  }

  closedir(p_dir);
  return 0;
}
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
  std::vector<Detection> detect_preprocessgpu(cv::Mat& images);
  std::vector<std::vector<Detection>> batch_inference_gpu(std::vector<cv::Mat>& imagess);

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
std::vector<Detection> YOLOv5Detector::detect_preprocessgpu(cv::Mat& image) {
  std::vector<Detection> result;
  // void* 用于指向int* float*
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

  cv::Size shape = image.size();

  /* prepare the host mem alloc */
  for (int32_t i = 0; i < engine->getNbBindings(); i++) {
    {
      std::cout << "  Bind[" << i << "] {"
                << "Name:" << engine->getBindingName(i)
                << ", Datatype:" << static_cast<int>(engine->getBindingDataType(i)) << ",Shape:(";
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
    // void* buffers[i] op&--> void**
    CHECK(cudaMalloc(&buffers[i], buffer_size * getElementSize(engine->getBindingDataType(i))));
    if (i == detection_boxes_index) {
      detection_boxes.resize(buffer_size);
    } else if (i == detection_scores_index) {
      detection_scores.resize(buffer_size);
    } else if (i == detection_labels_index) {
      detection_labels.resize(buffer_size);
    }
  }
  uint8_t* img_host = nullptr;
  uint8_t* img_device = nullptr;
  // yolov5 malloc the device but this yolort has malloc in the upper loop

  // CUDA_CHECK(cudaMalloc((void**)&buffers[0], 3 * net_img_height * net_img_width *
  // sizeof(float))); CUDA_CHECK(cudaMalloc((void**)&buffers[1], OUTPUT_SIZE * sizeof(float))); this
  // malloc the mem to shuffle the device and host
  CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
  CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
  size_t size_image = image.cols * image.rows * 3;
  memcpy(img_host, image.data, size_image);
  // binding indices inputshape (1,3,640,640)
  int32_t input_h = engine->getBindingDimensions(0).d[2];
  int32_t input_w = engine->getBindingDimensions(0).d[3];
  // float *buffer_idx = (float*)buffers[0];
  // buffer_idx point float* buffers()
  float scale =
      1 / (std::min((float)input_h / (float)shape.height, (float)input_w / (float)shape.width));
  CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));

  preprocess_kernel_img(
      img_device, image.cols, image.rows, (float*)buffers[0], input_h, input_w, stream);
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
    // reference address
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
std::vector<std::vector<Detection>> YOLOv5Detector::batch_inference_gpu(
    std::vector<cv::Mat>& images) {
  int images_len = images.size();

  // use for temp
  cv::Mat image;
  // the mem allocate
  std::vector<int32_t> num_detections;
  std::vector<float> detection_boxes;
  std::vector<float> detection_scores;
  std::vector<int> detection_labels;

  std::vector<std::vector<Detection>> result(images_len);
  std::vector<float> ratio_list(images_len);

  std::vector<void*> buffers(engine->getNbBindings());
  std::size_t batch_size = BATCH_SIZE;
  // getBindingDimensions [BCHW] get the input  H W
  int32_t input_h = engine->getBindingDimensions(0).d[2];
  int32_t input_w = engine->getBindingDimensions(0).d[3];
  // get the different scale
  for (int i = 0; i < images_len; ++i) {
    cv::Size shape = images[i].size();
    ratio_list[i] =
        1 / (std::min((float)input_h / (float)shape.height, (float)input_w / (float)shape.width));
  }
  int inputIndex = engine->getBindingIndex("images");
  int num_detections_index = engine->getBindingIndex("num_detections");
  int detection_boxes_index = engine->getBindingIndex("detection_boxes");
  int detection_scores_index = engine->getBindingIndex("detection_scores");
  int detection_labels_index = engine->getBindingIndex("detection_classes");

  for (int32_t i = 0; i < engine->getNbBindings(); i++) {
    {
      std::cout << "  Bind[" << i << "] {"
                << "Name:" << engine->getBindingName(i)
                << ", Datatype:" << static_cast<int>(engine->getBindingDataType(i)) << ",Shape:(";
      for (int j = 0; j < engine->getBindingDimensions(i).nbDims; j++) {
        std::cout << engine->getBindingDimensions(i).d[j] << ",";
      }
      std::cout << ")"
                << "}" << std::endl;
    }
    int buffer_size = 1;
    for (int j = 0; j < engine->getBindingDimensions(i).nbDims; j++) {
      buffer_size *= engine->getBindingDimensions(i).d[j];
    }
    // malloc all the data
    CHECK(cudaMalloc(&buffers[i], buffer_size * getElementSize(engine->getBindingDataType(i))));
    if (i == detection_boxes_index) {
      detection_boxes.resize(buffer_size);
    } else if (i == detection_scores_index) {
      detection_scores.resize(buffer_size);
    } else if (i == detection_labels_index) {
      detection_labels.resize(buffer_size);
    } else if (i == num_detections_index) {
      num_detections.resize(buffer_size);
    }
  }

  uint8_t* img_host = nullptr;
  uint8_t* img_device = nullptr;
  // init the store space except out of the mem boudary
  CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
  CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
  // std::cout<<buffer_size<<std::endl;
  float* buffer_idx = (float*)buffers[0];

  for (int i = 0; i < images_len; ++i) {
    image = images[i];
    // yolov5 malloc the device but this yolort has malloc in the upper loop

    // CUDA_CHECK(cudaMalloc((void**)&buffers[0], 3 * net_img_height * net_img_width *
    // sizeof(float))); CUDA_CHECK(cudaMalloc((void**)&buffers[1], OUTPUT_SIZE *
    // sizeof(float))); this malloc the mem to shuffle the device and host
    // the mem allocate
    size_t size_image = image.cols * image.rows * 3;
    // the memory address shifting
    size_t size_image_dst = input_h * input_w * 3;
    // copy the imagedata to the img_host
    memcpy(img_host, image.data, size_image);
    // copy data to device memory
    CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
    preprocess_kernel_img(img_device, image.cols, image.rows, buffer_idx, input_h, input_w, stream);
    buffer_idx += size_image_dst;
  }

  // context->enqueueV2(buffers.data(), stream, nullptr);
  // inference in batch
  context->enqueue(8, (void**)buffers.data(), stream, nullptr);

  // buffers contain 8batch result can copy by the cudaMemcpy (DevicetoHost)
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
          num_detections.data(),
          buffers[num_detections_index],
          num_detections.size() * getElementSize(engine->getBindingDataType(i)),
          cudaMemcpyDeviceToHost));
    }
  }

  //  free the buffers
  for (int i = 0; i < engine->getNbBindings(); ++i) {
    CHECK(cudaFree(buffers[i]));
  }

  // number j pic
  for (int j = 0; j < images_len; ++j) {
    image = images[j];
    int x_offset = (input_w * ratio_list[j] - image.cols) / 2;
    int y_offset = (input_h * ratio_list[j] - image.rows) / 2;

    for (int32_t i = 0; i < num_detections[j]; ++i) {
      result[j].emplace_back();
      Detection& detection = result[j].back();
      detection.box.x = detection_boxes[j * 100 * 4 + 4 * i] * ratio_list[j] - x_offset;
      detection.box.y = detection_boxes[j * 100 * 4 + 4 * i + 1] * ratio_list[j] - y_offset;
      detection.box.width =
          detection_boxes[j * 100 * 4 + 4 * i + 2] * ratio_list[j] - x_offset - detection.box.x;
      detection.box.height =
          detection_boxes[j * 100 * 4 + 4 * i + 3] * ratio_list[j] - y_offset - detection.box.y;
      detection.classId = detection_labels[j * 100 + i];
      detection.conf = detection_scores[j * 100 + i];
    }
  }

  std::cout << result.size() << std::endl;
  return result;
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
  double iStart, iElaps;
  cmd.add("int8", '\0', "Enable INT8 inference.");
  cmd.add("fp16", '\0', "Enable FP16 inference.");
  cmd.add<std::string>("model_path", 'm', "Path to onnx model.", true, "yolov5.onnx");
  cmd.add<std::string>("image", 'i', "Image source to be detected.", false, "bus.jpg");
  cmd.add<std::string>("class_names", 'c', "Path of dataset labels.", true, "coco.names");
  cmd.add<std::string>("images_folder", 'f', "Image file store folder", false, "images");
  cmd.add<int>("batch", 'b', "batch or not", true, 1);

  cmd.parse_check(argc, argv);
  std::string imagePath = cmd.get<std::string>("image");
  std::string modelPath = cmd.get<std::string>("model_path");
  std::string classNamesPath = cmd.get<std::string>("class_names");
  std::vector<std::string> classNames = loadNames(classNamesPath);
  std::string images_folder = cmd.get<std::string>("images_folder");
  int batch_infer = cmd.get<int>("batch");
  if (batch_infer == 1) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
      std::cerr << "Read image file fail: " << imagePath << std::endl;
      return -1;
    }
    YOLOv5Detector yolo_detector(modelPath.c_str(), cmd.exist("int8"), cmd.exist("fp16"));
    iStart = cpuSecond();
    std::vector<Detection> result = yolo_detector.detect(image);
    iElaps = cpuSecond() - iStart;
    printf("Time elapsed cpu %f sec\n", iElaps);
    std::cout << "Detected " << result.size() << " objects." << std::endl;

    visualizeDetection(image, result, classNames);

    cv::imwrite("result.jpg", image);

    return 0;
  } else {
    YOLOv5Detector yolo_detector_batch(modelPath.c_str(), cmd.exist("int8"), cmd.exist("fp16"));
    std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
    std::vector<std::string> file_names;
    if (read_files_in_dir(images_folder.c_str(), file_names) < 0) {
      std::cerr << "read_files_in_dir failed." << std::endl;
      return -1;
    }
    printf("read success");
    int fcount = 0;
    if ((int)file_names.size() < BATCH_SIZE) {
      for (int i = 0; i < BATCH_SIZE - (int)file_names.size(); ++i) {
        file_names.push_back(file_names[0]);
      }
    }
    for (int f = 0; f < BATCH_SIZE; f++) {
      cv::Mat img = cv::imread(images_folder + "/" + file_names[f]);
      if (img.empty()) {
        printf("there was some error in read picture");
        std::cout << images_folder + "/" + file_names[f] << std::endl;
        continue;
      }
      imgs_buffer[f] = img;
    }
    std::cout << imgs_buffer.size() << std::endl;
    std::vector<std::vector<Detection>> batch_result;

    iStart = cpuSecond();
    batch_result = yolo_detector_batch.batch_inference_gpu(imgs_buffer);
    iElaps = cpuSecond() - iStart;
    printf("Time elapsed batch %d gpu %f sec\n", BATCH_SIZE,iElaps);
    for (int i = 0; i < BATCH_SIZE; i++) {
      visualizeDetection(imgs_buffer[i], batch_result[i], classNames);
      std::string Name = "result" + std::to_string(i) + ".jpg";
      cv::imwrite(Name, imgs_buffer[i]);
    }
  }
}
