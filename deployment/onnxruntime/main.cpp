#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <codecvt>
#include <fstream>
#include <iostream>
#include "cmdline.h"

struct Detection {
  cv::Rect box;
  float conf{};
  int classId{};
};

//遍历文件夹，dir最后要加/，filespec中没有*号，示例格式（"Package/", ".txt"）
std::vector<std::string> getFiles(const char* dir, const char* filespec) {

    //执行查询
    if (1) {
        char cmd[100];
        system("md C:\\Temp");
        sprintf(cmd, "dir \"%s\" /b | findstr %s >> C:\\Temp\\filelist.dat", dir, filespec);
        system(cmd);
    }

    //获取结果
    std::vector<std::string> retstrs;
    {
        std::ifstream fin("C:\\Temp\\filelist.dat", std::ios::in);
        if (fin) {
            std::string str;
            while (getline(fin, str)) {
                if (str != "") retstrs.push_back(dir + str);
            }
        }
        fin.close();
    }

    //删除临时文件
    system("del /f /q C:\\Temp\\filelist.dat");
    system("rd /s /q C:\\Temp");

    return retstrs;
}


namespace utils {
std::wstring charToWstring(const char* str) {
  typedef std::codecvt_utf8<wchar_t> convert_type;
  std::wstring_convert<convert_type, wchar_t> converter;

  return converter.from_bytes(str);
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
} // namespace utils

class YOLOv5Detector {
 public:
  explicit YOLOv5Detector(std::nullptr_t){};
  YOLOv5Detector(const std::string& modelPath, const bool& isGPU);

  std::vector<Detection> detect(cv::Mat& image);
  std::vector<Detection> detect_batch(std::string& batch_image_path);

 private:
  Ort::Env env{nullptr};
  Ort::SessionOptions sessionOptions{nullptr};
  Ort::Session session{nullptr};

  static void preprocessing(cv::Mat& image, float* blob);
  static std::vector<Detection> postprocessing(
      cv::Mat& image,
      std::vector<Ort::Value>& outputTensors);

  static std::vector<Detection> batch_postprocessing(
      std::vector<cv::Mat>& image,
      std::vector<Ort::Value>& outputTensors);

  std::vector<const char*> inputNames;
  std::vector<const char*> outputNames;
};

YOLOv5Detector::YOLOv5Detector(const std::string& modelPath, const bool& isGPU = true) {
  env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
  sessionOptions = Ort::SessionOptions();

  std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
  auto cudaAvailable =
      std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
  OrtCUDAProviderOptions cudaOption;

  if (isGPU && (cudaAvailable == availableProviders.end())) {
    std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
    std::cout << "Inference device: CPU" << std::endl;
  } else if (isGPU && (cudaAvailable != availableProviders.end())) {
    std::cout << "Inference device: GPU" << std::endl;
    sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
  } else {
    std::cout << "Inference device: CPU" << std::endl;
  }

#ifdef _WIN32
  std::wstring w_modelPath = utils::charToWstring(modelPath.c_str());
  session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
  session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

  Ort::AllocatorWithDefaultOptions allocator;

  inputNames.push_back(session.GetInputName(0, allocator));

  for (int i = 0; i < 3; ++i)
    outputNames.push_back(session.GetOutputName(i, allocator));
}

void YOLOv5Detector::preprocessing(cv::Mat& image, float* blob) {
  cv::Mat floatImage;
  cv::cvtColor(image, floatImage, cv::COLOR_BGR2RGB);
  std::vector<cv::Mat> chw(image.channels());

  floatImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
  cv::Size imageSize{image.cols, image.rows};

  // hwc -> chw
  for (int i = 0; i < image.channels(); ++i) {
    chw[i] = cv::Mat(imageSize, CV_32FC1, blob + i * imageSize.width * imageSize.height);
  }
  cv::split(floatImage, chw);
}

std::vector<Detection> YOLOv5Detector::postprocessing(
    cv::Mat& image,
    std::vector<Ort::Value>& outputTensors) {
  const auto* scoresTensor = outputTensors[0].GetTensorData<float>();
  const auto* classIdsTensor = outputTensors[1].GetTensorData<int64_t>();
  const auto* boxesTensor = outputTensors[2].GetTensorData<float>();

  size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
  std::vector<Detection> detections;
  for (int i = 0; i < count; ++i) {
    Detection det;
    int x = (int)boxesTensor[i * 4];
    int y = (int)boxesTensor[i * 4 + 1];
    int width = (int)boxesTensor[i * 4 + 2] - x;
    int height = (int)boxesTensor[i * 4 + 3] - y;

    det.conf = scoresTensor[i];
    det.classId = (int)classIdsTensor[i];
    det.box = cv::Rect(x, y, width, height);
    detections.push_back(det);
  }

  return detections;
}

std::vector<Detection> YOLOv5Detector::detect(cv::Mat& image) {
  size_t inputTensorSize = image.rows * image.cols * image.channels();
  std::vector<int64_t> imageShape{image.channels(), image.rows, image.cols};
  auto* blob = new float[inputTensorSize];

  this->preprocessing(image, blob);

  std::vector<float> inputTensorValues(blob, blob + inputTensorSize);
  std::vector<Ort::Value> inputTensors;

  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  inputTensors.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo,
      inputTensorValues.data(),
      inputTensorValues.size(),
      imageShape.data(),
      imageShape.size()));

  std::vector<Ort::Value> outputTensors = this->session.Run(
      Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1, outputNames.data(), 3);

  std::cout << "输出容器的元素数目为: " << outputTensors.size() << std::endl;

  std::vector<Detection> result = this->postprocessing(image, outputTensors);

  delete[] blob;

  return result;
}



/***---------------------------------------------------***/


std::vector<Detection> YOLOv5Detector::batch_postprocessing(
    std::vector<cv::Mat>& image,
    std::vector<Ort::Value>& outputTensors) 
{
    const auto* scoresTensor = outputTensors[0].GetTensorData<float>();
    const auto* classIdsTensor = outputTensors[1].GetTensorData<int64_t>();
    const auto* boxesTensor = outputTensors[2].GetTensorData<float>();

    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<Detection> detections;
    for (int i = 0; i < count; ++i) {
        Detection det;
        int x = (int)boxesTensor[i * 4];
        int y = (int)boxesTensor[i * 4 + 1];
        int width = (int)boxesTensor[i * 4 + 2] - x;
        int height = (int)boxesTensor[i * 4 + 3] - y;

        det.conf = scoresTensor[i];
        det.classId = (int)classIdsTensor[i];
        det.box = cv::Rect(x, y, width, height);
        detections.push_back(det);
    }

    return detections;

    
}

std::vector<Detection> YOLOv5Detector::detect_batch(std::string& batch_image_path) {
    
    std::vector<Ort::Value> inputTensors;
    std::vector<float> inputTensorValues_global;
    
    std::vector<cv::Mat> image_list;

    cv::Mat img;

    //遍历文件夹
    const char* tmp = batch_image_path.c_str();//string convert to const char*
    auto files = getFiles(tmp, ".jpg");
    std::cout << "find " << files.size() << " files." << std::endl;

    for (int i = 0; i < files.size(); i++)
    {
        auto fi = files[i];
        std::cout << "fi:" <<fi<< std::endl;
        
        img = cv::imread(fi);
        size_t inputTensorSize = img.rows * img.cols * img.channels();

        std::cout << "inputTensorSize is: " << inputTensorSize << std::endl;

        std::vector<int64_t> imageShape{ img.channels(), img.rows, img.cols };
        auto* blob = new float[inputTensorSize];

        this->preprocessing(img, blob);

        std::vector<float> inputTensorValues(blob, blob + inputTensorSize);
        inputTensorValues_global.swap(inputTensorValues);

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorValues_global.data(),
            inputTensorValues_global.size(),
            imageShape.data(),
            imageShape.size()));

        image_list.push_back(img);

        delete[] blob;

    }

    std::cout << "inputTensors size is: "<<inputTensors.size() << std::endl;

    std::cout<< "infer..." << std::endl;

    if (inputTensors.data() == nullptr)
    {
        std::cout << "输入数据为空!" << std::endl;
    }

    std::vector<Ort::Value> outputTensors = this->session.Run(
        Ort::RunOptions{ nullptr }, inputNames.data(), inputTensors.data(), 1, outputNames.data(), 3);

    std::cout << "完成推理!" << std::endl;
    std::cout <<"输出容器的元素数目为: " <<outputTensors.size() << std::endl;

    //std::vector<Detection> result = this->batch_postprocessing(image_list, outputTensors);

    //批推理的后处理暂未添加
    std::vector<Detection> result;

    return result;
}

int main(int argc, char* argv[]) {
  cmdline::parser cmd;
  cmd.add<std::string>("model_path", 'm', "Path to onnx model.", true, "yolov5.onnx");
  cmd.add<std::string>("image", 'i', "image source to be detected,file for single, fold for batch.", true, "bus.jpg");
  cmd.add<std::string>("class_names", 'c', "Path of dataset labels.", true, "coco.names");
  cmd.add("gpu", '\0', "Enable cuda device or cpu.");
  cmd.add("batch", '\0', "Enable batch infer or single infer.");

  cmd.parse_check(argc, argv);

  bool isGPU = cmd.exist("gpu");
  std::string classNamesPath = cmd.get<std::string>("class_names");
  std::vector<std::string> classNames = utils::loadNames(classNamesPath);
  std::string image_path = cmd.get<std::string>("image");
  std::string modelPath = cmd.get<std::string>("model_path");


  if (classNames.empty()) {
    std::cout << "Empty class names file." << std::endl;
    return -1;
  }

  YOLOv5Detector detector{nullptr};
  try {
    detector = YOLOv5Detector(modelPath, isGPU);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }


  bool infer_flag = cmd.exist("batch");

  if (!infer_flag)
  {
      cv::Mat image = cv::imread(image_path);

      std::vector<Detection> result = detector.detect(image);

      utils::visualizeDetection(image, result, classNames);

      cv::imshow("result", image);
      // cv::imwrite("result.jpg", image);
      cv::waitKey(0);
  }
  else
  {
      
      std::cout << "begin batch_infer..." << std::endl;

      std::vector<Detection> result = detector.detect_batch(image_path);
 

  }



  return 0;
}
