#ifdef _WIN32 // or _MSC_VER, as you wish
#define NOMINMAX
#include <windows.h>
#endif

#include <chrono>
#include <iostream>
#include <memory>
#include "cmdline.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <torch/script.h>
#include <torch/torch.h>

#include <torchvision/ops/nms.h>
#include <torchvision/vision.h>

std::vector<std::string> LoadNames(const std::string& path) {
  // load class names
  std::vector<std::string> class_names;
  std::ifstream infile(path);
  if (infile.good()) {
    std::string line;
    while (getline(infile, line)) {
      class_names.emplace_back(line);
    }
    infile.close();
  } else {
    std::cerr << "ERROR: Failed to access class name path: " << path
              << "\n\tDoes the file exist? Permission to read it?\n";
  }

  return class_names;
}

torch::Tensor ReadImage(const std::string& loc) {
  // Read Image from the location of image
  cv::Mat img = cv::imread(loc);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  img.convertTo(img, CV_32FC3, 1.0f / 255.0f); // normalization 1/255

  // Convert image to tensor
  torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3});
  img_tensor = img_tensor.permute({2, 0, 1}); // Reshape to C x H x W

  return img_tensor.clone();
};

struct Detection {
  cv::Rect bbox;
  float score;
  int class_idx;
};

void OverlayBoxes(
    cv::Mat& img,
    const std::vector<Detection>& detections,
    const std::vector<std::string>& class_names,
    const std::string& img_name,
    bool label = true) {
  for (const auto& detection : detections) {
    const auto& box = detection.bbox;
    float score = detection.score;
    int class_idx = detection.class_idx;

    cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

    if (label) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2) << score;
      std::string s = class_names[class_idx] + " " + ss.str();

      auto font_face = cv::FONT_HERSHEY_DUPLEX;
      auto font_scale = 1.0;
      int thickness = 1;
      int baseline = 0;
      auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
      cv::rectangle(
          img,
          cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
          cv::Point(box.tl().x + s_size.width, box.tl().y),
          cv::Scalar(0, 0, 255),
          -1);
      cv::putText(
          img,
          s,
          cv::Point(box.tl().x, box.tl().y - 5),
          font_face,
          font_scale,
          cv::Scalar(255, 255, 255),
          thickness);
    }
  }

  cv::imwrite(img_name, img);
}

int main(int argc, char* argv[]) {
  cmdline::parser cmd;
  cmd.add<std::string>(
      "checkpoint", 'c', "path of the generated torchscript file", true, "yolov5.torchscript.pt");
  cmd.add<std::string>("input_source", 'i', "image source to be detected", true, "bus.jpg");
  cmd.add<std::string>("labelmap", 'l', "path of dataset labels", true, "coco.names");
  cmd.add("gpu", '\0', "Enable cuda device or cpu");

#ifdef _WIN32
  cmd.parse_check(GetCommandLineA());
#else
  cmd.parse_check(argc, argv);
#endif

  // check if gpu flag is set
  bool is_gpu = cmd.exist("gpu");

  // set device type - CPU/GPU
  torch::DeviceType device_type;
  if (torch::cuda::is_available() && is_gpu) {
    std::cout << "Set GPU mode" << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Set CPU mode" << std::endl;
    device_type = torch::kCPU;
  }

  // load class names from dataset for visualization
  std::string labelmap = cmd.get<std::string>("labelmap");
  std::vector<std::string> class_names = LoadNames(labelmap);
  if (class_names.empty()) {
    return -1;
  }

  // load input image
  std::string image_path = cmd.get<std::string>("input_source");
  if (std::ifstream(image_path).fail()) {
    std::cerr << "ERROR: Failed to access image file path: " << image_path
              << "\n\tDoes the file exist? Permission to read it?\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    std::cout << "Loading model" << std::endl;
    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::string weights = cmd.get<std::string>("checkpoint");
    if (std::ifstream(weights).fail()) {
      std::cerr << "ERROR: Failed to access checkpoint file path: " << weights
                << "\n\tDoes the file exist? Permission to read it?\n";
      return -1;
    }

    module = torch::jit::load(weights);
    module.to(device_type);
    module.eval();
    std::cout << "Model loaded" << std::endl;
  } catch (const torch::Error& e) {
    std::cout << "Error loading the model: " << e.what() << std::endl;
    return -1;
  } catch (const std::exception& e) {
    std::cout << "Other error: " << e.what() << std::endl;
    return -1;
  }

  // TorchScript models require a List[IValue] as input
  std::vector<torch::jit::IValue> inputs;

  // YOLO accepts a List[Tensor] as main input
  std::vector<torch::Tensor> images;

  torch::TensorOptions options = torch::TensorOptions{device_type};

  // Run once to warm up
  std::cout << "Run once on empty image" << std::endl;
  auto img_dummy = torch::rand({3, 416, 320}, options);

  images.push_back(img_dummy);
  inputs.push_back(images);

  auto output = module.forward(inputs);

  images.clear();
  inputs.clear();

  /*** Pre-process ***/
  auto start = std::chrono::high_resolution_clock::now();

  // Read image
  auto img = ReadImage(image_path);
  img = img.to(device_type);

  images.push_back(img);
  inputs.push_back(images);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Pre-process takes : " << duration.count() << " ms" << std::endl;

  // Run third times to warm up
  for (int i = 0; i < 3; i++) {
    output = module.forward(inputs);
  }

  /*** Inference ***/
  // TODO: add synchronize point
  start = std::chrono::high_resolution_clock::now();

  output = module.forward(inputs);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // It should be known that it takes longer time at first time
  std::cout << "Inference takes : " << duration.count() << " ms" << std::endl;

  auto detections = output.toTuple()->elements().at(1).toList().get(0).toGenericDict();
  std::cout << "Detected labels: " << detections.at("labels") << std::endl;
  std::cout << "Detected boxes: " << detections.at("boxes") << std::endl;
  std::cout << "Detected scores: " << detections.at("scores") << std::endl;

  return 0;
}
