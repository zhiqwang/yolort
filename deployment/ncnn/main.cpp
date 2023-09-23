#include "layer.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>

DEFINE_LAYER_CREATOR(LetterBox)

class AnchorGenerator : public ncnn::Layer {
 public:
  AnchorGenerator() {
    one_blob_only = true;
  }

  virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt)
      const {
    return 0;
  }
};

DEFINE_LAYER_CREATOR(AnchorGenerator)

class LogitsDecoder : public ncnn::Layer {
 public:
  LogitsDecoder() {
    one_blob_only = true;
  }

  virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt)
      const {
    return 0;
  }
};

DEFINE_LAYER_CREATOR(LogitsDecoder)

struct Object {
  cv::Rect_<float> rect;
  int label;
  float prob;
};

static inline float intersection_area(const Object& a, const Object& b) {
  cv::Rect_<float> inter = a.rect & b.rect;
  return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) {
  int i = left;
  int j = right;
  float p = faceobjects[(left + right) / 2].prob;

  while (i <= j) {
    while (faceobjects[i].prob > p)
      i++;

    while (faceobjects[j].prob < p)
      j--;

    if (i <= j) {
      // swap
      std::swap(faceobjects[i], faceobjects[j]);

      i++;
      j--;
    }
  }

#pragma omp parallel sections
  {
#pragma omp section
    {
      if (left < j)
        qsort_descent_inplace(faceobjects, left, j);
    }
#pragma omp section
    {
      if (i < right)
        qsort_descent_inplace(faceobjects, i, right);
    }
  }
}

static void qsort_descent_inplace(std::vector<Object>& objects) {
  if (objects.empty())
    return;

  qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(
    const std::vector<Object>& faceobjects,
    std::vector<int>& picked,
    float nms_thresh) {
  picked.clear();

  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = faceobjects[i].rect.area();
  }

  for (int i = 0; i < n; i++) {
    const Object& a = faceobjects[i];

    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++) {
      const Object& b = faceobjects[picked[j]];

      // intersection over union
      float inter_area = intersection_area(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      // float IoU = inter_area / union_area
      if (inter_area / union_area > nms_thresh)
        keep = 0;
    }

    if (keep)
      picked.push_back(i);
  }
}

static inline float sigmoid(float x) {
  return static_cast<float>(1.f / (1.f + exp(-x)));
}

static int detect(const cv::Mat& bgr, std::vector<Object>& objects) {
  ncnn::Net yolort;

  // yolort.opt.use_vulkan_compute = true;
  // yolort.opt.use_bf16_storage = true;

  yolort.register_custom_layer("AnchorGenerator", AnchorGenerator_layer_creator);
  yolort.register_custom_layer("LogitsDecoder", LogitsDecoder_layer_creator);

  yolort.load_param("yolort-opt.param");
  yolort.load_model("yolort-opt.bin");

  const int target_size = 640;
  const float score_thresh = 0.25f;
  const float nms_thresh = 0.45f;

  int img_w = bgr.cols;
  int img_h = bgr.rows;

  // letterbox pad to multiple of 32
  int w = img_w;
  int h = img_h;
  float scale = 1.f;
  if (w > h) {
    scale = (float)target_size / w;
    w = target_size;
    h = h * scale;
  } else {
    scale = (float)target_size / h;
    h = target_size;
    w = w * scale;
  }

  ncnn::Mat in =
      ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

  // pad to target_size rectangle
  // yolov5/utils/datasets.py letterbox
  int wpad = (w + 31) / 32 * 32 - w;
  int hpad = (h + 31) / 32 * 32 - h;
  ncnn::Mat in_pad;
  ncnn::copy_make_border(
      in,
      in_pad,
      hpad / 2,
      hpad - hpad / 2,
      wpad / 2,
      wpad - wpad / 2,
      ncnn::BORDER_CONSTANT,
      114.f);

  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  in_pad.substract_mean_normalize(0, norm_vals);

  ncnn::Extractor ex = yolort.create_extractor();

  ex.input("images", in_pad);

  std::vector<Object> anchors;

  // apply nms with nms_thresh
  std::vector<int> picked;
  nms_sorted_bboxes(anchors, picked, nms_thresh);

  int count = picked.size();

  objects.resize(count);
  for (int i = 0; i < count; i++) {
    objects[i] = anchors[picked[i]];

    // adjust offset to original unpadded
    float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
    float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
    float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
    float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

    // clip
    x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
    y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
    x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
    y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

    objects[i].rect.x = x0;
    objects[i].rect.y = y0;
    objects[i].rect.width = x1 - x0;
    objects[i].rect.height = y1 - y0;
  }

  return 0;
}

int main(int argc, char* argv[]) {
  cmdline::parser cmd;
  cmd.add<std::string>("model_path", 'm', "Path to ncnn model.", true, "yolov5.param");
  cmd.add<std::string>("image", 'i', "Image source to be detected.", true, "bus.jpg");
  cmd.add<std::string>("class_names", 'c', "Path of dataset labels.", true, "coco.names");
  cmd.add("gpu", '\0', "Enable cuda device or cpu.");

  cmd.parse_check(argc, argv);

  bool isGPU = cmd.exist("gpu");
  std::string classNamesPath = cmd.get<std::string>("class_names");
  std::vector<std::string> classNames = utils::loadNames(classNamesPath);
  std::string imagePath = cmd.get<std::string>("image");
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

  cv::Mat image = cv::imread(imagePath);

  std::vector<Detection> result = detector.detect(image);

  utils::visualizeDetection(image, result, classNames);

  cv::imshow("result", image);
  // cv::imwrite("result.jpg", image);
  cv::waitKey(0);

  return 0;
}
