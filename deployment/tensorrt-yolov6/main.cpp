#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"

#include <opencv2/opencv.hpp>

#include <math.h>
#include <array>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "trt_dep.hpp"

using nvinfer1::IBuilder;
using nvinfer1::IBuilderConfig;
using nvinfer1::ICudaEngine;
using nvinfer1::IHostMemory;
using nvinfer1::IInt8Calibrator;
using nvinfer1::INetworkDefinition;

using nvinfer1::Dims2;
using nvinfer1::Dims3;
using nvinfer1::IExecutionContext;
using nvinfer1::ILogger;
using Severity = nvinfer1::ILogger::Severity;

using cv::Mat;
using std::array;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios;
using std::ofstream;
using std::string;
using std::vector;

TrtSharedEnginePtr engine = nullptr;
int iH, iW, O1, O2;
#define DEVICE 0 // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.3
static const int NUM_CLASSES = 80;

cv::Mat static_resize(cv::Mat& img) {
  float r = std::min(iW / (img.cols * 1.0), iH / (img.rows * 1.0));
  // r = std::min(r, 1.0f);
  int unpad_w = r * img.cols;
  int unpad_h = r * img.rows;
  cv::Mat re(unpad_h, unpad_w, CV_8UC3);
  cv::resize(img, re, re.size());
  cv::Mat out(iH, iW, CV_8UC3, cv::Scalar(114, 114, 114));
  re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
  return out;
}

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
    float nms_threshold) {
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
      if (inter_area / union_area > nms_threshold)
        keep = 0;
    }

    if (keep)
      picked.push_back(i);
  }
}

static void generate_yolox_proposals(
    int num_anchors,
    float* feat_blob,
    float prob_threshold,
    std::vector<Object>& objects) {
  std::cout << num_anchors << std::endl;
  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    int basic_pos = anchor_idx * (NUM_CLASSES + 5);
    float x_center = feat_blob[basic_pos + 0];
    float y_center = feat_blob[basic_pos + 1];
    float w = feat_blob[basic_pos + 2];
    float h = feat_blob[basic_pos + 3];
    float x0 = x_center - w * 0.5f;
    float y0 = y_center - h * 0.5f;

    float box_objectness = feat_blob[basic_pos + 4];
    for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++) {
      float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
      float box_prob = box_objectness * box_cls_score;
      if (box_prob > prob_threshold) {
        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = w;
        obj.rect.height = h;
        obj.label = class_idx;
        obj.prob = box_prob;
        objects.push_back(obj);
      }

    } // class loop

  } // point anchor loop
}

float* blobFromImage(cv::Mat& img) {
  float* blob = new float[img.total() * 3];
  int channels = 3;
  int img_h = img.rows;
  int img_w = img.cols;
  for (size_t c = 0; c < channels; c++) {
    for (size_t h = 0; h < img_h; h++) {
      for (size_t w = 0; w < img_w; w++) {
        blob[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c] / 255.0;
      }
    }
  }
  return blob;
}

static void decode_outputs(
    float* prob,
    std::vector<Object>& objects,
    float scale,
    const int img_w,
    const int img_h) {
  std::vector<Object> proposals;
  generate_yolox_proposals(O1, prob, BBOX_CONF_THRESH, proposals);
  std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

  qsort_descent_inplace(proposals);

  std::vector<int> picked;
  nms_sorted_bboxes(proposals, picked, NMS_THRESH);

  int count = picked.size();

  std::cout << "num of boxes: " << count << std::endl;

  objects.resize(count);
  for (int i = 0; i < count; i++) {
    objects[i] = proposals[picked[i]];

    // adjust offset to original unpadded
    float x0 = (objects[i].rect.x) / scale;
    float y0 = (objects[i].rect.y) / scale;
    float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
    float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

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
}

const float color_list[80][3] = {
    {0.000, 0.447, 0.741}, {0.850, 0.325, 0.098}, {0.929, 0.694, 0.125}, {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188}, {0.301, 0.745, 0.933}, {0.635, 0.078, 0.184}, {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600}, {1.000, 0.000, 0.000}, {1.000, 0.500, 0.000}, {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000}, {0.000, 0.000, 1.000}, {0.667, 0.000, 1.000}, {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000}, {0.333, 1.000, 0.000}, {0.667, 0.333, 0.000}, {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000}, {1.000, 0.333, 0.000}, {1.000, 0.667, 0.000}, {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500}, {0.000, 0.667, 0.500}, {0.000, 1.000, 0.500}, {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500}, {0.333, 0.667, 0.500}, {0.333, 1.000, 0.500}, {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500}, {0.667, 0.667, 0.500}, {0.667, 1.000, 0.500}, {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500}, {1.000, 0.667, 0.500}, {1.000, 1.000, 0.500}, {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000}, {0.000, 1.000, 1.000}, {0.333, 0.000, 1.000}, {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000}, {0.333, 1.000, 1.000}, {0.667, 0.000, 1.000}, {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000}, {0.667, 1.000, 1.000}, {1.000, 0.000, 1.000}, {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000}, {0.333, 0.000, 0.000}, {0.500, 0.000, 0.000}, {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000}, {1.000, 0.000, 0.000}, {0.000, 0.167, 0.000}, {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000}, {0.000, 0.667, 0.000}, {0.000, 0.833, 0.000}, {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167}, {0.000, 0.000, 0.333}, {0.000, 0.000, 0.500}, {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833}, {0.000, 0.000, 1.000}, {0.000, 0.000, 0.000}, {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286}, {0.429, 0.429, 0.429}, {0.571, 0.571, 0.571}, {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857}, {0.000, 0.447, 0.741}, {0.314, 0.717, 0.741}, {0.50, 0.5, 0}};

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects) {
  static const char* class_names[] = {"person",        "bicycle",      "car",
                                      "motorcycle",    "airplane",     "bus",
                                      "train",         "truck",        "boat",
                                      "traffic light", "fire hydrant", "stop sign",
                                      "parking meter", "bench",        "bird",
                                      "cat",           "dog",          "horse",
                                      "sheep",         "cow",          "elephant",
                                      "bear",          "zebra",        "giraffe",
                                      "backpack",      "umbrella",     "handbag",
                                      "tie",           "suitcase",     "frisbee",
                                      "skis",          "snowboard",    "sports ball",
                                      "kite",          "baseball bat", "baseball glove",
                                      "skateboard",    "surfboard",    "tennis racket",
                                      "bottle",        "wine glass",   "cup",
                                      "fork",          "knife",        "spoon",
                                      "bowl",          "banana",       "apple",
                                      "sandwich",      "orange",       "broccoli",
                                      "carrot",        "hot dog",      "pizza",
                                      "donut",         "cake",         "chair",
                                      "couch",         "potted plant", "bed",
                                      "dining table",  "toilet",       "tv",
                                      "laptop",        "mouse",        "remote",
                                      "keyboard",      "cell phone",   "microwave",
                                      "oven",          "toaster",      "sink",
                                      "refrigerator",  "book",         "clock",
                                      "vase",          "scissors",     "teddy bear",
                                      "hair drier",    "toothbrush"};

  cv::Mat image = bgr.clone();
  for (size_t i = 0; i < objects.size(); i++) {
    const Object& obj = objects[i];

    // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,obj.rect.x,
    // obj.rect.y, obj.rect.width, obj.rect.height);

    cv::Scalar color =
        cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
    float c_mean = cv::mean(color)[0];
    cv::Scalar txt_color;
    if (c_mean > 0.5) {
      txt_color = cv::Scalar(0, 0, 0);
    } else {
      txt_color = cv::Scalar(255, 255, 255);
    }

    cv::rectangle(image, obj.rect, color * 255, 2);

    char text[256];
    sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

    cv::Scalar txt_bk_color = color * 0.7 * 255;

    int x = obj.rect.x;
    int y = obj.rect.y + 1;
    // int y = obj.rect.y - label_size.height - baseLine;
    if (y > image.rows)
      y = image.rows;
    // if (x + label_size.width > image.cols)
    // x = image.cols - label_size.width;

    cv::rectangle(
        image,
        cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
        txt_bk_color,
        -1);

    cv::putText(
        image,
        text,
        cv::Point(x, y + label_size.height),
        cv::FONT_HERSHEY_SIMPLEX,
        0.4,
        txt_color,
        1);
  }

  cv::imwrite("det_res.jpg", image);
  fprintf(stderr, "save vis file\n");
  /* cv::imshow("image", image); */
  /* cv::waitKey(0); */
}

extern "C" __declspec(dllexport) void __cdecl Init(char* model_path);
__declspec(dllexport) void __cdecl Init(char* model_path) {
  engine = deserialize(model_path);
  Dims3 i_dims =
      static_cast<Dims3&&>(engine->getBindingDimensions(engine->getBindingIndex("image_arrays")));
  Dims3 o_dims =
      static_cast<Dims3&&>(engine->getBindingDimensions(engine->getBindingIndex("outputs")));
  iH = i_dims.d[2];
  iW = i_dims.d[3];
  O1 = o_dims.d[1];
  O2 = o_dims.d[2];
}

extern "C" __declspec(
    dllexport) void __cdecl Infer(int aWidth, int aHeight, int aChannel, unsigned char* aBytes);
__declspec(
    dllexport) void __cdecl Infer(int aWidth, int aHeight, int aChannel, unsigned char* aBytes) {
  cv::Mat img(aHeight, aWidth, CV_MAKETYPE(CV_8U, aChannel), aBytes);
  int img_w = img.cols;
  int img_h = img.rows;
  cv::Mat pr_img = static_resize(img);
  // cv::imwrite("pr_img.jpg", pr_img);
  float* blob;
  blob = blobFromImage(pr_img);
  float scale = std::min(iW / (img.cols * 1.0), iH / (img.rows * 1.0));
  auto in_dims = engine->getBindingDimensions(0);
  auto in_size = 1;
  for (int j = 0; j < in_dims.nbDims; j++) {
    // std::cout << in_dims.d[j] << std::endl;
    in_size *= in_dims.d[j];
  }
  auto out_dims = engine->getBindingDimensions(1);
  auto out_size = 1;
  for (int j = 0; j < out_dims.nbDims; j++) {
    out_size *= out_dims.d[j];
  }
  static float* prob = new float[out_size];
  memset(prob, 0, sizeof(float) * out_size);
  infer_with_engine(engine, blob, in_size, out_size, prob);
  std::vector<Object> objects;
  decode_outputs(prob, objects, scale, img_w, img_h);
  draw_objects(img, objects);
  // delete the pointer to the float
  delete blob;
}

extern "C" __declspec(dllexport) void __cdecl Destroy();
__declspec(dllexport) void __cdecl Destroy() {
  engine->destroy();
}
int main(int argc, char** argv) {
  if (argc == 5 && std::string(argv[1]) == "-model_path" && std::string(argv[3]) == "-image_path") {
    char* model_path = argv[2];
    char* image_path = argv[4];
    Init(model_path);
    cv::Mat img = cv::imread(image_path);
    Infer(img.cols, img.rows, img.channels(), img.data);

  } else {
    std::cerr << "--> arguments not right!" << std::endl;
    std::cerr << "--> yolov6.exe -model_path ./output.trt -image_path ./demo1.jpg" << std::endl;
    return -1;
  }
}
