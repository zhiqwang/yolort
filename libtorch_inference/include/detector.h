# pragma once

#include <memory>

#include <torch/script.h>
#include <torch/torch.h>

#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "utils.h"

class Detector {
 public:
  /***
   * @brief constructor
   * @param model_path - path of the TorchScript weight file
   * @param device_type - inference with CPU/GPU
   */
  Detector(const std::string& model_path, const torch::DeviceType& device_type);

  /***
   * @brief inference module
   * @param img - input image
   * @param conf_threshold - confidence threshold
   * @param iou_threshold - IoU threshold for nms
   * @return detection result - bounding box, score, class index
   */
  std::vector<Detection>
  Run(const cv::Mat& img, float conf_threshold, float iou_threshold);

 private:
  /***
   * @brief Padded resize
   * @param src - input image
   * @param dst - output image
   * @param out_size - desired output size
   * @return padding information - pad width, pad height and zoom scale
   */
  static std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst,
      const cv::Size& out_size = cv::Size(640, 640));

  static inline torch::Tensor GetBoundingBoxIoU(const torch::Tensor& box1,
      const torch::Tensor& box2);

  /***
   * @brief Performs Non-Maximum Suppression (NMS) on inference results
   * @param detections - inference results from the network,
   * example [1, 25200, 85], 85 = 4(xywh) + 1(obj conf) + 80(class score)
   * @param conf_thres - object confidence(objectness) threshold
   * @param iou_thres - IoU threshold for NMS algorithm
   * @return detections with shape: nx7 (batch_index, x1, y1, x2, y2, score, classification)
   */
  static torch::Tensor PostProcessing(const torch::Tensor& detections,
      float conf_thres = 0.4, float iou_thres = 0.6);

  /***
   * @brief Rescale coordinates to original input image
   * @param data - detection result after inference and nms
   * @param pad_w - width padding
   * @param pad_h - height padding
   * @param scale - zoom scale
   * @param img_shape - original input image shape
   * @return rescaled detections
   */
  static std::vector<Detection> ScaleCoordinates(const at::TensorAccessor<float, 2>& data,
      float pad_w, float pad_h, float scale, const cv::Size& img_shape);

  torch::jit::script::Module module_;
  torch::Device device_;
  bool half_;
};
