#include "detector.h"


Detector::Detector(const std::string& model_path, const torch::DeviceType& device_type) : device_(device_type) {
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module_ = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model!\n";
        std::exit(EXIT_FAILURE);
    }

    half_ = (device_ != torch::kCPU);
    module_.to(device_);

    if (half_) {
        module_.to(torch::kHalf);
    }

    module_.eval();
}


std::vector<std::tuple<cv::Rect, float, int>>
Detector::Run(const cv::Mat& img, float conf_threshold, float iou_threshold) {
    torch::NoGradGuard no_grad;
    std::cout << "----------New Frame----------" << std::endl;

    /*** Pre-process ***/

    auto start = std::chrono::high_resolution_clock::now();

    // keep the original image for visualization purpose
    cv::Mat img_input = img.clone();
    std::vector<float> pad_info = LetterboxImage(img_input, img_input, cv::Size(640, 640));
    const float pad_w = pad_info[0];
    const float pad_h = pad_info[1];
    const float scale = pad_info[2];

    cv::cvtColor(img_input, img_input, cv::COLOR_BGR2RGB);  // BGR -> RGB
    img_input.convertTo(img_input, CV_32FC3, 1.0f / 255.0f);  // normalization 1/255
    auto tensor_img = torch::from_blob(img_input.data,
            {1, img_input.rows, img_input.cols, img_input.channels()}).to(device_);

    tensor_img = tensor_img.permute({0, 3, 1, 2}).contiguous();  // BHWC -> BCHW (Batch, Channel, Height, Width)

    if (half_) {
        tensor_img = tensor_img.to(torch::kHalf);
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(tensor_img);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // It should be known that it takes longer time at first time
    std::cout << "pre-process takes : " << duration.count() << " ms" << std::endl;

    /*** Inference ***/

    start = std::chrono::high_resolution_clock::now();

    // inference
    torch::jit::IValue output = module_.forward(inputs);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // It should be known that it takes longer time at first time
    std::cout << "inference takes : " << duration.count() << " ms" << std::endl;

    /*** Post-process ***/

    start = std::chrono::high_resolution_clock::now();
    auto detections = output.toTuple()->elements()[0].toTensor();

    // result: n * 7
    // batch index(0), top-left x/y (1,2), bottom-right x/y (3,4), score(5), class id(6)
    auto result = PostProcessing(detections, conf_threshold, iou_threshold);

    // Note - only the first image in the batch will be used for demo
    auto idx_mask = result * (result.select(1, 0) == 0).to(torch::kFloat32).unsqueeze(1);
    auto idx_mask_index =  torch::nonzero(idx_mask.select(1, 1)).squeeze();
    const auto& result_data_demo = result.index_select(0, idx_mask_index).slice(1, 1, 7);

    // use accessor to access tensor elements efficiently
    const auto& demo_data = result_data_demo.accessor<float, 2>();

    // remap to original image and list bounding boxes for debugging purpose
    std::vector<std::tuple<cv::Rect, float, int>> demo_data_vec;
    for (int i = 0; i < result.size(0) ; i++) {
        auto x1 = static_cast<int>((demo_data[i][Det::tl_x] - pad_w)/scale);
        auto y1 = static_cast<int>((demo_data[i][Det::tl_y] - pad_h)/scale);
        auto x2 = static_cast<int>((demo_data[i][Det::br_x] - pad_w)/scale);
        auto y2 = static_cast<int>((demo_data[i][Det::br_y] - pad_h)/scale);
        cv::Rect rect(cv::Point(x1, y1), cv::Point(x2, y2));
        std::tuple<cv::Rect, float, int> t = std::make_tuple(rect,
            demo_data[i][Det::score], demo_data[i][Det::class_idx]);
        demo_data_vec.emplace_back(t);
    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // It should be known that it takes longer time at first time
    std::cout << "post-process takes : " << duration.count() << " ms" << std::endl;

    return demo_data_vec;
}


std::vector<float> Detector::LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size) {
    auto in_h = static_cast<float>(src.rows);
    auto in_w = static_cast<float>(src.cols);
    float out_h = out_size.height;
    float out_w = out_size.width;

    float scale = std::min(out_w / in_w, out_h / in_h);

    int mid_h = static_cast<int>(in_h * scale);
    int mid_w = static_cast<int>(in_w * scale);

    cv::resize(src, dst, cv::Size(mid_w, mid_h));

    int top = (static_cast<int>(out_h) - mid_h) / 2;
    int down = (static_cast<int>(out_h)- mid_h + 1) / 2;
    int left = (static_cast<int>(out_w)- mid_w) / 2;
    int right = (static_cast<int>(out_w)- mid_w + 1) / 2;

    cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
    return pad_info;
}


// returns the IoU of bounding boxes
torch::Tensor Detector::GetBoundingBoxIoU(const torch::Tensor& box1, const torch::Tensor& box2) {
    // get the coordinates of bounding boxes
    const torch::Tensor& b1_x1 = box1.select(1, 0);
    const torch::Tensor& b1_y1 = box1.select(1, 1);
    const torch::Tensor& b1_x2 = box1.select(1, 2);
    const torch::Tensor& b1_y2 = box1.select(1, 3);

    const torch::Tensor& b2_x1 = box2.select(1, 0);
    const torch::Tensor& b2_y1 = box2.select(1, 1);
    const torch::Tensor& b2_x2 = box2.select(1, 2);
    const torch::Tensor& b2_y2 = box2.select(1, 3);

    // get the coordinates of the intersection rectangle
    torch::Tensor inter_rect_x1 =  torch::max(b1_x1, b2_x1);
    torch::Tensor inter_rect_y1 =  torch::max(b1_y1, b2_y1);
    torch::Tensor inter_rect_x2 =  torch::min(b1_x2, b2_x2);
    torch::Tensor inter_rect_y2 =  torch::min(b1_y2, b2_y2);

    // calculate intersection area
    torch::Tensor inter_area = torch::max(inter_rect_x2 - inter_rect_x1 + 1,torch::zeros(inter_rect_x2.sizes()))
                               * torch::max(inter_rect_y2 - inter_rect_y1 + 1, torch::zeros(inter_rect_x2.sizes()));

    // calculate union area
    torch::Tensor b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1);
    torch::Tensor b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1);

    // calculate IoU
    torch::Tensor iou = inter_area / (b1_area + b2_area - inter_area);

    return iou;
}


torch::Tensor Detector::PostProcessing(const torch::Tensor& detections, float conf_thres, float iou_thres) {
    constexpr int item_attr_size = 5;
    int batch_size = detections.size(0);
    auto num_classes = detections.size(2) - item_attr_size;  // 80 for coco dataset

    // get candidates which object confidence > threshold
    auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);

    // compute overall score = obj_conf * cls_conf, similar to x[:, 5:] *= x[:, 4:5]
    detections.slice(2, item_attr_size, item_attr_size + num_classes) *=
            detections.select(2, 4).unsqueeze(2);

    // convert bounding box format from (center x, center y, width, height) to (x1, y1, x2, y2)
    torch::Tensor box = torch::zeros(detections.sizes(), detections.options());
    box.select(2, Det::tl_x) = detections.select(2, 0) - detections.select(2, 2).div(2);
    box.select(2, Det::tl_y) = detections.select(2, 1) - detections.select(2, 3).div(2);
    box.select(2, Det::br_x) = detections.select(2, 0) + detections.select(2, 2).div(2);
    box.select(2, Det::br_y) = detections.select(2, 1) + detections.select(2, 3).div(2);
    detections.slice(2, 0, 4) = box.slice(2, 0, 4);

    bool is_initialized = false;
    torch::Tensor output = torch::zeros({0, 7});

    // iterating all images in the batch
    for (int batch_i = 0; batch_i < batch_size; batch_i++) {
        auto det = torch::masked_select(detections[batch_i],
            conf_mask[batch_i]).view({-1, num_classes + item_attr_size});

        // if none remain then process next image
        if (det.size(0) == 0) {
            continue;
        }

        // get the max classes score at each result (e.g. elements 5-84)
        std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(det.slice(1,
            item_attr_size, item_attr_size + num_classes), 1);

        // class score
        auto max_conf_score = std::get<0>(max_classes);
        // index
        auto max_conf_index = std::get<1>(max_classes);

        max_conf_score = max_conf_score.to(torch::kFloat32).unsqueeze(1);
        max_conf_index = max_conf_index.to(torch::kFloat32).unsqueeze(1);

        // shape: n * 6, top-left x/y (0,1), bottom-right x/y (2,3), score(4), class index(5)
        det = torch::cat({det.slice(1, 0, 4), max_conf_score, max_conf_index}, 1);

        // get unique classes
        std::vector<torch::Tensor> img_classes;

        auto len = det.size(0);
        for (int i = 0; i < len; i++) {
            bool found = false;
            for (const auto& cls : img_classes) {
                auto ret = (det[i][Det::class_idx] == cls);
                if (torch::nonzero(ret).size(0) > 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                img_classes.emplace_back(det[i][Det::class_idx]);
            }
        }

        // iterating all unique classes
        for (const auto& cls : img_classes) {
            auto cls_mask = det * (det.select(1, Det::class_idx) == cls).to(torch::kFloat32).unsqueeze(1);
            auto class_mask_index =  torch::nonzero(cls_mask.select(1, Det::score)).squeeze();
            auto bbox_by_class = det.index_select(0, class_mask_index).view({-1, 6});

            // sort by confidence (descending)
            std::tuple<torch::Tensor,torch::Tensor> sort_ret = torch::sort(bbox_by_class.select(1, 4), -1, true);
            auto conf_sort_index = std::get<1>(sort_ret);

            bbox_by_class = bbox_by_class.index_select(0, conf_sort_index.squeeze()).cpu();
            int num_by_class = bbox_by_class.size(0);

            // Non-Maximum Suppression (NMS)
            for(int i = 0; i < num_by_class - 1; i++) {
                auto iou = GetBoundingBoxIoU(bbox_by_class[i].unsqueeze(0),
                    bbox_by_class.slice(0, i + 1, num_by_class));
                auto iou_mask = (iou < iou_thres).to(torch::kFloat32).unsqueeze(1);

                bbox_by_class.slice(0, i + 1, num_by_class) *= iou_mask;

                // remove from list
                auto non_zero_index = torch::nonzero(bbox_by_class.select(1, 4)).squeeze();
                bbox_by_class = bbox_by_class.index_select(0, non_zero_index).view({-1, 6});
                // update remain number of detections
                num_by_class = bbox_by_class.size(0);
            }

            torch::Tensor batch_index = torch::zeros({bbox_by_class.size(0), 1}).fill_(batch_i);

            if (!is_initialized) {
                output = torch::cat({batch_index, bbox_by_class}, 1);
                is_initialized = true;
            }
            else {
                auto out = torch::cat({batch_index, bbox_by_class}, 1);
                output = torch::cat({output,out}, 0);
            }
        }
    }

    return output;
}
