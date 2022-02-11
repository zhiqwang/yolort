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
#include<io/image/image.h>

#include <torch/script.h>
#include <torch/torch.h>

#include <ops/nms.h>
#include <vision.h>

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
    }
    else {
        std::cerr << "ERROR: Failed to access class name path: " << path
            << "\n\tDoes the file exist? Permission to read it?\n";
    }

    return class_names;
}

torch::Tensor ReadImage(const std::string& loc) {
    // Read Image from the location of image
    cv::Mat img = cv::imread(loc);
    cv::resize(img, img, cv::Size(576, 2016));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3, 1.0f / 255.0f);

    // Convert image to tensor
    torch::Tensor img_tensor = torch::from_blob(img.data, { 1, img.rows, img.cols, img.channels() });
    img_tensor = img_tensor.permute({ 0, 3, 1, 2 }).contiguous(); // Reshape to B x C x H x W

    return img_tensor.clone();
};



int main(int argc, char* argv[]) {
    cmdline::parser cmd;
    cmd.add<std::string>(
        "checkpoint", 'c', "path of the generated torchscript file", true, "../model_file/best_0128.pt");
    cmd.add<std::string>("input_source", 'i', "image source to be detected", true, "../test_pic/11.png");
    cmd.add<std::string>("labelmap", 'l', "path of dataset labels", true, "../labels/coco.names");
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
    }
    else {
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
    }
    catch (const torch::Error& e) {
        std::cout << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cout << "Other error: " << e.what() << std::endl;
        return -1;
    }

    // TorchScript models require a List[IValue] as input
    std::vector<torch::jit::IValue> inputs;


    torch::TensorOptions options = torch::TensorOptions{ device_type };



    /*** Pre-process ***/
    auto start = std::chrono::high_resolution_clock::now();

    // Read image
    auto img = ReadImage(image_path);
    img = img.to(device_type);

    auto batch_of_tensors = torch::cat({ img,img,img,img ,img ,img ,img ,img ,img, img});

    inputs.push_back(batch_of_tensors);




    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Pre-process takes : " << duration.count() << " ms" << std::endl;




    for (int i = 0; i < 10; i++)
    {
        /*** Inference ***/
        // TODO: add synchronize point
        start = std::chrono::high_resolution_clock::now();

        auto output2 = module.forward(inputs);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // It should be known that it takes longer time at first time
        std::cout << "Inference takes : " << duration.count() << " ms" << std::endl;


         /*auto detections = output2.toTuple()->elements().at(1).toList().get(0).toGenericDict();
         std::cout << "Detected labels: " << detections.at("labels") << std::endl;
         std::cout << "Detected boxes: " << detections.at("boxes") << std::endl;
         std::cout << "Detected scores: " << detections.at("scores") << std::endl;*/

         auto detections = output2.toTuple();
         std::cout << ">>OKey,output2: " << detections<< std::endl;
    }
    return 0;
}
