#include <iostream>
#include <codecvt>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>


struct Detection
{
    cv::Rect box;
    float conf{};
    int classId{};
};

namespace utils
{
    std::wstring charToWstring(const char* str)
    {
        typedef std::codecvt_utf8<wchar_t> convert_type;
        std::wstring_convert<convert_type, wchar_t> converter;

        return converter.from_bytes(str);
    }

    void visualizeDetection(cv::Mat& image, std::vector<Detection>& detections, const std::vector<std::string>& classNames)
    {
        for (const Detection& detection : detections)
        {
            cv::rectangle(image, detection.box, cv::Scalar(229, 160, 21), 2);

            int x = detection.box.x;
            int y = detection.box.y;

            int conf = (int)(detection.conf * 100);
            int classId = detection.classId;
            std::string label = classNames[classId] + " 0." + std::to_string(conf);

            int baseline = 0;
            cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.8, 2, &baseline);
            cv::rectangle(image, cv::Point(x, y - 25), cv::Point(x + size.width, y), cv::Scalar(229, 160, 21), -1);

            cv::putText(image, label, cv::Point(x, y - 3), cv::FONT_ITALIC, 0.8, cv::Scalar(255, 255, 255), 2);
        }
    }
}

class Yolov5Detector
{
public:
    Yolov5Detector(const std::string& modelPath,
                   const std::string& device);

    std::vector<Detection> detect(cv::Mat& image);

private:
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};

    static void preprocessing(cv::Mat &image, float* blob);
    static std::vector<Detection> postprocessing(cv::Mat& image, std::vector<Ort::Value>& outputTensors);

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
};


Yolov5Detector::Yolov5Detector(const std::string& modelPath, const std::string& device = "cpu")
{
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();


#ifdef _WIN32
    std::wstring w_modelPath = utils::charToWstring(modelPath.c_str());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    if (device == "gpu" || device == "GPU" || device == "cuda" || device == "CUDA")
    {
        // TODO
    }
    Ort::AllocatorWithDefaultOptions allocator;

    const char* inputName = session.GetInputName(0, allocator);
    const char* outputName1 = session.GetOutputName(0, allocator);
    std::cout << "Output Name: " << outputName1 << std::endl;

    const char* outputName2 = session.GetOutputName(1, allocator);
    std::cout << "Output Name: " << outputName2 << std::endl;

    const char* outputName3 = session.GetOutputName(2, allocator);
    std::cout << "Output Name: " << outputName3 << std::endl;

    inputNames.push_back(inputName);
    outputNames.push_back(outputName1);
    outputNames.push_back(outputName2);
    outputNames.push_back(outputName3);

}

void Yolov5Detector::preprocessing(cv::Mat &image, float* blob)
{
    cv::Mat floatImage;
    cv::cvtColor(image, floatImage, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> chw(image.channels());

    floatImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    cv::Size imageSize {image.cols, image.rows};

    // hwc -> chw
    for (int i = 0; i < image.channels(); ++i)
    {
        chw[i] = cv::Mat(imageSize, CV_32FC1, blob + i * imageSize.width * imageSize.height);
    }
    cv::split(floatImage, chw);
}

std::vector<Detection> Yolov5Detector::postprocessing(cv::Mat& image, std::vector<Ort::Value>& outputTensors)
{
    const auto* scoresTensor = outputTensors[0].GetTensorData<float>();
    const auto* classIdsTensor = outputTensors[1].GetTensorData<float>();
    const auto* boxesTensor = outputTensors[2].GetTensorData<float>();

    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<Detection> detections;
    for (int i = 0; i < count; ++i)
    {
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

std::vector<Detection> Yolov5Detector::detect(cv::Mat &image)
{
    size_t inputTensorSize = image.rows * image.cols * image.channels();
    std::vector<int64_t> imageShape {image.channels(), image.rows, image.cols};
    auto* blob = new float[inputTensorSize];

    this->preprocessing(image, blob);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);
    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorValues.size(), imageShape.data(), imageShape.size()
    ));

    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                                                              inputTensors.data(), 1, outputNames.data(), 3);

    std::vector<Detection> result = this->postprocessing(image, outputTensors);

    delete[] blob;

    return result;
}

int main(int argc, char* argv[])
{
    std::cout << "in main" << std::endl;
    if (argc == 3)
    {
        std::cout << "Im here" << std::endl;
        std::cerr << "Usage: " << argv[0] << " model_path image_path" << std::endl;
    }

    std::string modelPath = "yolov5s_.simp.onnx"; // argv[1];
    std::string imagePath = "zidane.jpg"; // argv[2];

    const std::vector<std::string> classNames {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
            "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    Yolov5Detector detector(modelPath, "cpu");

    cv::Mat image = cv::imread(imagePath);
    std::vector<Detection> result = detector.detect(image);

    utils::visualizeDetection(image, result, classNames);

    cv::imshow("result", image);

    //cv::imwrite("result.jpg", image);
    cv::waitKey(0);

    return 0;
}
