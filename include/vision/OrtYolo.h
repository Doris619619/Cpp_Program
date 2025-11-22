#pragma once
#include "D:/Coding/Cpp/Projects/CSC3002Proj/Vision_Core/thirdparty/onnxruntime/include/onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace vision {

    struct RawDet {         // 原始检测框数据结构(model output)
        float cx, cy, w, h;
        float conf;
        int cls_id;
    };

    class OrtYoloDetector {
    public:
        struct SessionOptions {
            std::string model_path = "data/models/yolov8n_640.onnx";
            int input_w = 640;
            int input_h = 640;
            bool fake_infer = true;
        };

    /*struct Env {
        这里env需要有哪些内容？env与模型是什么关系？需要实现哪些操作？
    };*/
    //Ort::Env env;//(ORT_LOGGING_LEVEL_WARNING, "YOLOv8n");  // log level: warning, env name: YOLOv8n
    //Ort::SessionOptions session_options;
    //session_options.SetIntraOpNumThreads(0);   // auto decide threads usage

        explicit OrtYoloDetector(const SessionOptions& opt);
        bool const isReady();
        std::vector<RawDet> infer(const cv::Mat& resized_rgb); // resized 640x640

    private:
        SessionOptions opt_;
        bool ready_ = false;

        // onnx runtime session
        Ort::Env env_;                          // env object
        Ort::SessionOptions session_options_;   // session options config
        std::unique_ptr<Ort::Session> session_; // session instance
    };

} // namespace vision