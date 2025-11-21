#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace vision {

struct RawDet {
    float cx, cy, w, h;
    float conf;
    int cls_id;
};

class OrtYoloDetector {
public:
    struct Options {
        std::string model_path;
        int input_w = 640;
        int input_h = 640;
        bool fake_infer = true;
    };

    /*struct Env {
        这里env需要有哪些内容？env与模型是什么关系？需要实现哪些操作？
    };*/

    explicit OrtYoloDetector(const Options& opt);
    bool isReady() const;
    std::vector<RawDet> infer(const cv::Mat& resized_rgb); // resized 640x640

private:
    Options opt_;
    bool ready_ = false;
};

} // namespace vision