#pragma once
#include <opencv2/core.hpp>
#include <memory>

namespace vision {

struct Mog2Config {
    int history = 500;
    int var_threshold = 16;
    bool detect_shadows = false;
};

class Mog2Manager {
public:
    Mog2Manager(const Mog2Config& cfg);
    cv::Mat apply(const cv::Mat& bgr); // 返回前景掩码
    float ratioInRoi(const cv::Mat& fg, const cv::Rect& roi) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision