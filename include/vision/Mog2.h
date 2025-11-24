#pragma once
#include <opencv2/core.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc.hpp>

// 简化：移除 pimpl，直接持有 MOG2 指针，避免不完整类型删除问题

namespace vision {

struct Mog2Config {
    int history = 500;
    int var_threshold = 16;
    bool detect_shadows = false;
};

class Mog2Manager {
public:
    Mog2Manager(const Mog2Config& cfg);
    ~Mog2Manager() = default;
    cv::Mat apply(const cv::Mat& bgr); // 返回前景掩码
    float ratioInRoi(const cv::Mat& fg, const cv::Rect& roi) const;
    static float ratioInPoly(const cv::Mat& fg, const std::vector<cv::Point>& poly);
private:
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2_;
};

} // namespace vision

// 全局辅助：与类静态方法一致（便于外部直接调用）
float ratioInPoly(const cv::Mat& fg, const std::vector<cv::Point>& poly);