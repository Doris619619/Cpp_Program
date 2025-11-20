#include "vision/Mog2.h"
#include <opencv2/video/background_segm.hpp>

namespace vision {

struct Mog2Manager::Impl {
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2;
    Impl(const Mog2Config& cfg) {
        mog2 = cv::createBackgroundSubtractorMOG2(cfg.history,
                                                  cfg.var_threshold,
                                                  cfg.detect_shadows);
    }
};

// Mog2Manager initialization
Mog2Manager::Mog2Manager(const Mog2Config& cfg)
    : impl_(new Impl(cfg)) {}

cv::Mat Mog2Manager::apply(const cv::Mat& bgr) {
    cv::Mat fg;
    impl_->mog2->apply(bgr, fg);
    return fg;
}

float Mog2Manager::ratioInRoi(const cv::Mat& fg, const cv::Rect& roi) const {
    cv::Rect bounded = roi & cv::Rect(0,0, fg.cols, fg.rows);
    if (bounded.width <=0 || bounded.height <=0) return 0.f;
    cv::Mat sub = fg(bounded);
    int nonzero = cv::countNonZero(sub);
    int total = bounded.width * bounded.height;
    if (total == 0) return 0.f;
    return static_cast<float>(nonzero) / static_cast<float>(total);
}

} // namespace vision