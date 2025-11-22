#pragma once
#include "Types.h"
#include "Config.h"
#include <opencv2/core.hpp>
#include <memory>

namespace vision {

class Publisher;
class VisionA {
public:
    explicit VisionA(const VisionConfig& cfg);  // constructor
    ~VisionA();                                 // destructor   

    std::vector<SeatFrameState> processFrame(const cv::Mat& bgr,
                                             int64_t ts_ms,
                                             int64_t frame_index = -1);

    void setPublisher(Publisher* p); // 不持有 not set yet

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision