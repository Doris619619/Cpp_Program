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

    // 获取上一帧的所有检测结果（人和物体）
    void getLastDetections(std::vector<BBox>& out_persons, std::vector<BBox>& out_objects) const;

    void setPublisher(Publisher* p); // 不持有 not set yet

    // 新增: 返回座位数量，避免为了统计而进行一次推理
    int seatCount() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision