// 单一实现：支持 fake_infer 生成少量随机框，便于演示
#include "vision/OrtYolo.h"
#include <random>

namespace vision {

OrtYoloDetector::OrtYoloDetector(const Options& opt) : opt_(opt) {
    // 后续可在此初始化 ONNX Runtime 会话
    ready_ = true;
}

bool OrtYoloDetector::isReady() const { return ready_; }

std::vector<RawDet> OrtYoloDetector::infer(const cv::Mat& resized_rgb) {
    if (!opt_.fake_infer) {
        // TODO: 接入 ONNX Runtime 真实推理
        return {};
    }
    // fake 推理：随机生成 0~2 个检测框
    static std::mt19937 gen{123};
    std::uniform_real_distribution<float> uf(0.f, 1.f);
    std::vector<RawDet> dets;

    if (uf(gen) > 0.4f) { // person
        dets.push_back({uf(gen)*opt_.input_w, uf(gen)*opt_.input_h, 80.f, 120.f, 0.82f, 0});
    }
    if (uf(gen) > 0.7f) { // object
        dets.push_back({uf(gen)*opt_.input_w, uf(gen)*opt_.input_h, 60.f, 40.f, 0.63f, 1});
    }
    return dets;
}

} // namespace vision