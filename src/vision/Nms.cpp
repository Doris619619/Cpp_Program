#include "vision/Types.h"
#include <algorithm>
#include <cmath>

namespace vision {

// 简易 IoU
static float iouRect(const cv::Rect& a, const cv::Rect& b) {
    int inter_x = std::max(a.x, b.x);
    int inter_y = std::max(a.y, b.y);
    int inter_w = std::min(a.x + a.width, b.x + b.width) - inter_x;
    int inter_h = std::min(a.y + a.height, b.y + b.height) - inter_y;
    if (inter_w <=0 || inter_h<=0) return 0.f;
    float inter = inter_w * inter_h;
    float ua = a.width * a.height + b.width * b.height - inter;
    return ua<=0 ? 0.f : inter / ua;
}

// 占位 NMS：同类别，简单阈值抑制
template<typename BoxT>
static std::vector<BoxT> nmsClasswise(std::vector<BoxT> boxes, float iou_thres) {
    std::sort(boxes.begin(), boxes.end(), [](const BoxT& a, const BoxT& b){
        return a.conf > b.conf;
    });
    std::vector<BoxT> kept;
    std::vector<bool> removed(boxes.size(), false);
    for (size_t i=0;i<boxes.size();++i) {
        if (removed[i]) continue;
        kept.push_back(boxes[i]);
        for (size_t j=i+1;j<boxes.size();++j) {
            if (removed[j]) continue;
            if (boxes[i].cls_id == boxes[j].cls_id &&
                iouRect(boxes[i].rect, boxes[j].rect) > iou_thres) {
                removed[j] = true;
            }
        }
    }
    return kept;
}

} // namespace vision