#include "vision/Snapshotter.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>

namespace vision {
/*     Snapshotter 快照器: 策略判定与保存 
*
* Snapshotter class implementation
*  - saveSnapshot: 根据策略决定是否保存快照，并执行保存
* 
* SnapshotPolicy struct defines the saving policy
*/

// Snapshotter initialization
Snapshotter::Snapshotter(const std::string& dir, const SnapshotPolicy& policy)
    : dir_(dir), policy_(policy) {
    std::filesystem::create_directories(dir_);
}

std::string Snapshotter::saveSnapshot(const std::string& seat_id,
                                   int state_hash,                      // State hash(?seat to state?)
                                   int64_t ts_ms,                       // Timestamp (ms)
                                   const cv::Mat& bgr,                  // BGR image
                                   const std::vector<cv::Rect>& boxes) {
    int64_t last_ts = last_snap_ts_[seat_id];                       // last snapshot timestamps
    int last_hash = last_state_hash_[seat_id];                      // last state hash
    bool changed = (state_hash != last_hash);                       // state changed or not
    bool due_heartbeat = policy_.heartbeat_ms > 0 &&
                         (ts_ms - last_ts >= policy_.heartbeat_ms);

    if (last_ts == 0) changed = true; // 第一次强制快照

    if (policy_.on_change_only && !changed && !due_heartbeat) {
        return "";
    }
    if (!policy_.on_change_only && (ts_ms - last_ts < policy_.min_interval_ms)) {
        return "";
    }
    if (policy_.on_change_only && !changed && due_heartbeat == false) {
        return "";
    }
    // 状态变且间隔不足也不保存
    if (changed && (ts_ms - last_ts < policy_.min_interval_ms)) {
        return "";
    }

    // 绘制框 (简单画线)
    cv::Mat draw = bgr.clone();
    //for (auto &r : boxes) {
        //cv::rectangle(draw, r, cv::Scalar(0,255,0), 2);
    //}
    std::string filename = "seat_" + seat_id + "_" + std::to_string(ts_ms) + ".jpg";
    std::string full = dir_ + "/" + filename;
    //std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, policy_.jpg_quality};
    //cv::imwrite(full, draw, params);

    last_snap_ts_[seat_id] = ts_ms;
    last_state_hash_[seat_id] = state_hash;
    return full;
}

} // namespace vision