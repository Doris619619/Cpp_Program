#pragma once
#include <string>
#include <unordered_map>
#include <opencv2/core.hpp>

namespace vision {

struct SnapshotPolicy {
    int min_interval_ms = 3000;
    bool on_change_only = true;
    int heartbeat_ms = 5000;
    int jpg_quality = 90;
};

class Snapshotter {
public:
    Snapshotter(const std::string& dir, const SnapshotPolicy& policy);
    std::string saveSnapshot(const std::string& seat_id,
                          int state_hash,
                          int64_t ts_ms,
                          const cv::Mat& bgr,
                          const std::vector<cv::Rect>& boxes);

private:
    std::string dir_;
    SnapshotPolicy policy_;
    std::unordered_map<std::string, int64_t> last_snap_ts_;
    std::unordered_map<std::string, int> last_state_hash_;
};

} // namespace vision