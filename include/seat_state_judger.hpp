#ifndef SEAT_STATE_JUDGER_HPP
#define SEAT_STATE_JUDGER_HPP
#include "SeatDatabase.h"  // YZC：这是数据库头文件，记得修改路径，不然会报错！！！
#include "data_structures.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <json.hpp>
using json = nlohmann::json;
using namespace cv;
using namespace std;

// 配置参数
const float IOU_THRESHOLD = 0.3f;
const float FG_RATIO_THRESHOLD = 0.05f;
const int ANOMALY_THRESHOLD_SECONDS = 120;
const int MORPH_KERNEL_SIZE = 5;

class SeatStateJudger {
public:
    SeatStateJudger();
    ~SeatStateJudger() = default;

    // 原有接口（不变）
    void processAData(
        const A2B_Data& a_data,
        const json& seat_j,
        B2CD_State& state,
        vector<B2CD_Alert>& alerts,
        B2C_SeatSnapshot& out_snapshot,
        optional<B2C_SeatEvent>& out_event
    );
    bool readLastFrameData(
        vector<A2B_Data>& out_a2b_data_list,
        vector<json>& out_seat_j_list
    );
    void run(const std::string& jsonl_path = "");
    string stateToStr(B2CD_State::SeatStatus status);
    string msToISO8601(int64_t ts_ms);
    bool readJsonlFile(
        const string& jsonl_path,
        vector<vector<A2B_Data>>& out_batch_a2b_data,
        vector<vector<json>>& out_batch_seat_j
    );

    // 向A同学返回需要入库的frame_index列表
    vector<int> getNeedStoreFrameIndexes() const {
        return vector<int>(need_store_frame_indexes_.begin(), need_store_frame_indexes_.end());
    }

    // 重置存储的帧索引（多次运行时避免重复）
    void resetNeedStoreFrameIndexes() {
        need_store_frame_indexes_.clear();
    }

private:
    // 原有存储变量（不变）
    unordered_map<string, int64_t> last_seat_ts_;
    unordered_map<string, B2CD_State::SeatStatus> last_seat_status_;
    unordered_map<string, int> last_seat_status_duration_;
    unordered_map<string, int> anomaly_occupied_duration_;

    // 存储需要入库的frame_index（视频帧唯一ID）
    unordered_set<int> need_store_frame_indexes_;

    // 新增：数据库引用
    SeatDatabase& db_;
    
    class SeatTimer {
    public:
        bool is_running = false;
        chrono::steady_clock::time_point start_time;
        int getElapsedSeconds();
    };
    Mat preprocessFgMask(const Mat& frame, const Rect& roi);
    float calculateIoU(const Rect& rect1, const Rect& rect2);
    string getISO8601Timestamp();
    Ptr<BackgroundSubtractorMOG2> mog2_;
    unordered_map<string, SeatTimer> seat_timers_;
    unordered_map<string, string> last_seat_states_;
};

#endif // SEAT_STATE_JUDGER_HPP
