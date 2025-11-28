#ifndef DATA_STRUCTURES_HPP
#define DATA_STRUCTURES_HPP

#include <opencv2/core.hpp>    // 包含 cv::Mat、cv::Rect 等核心类型
#include <opencv2/imgproc.hpp> // 图像处理相关类型
#include <string>
#include <vector>
#include <optional>            // 用于 std::optional（B2C_SeatEvent）

// 1. A模块→B模块的单物体检测结果
struct DetectedObject {
    std::string class_name;  // 类别："person" 或 "object"
    cv::Rect bbox;           // 边界框（x1,y1,width,height）
    float score;             // 置信度（≥0.5）
    int class_id;            // 类别ID（对应代码中的赋值）
};

// 2. A模块→B模块的完整数据（每帧+每个座位）
struct A2B_Data {
    int frame_id;                          // 帧唯一ID
    std::string seat_id;                   // 座位ID（如"Lib1-F2-015"）
    cv::Rect seat_roi;                     // 座位ROI坐标
    std::vector<DetectedObject> objects;   // 该座位内的检测结果列表
    std::string timestamp;                 // 帧采集时间（YYYY-MM-DD HH:MM:SS.ms）
    cv::Mat frame;                         // 原始帧图像
};

// 3. B模块→C/D模块的座位状态数据
struct B2CD_State {
    std::string seat_id;                   // 座位ID
    enum SeatStatus {                      // 座位状态枚举
        UNSEATED = 0,
        SEATED = 1,
        ANOMALY_OCCUPIED = 2
    } status;
    int status_duration;                   // 当前状态持续时间（秒）
    float confidence;                      // 状态置信度（0-1）
    std::string timestamp;                 // 状态更新时间
    int source_frame_id;                   // 关联的帧ID
};

// 4. B模块→C/D模块的异常警报数据
struct B2CD_Alert {
    std::string alert_id;                  // 警报唯一ID（seat_id_时间戳）
    std::string seat_id;                   // 异常座位ID
    std::string alert_type;                // 固定为"AnomalyOccupied"
    std::string alert_desc;                // 警报描述（如"Bag detected, 持续135秒"）
    std::string timestamp;                 // 警报触发时间
    bool is_processed = false;             // 处理状态（初始未处理）
};

// B模块→C模块：座位状态变化事件（对应seat_events表）
struct B2C_SeatEvent {
    std::string seat_id;          // 对应seat_events.seat_id
    std::string state;            // 对应seat_events.state（值："Seated"/"Unseated"/"Anomaly"）
    std::string timestamp;        // 对应seat_events.timestamp（ISO8601格式，如"2025-11-21T15:30:00.123"）
    int duration_sec;             // 对应seat_events.duration_sec（当前状态持续秒数）
};

// B模块→C模块：座位状态快照（对应seat_snapshots表）
struct B2C_SeatSnapshot {
    std::string seat_id;          // 对应seat_snapshots.seat_id
    std::string state;            // 对应seat_snapshots.state
    int person_count;             // 对应seat_snapshots.person_count（检测到的人数）
    std::string timestamp;        // 对应seat_snapshots.timestamp（ISO8601格式）
};

#endif // DATA_STRUCTURES_HPP