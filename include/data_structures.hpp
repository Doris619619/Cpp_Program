#ifndef DATA_STRUCTURES_HPP
#define DATA_STRUCTURES_HPP
#pragma once
#include <opencv2/core.hpp>    // 包含 cv::Mat、cv::Rect 等核心类型
#include <opencv2/imgproc.hpp> // 图像处理相关类型
#include <string>
#include <vector>
#include <optional>            // 用于 std::optional（B2C_SeatEvent）

// 1. 对齐A的person_boxes/object_boxes
struct DetectedObject {
    std::string class_name;  // A的cls_name，"person" 或 "object"
    cv::Rect bbox;           // 对应A的x/y/w/h，检测框坐标（x,y,w,h） 
    float score;             // A的conf
    int class_id;            // A的cls_id
};

// 2. 对齐A同学的A2B_Data结构体
struct A2B_Data {
    int frame_id;                          // A的frame_index，帧ID
    std::string seat_id;                   // A的seat_id，座位ID（如"Lib1-F2-015"）
    cv::Rect seat_roi;                     // 座位ROI坐标
    std::vector<cv::Point2i> seat_poly;    // 座位多边形顶点（对应A的seat_poly）
    std::vector<DetectedObject> person_boxes; // 人检测框列表
    std::vector<DetectedObject> object_boxes; // 物体检测框列表
    std::string timestamp;                 // 由A的ts_ms转换，ISO8601格式时间戳（YYYY-MM-DD HH:MM:SS.ms）
    cv::Mat frame;                         // 由A的image_path读取，原始图像帧
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
    std::string seat_id;          
    std::string state;            // "Seated"/"Unseated"/"Anomaly"
    int person_count;             // 检测到的人数
    std::string timestamp;        // ISO8601格式
};

#endif // DATA_STRUCTURES_HPP