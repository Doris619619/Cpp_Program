#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <cstdint>
#include "Enums.h"

namespace vision {

// 基础检测框
struct BBox {
    cv::Rect    rect;           // 左上 + 宽高
    float       conf = 0.f;     // 总置信度 (0~1)
    int         cls_id = -1;    // 类别 id (default = -1)
    std::string cls_name;       // 类别名称 ("person", "object", "backpack"...)
};

// 处理方法判据参考输入类型
enum class InputType {
    DIRECTORY_IMAGE,
    VIDEO_FILE,
    IMAGE_FILE,
    NOT_EXISTS,
    UNKNOWN
};

// 座位帧状态
struct SeatFrameState {
    int seat_id = -1;              // 与seat.json中的seat_id一致
    int64_t ts_ms = 0;
    int64_t frame_index = -1;

    bool has_person = false;       // 该座位当前帧是否有人
    bool has_object = false;       // 该座位当前帧是否有物品

    float person_conf_max = 0.f;   // 该座位内最高 person 置信度
    float object_conf_max = 0.f;   // 该座位内最高允许物品类别置信度
    float fg_ratio    = 0.f;       // MOG2 前景占比 0~1

    int person_count = 0;          // 该座位内检测到的人数   
    int object_count = 0;          // 该座位内检测到的物品数 

    SeatOccupancyState occupancy_state = SeatOccupancyState::UNKNOWN;

    cv::Rect seat_roi;             // 座位 ROI（矩形或多边形的外接矩形）
    std::vector<cv::Point> seat_poly;  // 座位多边形（若有），优先使用多边形判定
    std::vector<BBox> person_boxes_in_roi;
    std::vector<BBox> object_boxes_in_roi;

    std::string snapshot_path;     // 若本帧触发快照则非空

    // 过程耗时 (ms) (performance metrics)
    int t_pre_ms  = 0;
    int t_inf_ms  = 0;
    int t_post_ms = 0;
};

// 返回单帧所有座位状态的 .json 字符串
std::string seatFrameStatesToJson(const std::vector<SeatFrameState>& states);

/* 返回包含帧级封装的一行 .jsonl
{
   frame_index, ts_ms, image_path, annotated_path,
   seats: [ { seat_id, seat_roi{x,y,w,h}, ... , person_boxes:[{x,y,w,h,conf,cls_id,cls_name}], object_boxes:[...] } ]
}
*/
std::string seatFrameStatesToJsonLine(
    const std::vector<SeatFrameState>& states,
    int64_t ts_ms,
    int64_t frame_index,
    const std::string& image_path,
    const std::string& annotated_path);

// 解析（如需要反序列化）
bool parseSeatFrameStatesFromJson(const std::string& json, std::vector<SeatFrameState>& out);

} // namespace vision

