#pragma once
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>

namespace vision {

struct SeatROI {
    int seat_id = -1;
    cv::Rect rect;                 // 首选矩形
    std::vector<cv::Point> poly;   // 若使用多边形，优先使用 poly 判定
};

// 从 seats.json 读取 seats 或 tables 并生成 seats
// 支持两种格式：
// 1) 直接 seats: { seat_id, roi:{x,y,w,h} } 或 { seat_id, poly:[[x,y],...] }
// 2) tables + seat_layout: 自动切分生成 2x2 或 NxM 的座位
bool loadSeatsFromJson(const std::string& path, std::vector<SeatROI>& out);

// 将 seats 保存为标准 seats.json （仅写出矩形或多边形其一）
bool saveSeatsToJson(const std::string& path, const std::vector<SeatROI>& seats);

// 将四边形桌面按 layout 切分为若干座位矩形（简单均分）
// layout 形如 "2x2"、"1x4" 等
std::vector<cv::Rect> splitTablePolyToSeats(const std::vector<cv::Point>& poly, const std::string& layout);

} // namespace vision
