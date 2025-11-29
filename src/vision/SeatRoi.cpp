#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include "vision/SeatRoi.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>

using nlohmann::json;

namespace vision {

static cv::Rect rectFromJson(const json& r) {
    return cv::Rect(r.value("x",0), r.value("y",0), r.value("w",0), r.value("h",0));
}

bool loadSeatsFromJson(const std::string& path, std::vector<SeatROI>& out) {
    out.clear();
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;
    json root;
    try { ifs >> root; } catch(const std::exception&) { return false; }

    if (root.contains("seats")) {
        for (auto& s : root["seats"]) {
            try {
                SeatROI seat;
                // seat_id 允许是整数或字符串(如 "A1"), 字符串时提取其中的数字部分
                if (s.contains("seat_id")) {
                    if (s["seat_id"].is_number_integer()) {
                        seat.seat_id = s["seat_id"].get<int>();
                    } else if (s["seat_id"].is_string()) {
                        std::string sid = s["seat_id"].get<std::string>();
                        std::string digits;
                        for (char c : sid) if (std::isdigit(static_cast<unsigned char>(c))) digits.push_back(c);
                        if (!digits.empty()) {
                            try { seat.seat_id = std::stoi(digits); } catch(...) { seat.seat_id = -1; }
                        } else {
                            seat.seat_id = -1; // 未能解析则置为 -1
                        }
                    } else {
                        seat.seat_id = -1;
                    }
                } else {
                    seat.seat_id = -1;
                }
                if (s.contains("roi")) seat.rect = rectFromJson(s["roi"]);
                if (s.contains("poly")) {
                    for (auto& p : s["poly"]) seat.poly.emplace_back(p[0].get<int>(), p[1].get<int>());
                }
                out.push_back(std::move(seat));
            } catch(const std::exception&) {
                // 单个 seat 解析失败，跳过
            }
        }
        return !out.empty();
    }

    // tables + seat_layout 自动切分
    if (root.contains("tables")) {
        for (auto& t : root["tables"]) {
            std::vector<cv::Point> poly;
            if (t.contains("poly")) {
                for (auto& p : t["poly"]) poly.emplace_back(p[0].get<int>(), p[1].get<int>());
            } else if (t.contains("roi")) {
                auto r = rectFromJson(t["roi"]);
                poly = { {r.x,r.y},{r.x+r.width,r.y},{r.x+r.width,r.y+r.height},{r.x,r.y+r.height} };
            }
            auto layout = t.value("seat_layout", std::string("2x2"));
            auto rects = splitTablePolyToSeats(poly, layout);
            int base = t.value("table_id", 0) * 100;
            for (int i=0;i<(int)rects.size();++i) {
                SeatROI s; s.seat_id = base + i + 1; s.rect = rects[i];
                out.push_back(std::move(s));
            }
        }
        return !out.empty();
    }
    return false;
}

bool saveSeatsToJson(const std::string& path, const std::vector<SeatROI>& seats) {
    json root;
    root["seats"] = json::array();
    for (auto& s : seats) {
        json j;
        j["seat_id"] = s.seat_id;
        if (s.poly.size() >= 3) {
            json poly = json::array();
            for (auto& p : s.poly) poly.push_back({p.x, p.y});
            j["poly"] = poly;
        } else {
            j["roi"] = { {"x", s.rect.x},{"y", s.rect.y},{"w", s.rect.width},{"h", s.rect.height} };
        }
        root["seats"].push_back(std::move(j));
    }
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;
    ofs << root.dump(2);
    return true;
}

static cv::Rect boundingRectOfPoly(const std::vector<cv::Point>& poly) {
    return cv::boundingRect(poly);
}

std::vector<cv::Rect> splitTablePolyToSeats(const std::vector<cv::Point>& poly, const std::string& layout) {
    auto R = boundingRectOfPoly(poly);
    int rows=1, cols=1;
    auto pos = layout.find('x');
    if (pos != std::string::npos) {
        try {
            rows = std::max(1, std::stoi(layout.substr(0,pos)));
            cols = std::max(1, std::stoi(layout.substr(pos+1)));
        } catch(...) {}
    }
    std::vector<cv::Rect> out;
    int cw = R.width / cols;
    int ch = R.height / rows;
    for (int r=0;r<rows;++r)
        for (int c=0;c<cols;++c)
            out.emplace_back(R.x + c*cw, R.y + r*ch, cw, ch);
    return out;
}

} // namespace vision

namespace vision {

struct SeatDef {
    std::string seat_id;
    std::string zone;
    cv::Rect roi;
};

static std::vector<SeatDef> loadSeats(const std::string& path) {
    std::vector<SeatDef> seats;
    std::ifstream ifs(path);
    if (!ifs.good()) return seats;
    nlohmann::json j; ifs >> j;
    for (auto &s : j["seats"]) {
        SeatDef d;
        d.seat_id = s["seat_id"].get<std::string>();
        d.zone = s.contains("zone") ? s["zone"].get<std::string>() : "";
        auto r = s["roi"];
        d.roi = cv::Rect(r["x"], r["y"], r["w"], r["h"]);
        seats.push_back(d);
    }
    return seats;
}

} // namespace vision