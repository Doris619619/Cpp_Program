#include "vision/Types.h"
#include <nlohmann/json.hpp>

namespace vision {

std::string seatFrameStatesToJson(const std::vector<SeatFrameState>& states) {
    nlohmann::json j = nlohmann::json::array();
    for (auto &s : states) {
        nlohmann::json o;
        o["seat_id"] = s.seat_id;
        o["ts_ms"] = s.ts_ms;
        o["frame_index"] = s.frame_index;
        o["has_person"] = s.has_person;
        o["has_object"] = s.has_object;
        o["person_conf"] = s.person_conf_max;
        o["object_conf"] = s.object_conf_max;
        o["fg_ratio"] = s.fg_ratio;
        o["person_count"] = s.person_count;
        o["object_count"] = s.object_count;
        o["occupancy_state"] = toString(s.occupancy_state);
        o["snapshot_path"] = s.snapshot_path;
        j.push_back(o);
    }
    return j.dump();
}

// used for printing into the output json file for B and C
std::string seatFrameStatesToJsonLine(
    const std::vector<SeatFrameState>& states,
    int64_t ts_ms,
    int64_t frame_index,
    const std::string& image_path,
    const std::string& annotated_path) {
    nlohmann::json root;
    root["ts_ms"] = ts_ms;
    root["frame_index"] = frame_index;
    root["image_path"] = image_path;
    root["annotated_path"] = annotated_path;

    nlohmann::json arr = nlohmann::json::array();
    for (const auto& s : states) {
        nlohmann::json o;
        o["seat_id"] = s.seat_id;
        o["ts_ms"] = s.ts_ms;
        o["frame_index"] = s.frame_index;
        o["has_person"] = s.has_person;
        o["has_object"] = s.has_object;
        o["person_conf"] = s.person_conf_max;
        o["object_conf"] = s.object_conf_max;
        o["fg_ratio"] = s.fg_ratio;
        o["person_count"] = s.person_count;
        o["object_count"] = s.object_count;
        o["occupancy_state"] = toString(s.occupancy_state);
        o["snapshot_path"] = s.snapshot_path;

        // ROI
        o["seat_roi"] = {
            {"x", s.seat_roi.x}, {"y", s.seat_roi.y},
            {"w", s.seat_roi.width}, {"h", s.seat_roi.height}
        };
        // 多边形（若存在）
        if (!s.seat_poly.empty()) {
            nlohmann::json poly = nlohmann::json::array();
            for (auto &p : s.seat_poly) poly.push_back({p.x, p.y});
            o["seat_poly"] = std::move(poly);
        }

        // Boxes
        nlohmann::json pboxes = nlohmann::json::array();
        for (const auto& b : s.person_boxes_in_roi) {
            pboxes.push_back({
                {"x", b.rect.x}, {"y", b.rect.y},
                {"w", b.rect.width}, {"h", b.rect.height},
                {"conf", b.conf}, {"cls_id", b.cls_id}, {"cls_name", b.cls_name}
            });
        }
        nlohmann::json oboxes = nlohmann::json::array();
        for (const auto& b : s.object_boxes_in_roi) {
            oboxes.push_back({
                {"x", b.rect.x}, {"y", b.rect.y},
                {"w", b.rect.width}, {"h", b.rect.height},
                {"conf", b.conf}, {"cls_id", b.cls_id}, {"cls_name", b.cls_name}
            });
        }
        o["person_boxes"] = std::move(pboxes);
        o["object_boxes"] = std::move(oboxes);

        // perf
        o["t_pre_ms"] = s.t_pre_ms;
        o["t_inf_ms"] = s.t_inf_ms;
        o["t_post_ms"] = s.t_post_ms;

        arr.push_back(std::move(o));
    }
    root["seats"] = std::move(arr);
    return root.dump();
}

// temp implementation, not used yet
bool parseSeatFrameStatesFromJson(const std::string& json, std::vector<SeatFrameState>& out) {
    return false;
};

} // namespace vision