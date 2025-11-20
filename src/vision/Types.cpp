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
        o["person_conf"] = s.person_conf;
        o["object_conf"] = s.object_conf;
        o["fg_ratio"] = s.fg_ratio;
        o["person_count"] = s.person_count;
        o["object_count"] = s.object_count;
        o["occupancy_state"] = toString(s.occupancy_state);
        o["snapshot_path"] = s.snapshot_path;
        j.push_back(o);
    }
    return j.dump();
}

} // namespace vision