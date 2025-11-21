#include "vision/Config.h"
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>
#include <fstream>

using nlohmann::json;

namespace vision {

static void try_get(const YAML::Node& n, const char* key, std::string& v) { if (n[key]) v = n[key].as<std::string>(); }
static void try_get(const YAML::Node& n, const char* key, int& v)         { if (n[key]) v = n[key].as<int>(); }
static void try_get(const YAML::Node& n, const char* key, float& v)       { if (n[key]) v = n[key].as<float>(); }
static void try_get(const YAML::Node& n, const char* key, bool& v)        { if (n[key]) v = n[key].as<bool>(); }

VisionConfig VisionConfig::fromYaml(const std::string& yaml_path) {
    VisionConfig c;
    try {
        YAML::Node r = YAML::LoadFile(yaml_path);
        try_get(r, "seats_json",   c.seats_json);
        try_get(r, "model_path",   c.model_path);
        try_get(r, "vision_yaml",  c.vision_yaml);
        try_get(r, "log_dir",      c.log_dir);
        try_get(r, "snapshot_dir", c.snapshot_dir);
        try_get(r, "states_output", c.states_output);
        try_get(r, "annotated_frames_dir", c.annotated_frames_dir);

        try_get(r, "input_w", c.input_w);
        try_get(r, "input_h", c.input_h);

        try_get(r, "conf_thres_person",     c.conf_thres_person);
        try_get(r, "conf_thres_person_low", c.conf_thres_person_low);
        try_get(r, "conf_thres_object",     c.conf_thres_object);
        try_get(r, "nms_iou",               c.nms_iou);
        try_get(r, "iou_seat_intersect",    c.iou_seat_intersect);
        try_get(r, "mog2_fg_ratio_thres",   c.mog2_fg_ratio_thres);

        try_get(r, "snapshot_jpg_quality",     c.snapshot_jpg_quality);
        try_get(r, "snapshot_min_interval_ms", c.snapshot_min_interval_ms);
        try_get(r, "snapshot_on_change_only",  c.snapshot_on_change_only);
        try_get(r, "snapshot_heartbeat_ms",    c.snapshot_heartbeat_ms);

        if (r["object_allow"]) {
            c.object_allow.clear();
            for (const auto& it : r["object_allow"]) c.object_allow.push_back(it.as<std::string>());
        }

        try_get(r, "intra_threads", c.intra_threads);

        try_get(r, "mog2_history",         c.mog2_history);
        try_get(r, "mog2_var_threshold",   c.mog2_var_threshold);
        try_get(r, "mog2_detect_shadows",  c.mog2_detect_shadows);
        try_get(r, "fg_morph_enable",      c.fg_morph_enable);
        try_get(r, "fg_morph_erode_iterations", c.fg_morph_erode_iterations);

        try_get(r, "dump_perf_log", c.dump_perf_log);
        try_get(r, "enable_async_snapshot", c.enable_async_snapshot);
        try_get(r, "yolo_variant", c.yolo_variant);
    } catch (...) {
        // keep defaults
    }
    return c;
}

VisionConfig VisionConfig::fromJson(const std::string& json_path) {
    VisionConfig c;
    try {
        std::ifstream ifs(json_path);
        json r; ifs >> r;
        auto get_s = [&](const char* k, std::string& v){ if(r.contains(k)) v = r[k].get<std::string>(); };
        auto get_i = [&](const char* k, int& v){ if(r.contains(k)) v = r[k].get<int>(); };
        auto get_f = [&](const char* k, float& v){ if(r.contains(k)) v = r[k].get<float>(); };
        auto get_b = [&](const char* k, bool& v){ if(r.contains(k)) v = r[k].get<bool>(); };

        get_s("seats_json", c.seats_json);
        get_s("model_path", c.model_path);
        get_s("vision_yaml", c.vision_yaml);
        get_s("log_dir", c.log_dir);
        get_s("snapshot_dir", c.snapshot_dir);
        get_s("states_output", c.states_output);
        get_s("annotated_frames_dir", c.annotated_frames_dir);

        get_i("input_w", c.input_w);
        get_i("input_h", c.input_h);

        get_f("conf_thres_person", c.conf_thres_person);
        get_f("conf_thres_person_low", c.conf_thres_person_low);
        get_f("conf_thres_object", c.conf_thres_object);
        get_f("nms_iou", c.nms_iou);
        get_f("iou_seat_intersect", c.iou_seat_intersect);
        get_f("mog2_fg_ratio_thres", c.mog2_fg_ratio_thres);

        get_i("snapshot_jpg_quality", c.snapshot_jpg_quality);
        get_i("snapshot_min_interval_ms", c.snapshot_min_interval_ms);
        get_b("snapshot_on_change_only", c.snapshot_on_change_only);
        get_i("snapshot_heartbeat_ms", c.snapshot_heartbeat_ms);

        if (r.contains("object_allow")) {
            c.object_allow.clear();
            for (auto& it : r["object_allow"]) c.object_allow.push_back(it.get<std::string>());
        }

        get_i("intra_threads", c.intra_threads);

        get_i("mog2_history", c.mog2_history);
        get_i("mog2_var_threshold", c.mog2_var_threshold);
        get_b("mog2_detect_shadows", c.mog2_detect_shadows);
        get_b("fg_morph_enable", c.fg_morph_enable);
        get_i("fg_morph_erode_iterations", c.fg_morph_erode_iterations);

        get_b("dump_perf_log", c.dump_perf_log);
        get_b("enable_async_snapshot", c.enable_async_snapshot);
        get_s("yolo_variant", c.yolo_variant);
    } catch (...) {
        // keep defaults
    }
    return c;
}

} // namespace vision