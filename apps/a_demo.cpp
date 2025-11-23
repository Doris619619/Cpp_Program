/*  Name:  a_demo.cpp
*   Usage: Configuration: cmake --build --preset msvc-ninja-debug
*          Build:         .\build\a_demo.exe [img_dir=data\samples] [out_states_file=None]
*   ==========================================================================================
*   Minimal demo implemented basic dataflow pipeline invoking VisionA
*/

/*#include <iostream>

int main() {
	std::cout << "a_demo stub running.\n";
	// TODO: integrate Vision pipeline.
	return 0;
}
*/
#include "vision/VisionA.h"
#include "vision/Publish.h"
#include "vision/Config.h"
#include "vision/FrameExtractor.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <fstream>

using namespace vision;

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    std::string img_dir = "data/frames";              // 输入图像目录
    std::string override_out_states;                   // 可通过第二个参数覆盖输出文件
    if (argc > 1) img_dir = argv[1];
    if (argc > 2) override_out_states = argv[2];

    std::cout << "a_demo starting...\n";
    std::cout << "CWD: " << std::filesystem::current_path().string() << "\n";
    std::cout << "img_dir: " << img_dir << "\n";
    std::cout.flush();

    // Load config with internal exception-safety (fromYaml keeps defaults on failure)
    VisionConfig cfg = VisionConfig::fromYaml("config/vision.yml");
    if (!std::filesystem::exists("config/vision.yml")) {
        std::cerr << "config/vision.yml not found relative to CWD.\n";
        return 1;
    }
    if (!std::filesystem::exists(cfg.seats_json)) {
        std::cerr << "seats json not found: " << cfg.seats_json << "\n";
        return 1;
    }
    if (!std::filesystem::exists(img_dir)) {
        std::cerr << "input path not found: " << img_dir << "\n";
        std::cerr << "Hint: use a directory of images or a video file path.\n";
        return 1;
    }
    // 使用配置中的 states_output，若用户cli提供则覆盖
    std::string out_states = override_out_states.empty() ? cfg.states_output : override_out_states;
    VisionA vision(cfg);
    std::cout << "Loaded seats from " << cfg.seats_json 
            << ": count=" << vision.processFrame(cv::Mat(10, 10, CV_8UC3), now_ms(), 0).size() 
            << " (dummy frame)\n";
    Publisher pub;
    pub.setCallback([](const std::vector<SeatFrameState>& states){
        std::cout << "Callback batch size = " << states.size() << "\n";
    });
    vision.setPublisher(&pub);
    // note: 上面在count之后的processFrame会在count输出之前就输出一堆info，
    //        其中包含不少failed的情况，疑似与.dll有关，需要纠正；后续输出都正常进行

    int64_t frame_index = 0;
    // 创建输出目录（若路径不含父目录则跳过）
    auto parent = std::filesystem::path(out_states).parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            std::cerr << "Failed to create directory: " << parent.string() << " : " << ec.message() << "\n";
        }
    }
    std::ofstream ofs(out_states, std::ios::app);
    if (!ofs) {
        std::cerr << "Failed to open output states file: " << out_states << "\n";
        return 1;
    }
    std::cout << "States output file: " << std::filesystem::absolute(out_states).string() << "\n";
    // 额外生成最新帧覆盖文件，便于快速查看（非行追加）
    std::string latest_frame_file = (parent.empty() ? std::string("last_frame.json") : (parent / "last_frame.json").string());
    // 帧标注目录
    std::error_code ec_mk;
    std::filesystem::create_directories(cfg.annotated_frames_dir, ec_mk);

    try {
        auto inputPath = std::filesystem::path(img_dir);
        if (std::filesystem::is_directory(inputPath)) {
            for (auto &entry : std::filesystem::directory_iterator(inputPath)) {
                if (!entry.is_regular_file()) continue;
                std::string src_path = entry.path().string();
                cv::Mat bgr = cv::imread(src_path);
                if (bgr.empty()) continue;
                auto states = vision.processFrame(bgr, now_ms(), frame_index++);
                int64_t ts = states.empty() ? now_ms() : states.front().ts_ms;
                for (auto &s : states) {
                    std::cout << s.seat_id << " " << toString(s.occupancy_state)
                              << " pc=" << s.person_conf_max
                              << " oc=" << s.object_conf_max
                              << " fg=" << s.fg_ratio
                              << " snap=" << (s.snapshot_path.empty() ? "-" : s.snapshot_path)
                              << "\n";
                }
                std::vector<BBox> all_persons, all_objects;
                vision.getLastDetections(all_persons, all_objects);
                cv::Mat vis = bgr.clone();
                for (auto &p : all_persons) {
                    cv::rectangle(vis, p.rect, cv::Scalar(255,0,255), 2);
                    std::string label = "person " + std::to_string(int(p.conf * 100)) + "%";
                    cv::putText(vis, label, cv::Point(p.rect.x, p.rect.y - 5),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,255), 1);
                }
                for (auto &o : all_objects) {
                    cv::rectangle(vis, o.rect, cv::Scalar(255,255,0), 2);
                    std::string label = o.cls_name + " " + std::to_string(int(o.conf * 100)) + "%";
                    cv::putText(vis, label, cv::Point(o.rect.x, o.rect.y - 5),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,0), 1);
                }
                auto color_for_state = [](SeatOccupancyState st){
                    switch(st){
                        case SeatOccupancyState::PERSON:        return cv::Scalar(0,0,255);
                        case SeatOccupancyState::OBJECT_ONLY:   return cv::Scalar(0,255,255);
                        case SeatOccupancyState::FREE:          return cv::Scalar(0,255,0);
                        default:                                return cv::Scalar(200,200,200);
                    }
                };
                for (auto &s : states) {
                    auto color = color_for_state(s.occupancy_state);
                    if (s.seat_poly.size() >= 3) {
                        std::vector<std::vector<cv::Point>> contours = { s.seat_poly };
                        cv::polylines(vis, contours, true, color, 3);
                        cv::Moments m = cv::moments(s.seat_poly);
                        if (m.m00 != 0) {
                            cv::Point center(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                            std::string seat_label = "S" + std::to_string(s.seat_id) + " " + toString(s.occupancy_state);
                            cv::putText(vis, seat_label, center,
                                       cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
                        }
                    } else {
                        cv::rectangle(vis, s.seat_roi, color, 3);
                        std::string seat_label = "Seat " + std::to_string(s.seat_id) + " " + toString(s.occupancy_state);
                        cv::putText(vis, seat_label, cv::Point(s.seat_roi.x, s.seat_roi.y - 10),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
                    }
                }
                std::string fname = entry.path().filename().string();
                std::string annotated_path = (std::filesystem::path(cfg.annotated_frames_dir) / fname).string();
                cv::imwrite(annotated_path, vis);
                std::string line = seatFrameStatesToJsonLine(states, ts, frame_index-1, src_path, annotated_path);
                ofs << line << "\n";
                {
                    std::ofstream lf(latest_frame_file, std::ios::trunc);
                    if (lf) lf << line << "\n";
                }
            }
        } else {
            // 文件：按视频处理并逐帧送入 VisionA
            std::cout << "Input is a video file, iterating frames...\n";
            auto src_path = inputPath.string();
            size_t processed = 0;
            FrameExtractor::iterate(
                src_path,
                [&](int frameIdx, const cv::Mat& bgr, double /*tsec*/) -> bool {
                    auto states = vision.processFrame(bgr, now_ms(), frame_index++);
                    int64_t ts = states.empty() ? now_ms() : states.front().ts_ms;
                    for (auto &s : states) {
                        std::cout << s.seat_id << " " << toString(s.occupancy_state)
                                  << " pc=" << s.person_conf_max
                                  << " oc=" << s.object_conf_max
                                  << " fg=" << s.fg_ratio
                                  << " snap=" << (s.snapshot_path.empty() ? "-" : s.snapshot_path)
                                  << "\n";
                    }
                    std::vector<BBox> all_persons, all_objects;
                    vision.getLastDetections(all_persons, all_objects);
                    cv::Mat vis = bgr.clone();
                    for (auto &p : all_persons) {
                        cv::rectangle(vis, p.rect, cv::Scalar(255,0,255), 2);
                        std::string label = "person " + std::to_string(int(p.conf * 100)) + "%";
                        cv::putText(vis, label, cv::Point(p.rect.x, p.rect.y - 5),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,255), 1);
                    }
                    for (auto &o : all_objects) {
                        cv::rectangle(vis, o.rect, cv::Scalar(255,255,0), 2);
                        std::string label = o.cls_name + " " + std::to_string(int(o.conf * 100)) + "%";
                        cv::putText(vis, label, cv::Point(o.rect.x, o.rect.y - 5),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,0), 1);
                    }
                    auto stem = inputPath.stem().string();
                    char buf[64];
                    std::snprintf(buf, sizeof(buf), "%s_%06d.jpg", stem.c_str(), frameIdx);
                    std::string annotated_path = (std::filesystem::path(cfg.annotated_frames_dir) / buf).string();
                    cv::imwrite(annotated_path, vis);
                    std::string line = seatFrameStatesToJsonLine(states, ts, frame_index-1, src_path, annotated_path);
                    ofs << line << "\n";
                    {
                        std::ofstream lf(latest_frame_file, std::ios::trunc);
                        if (lf) lf << line << "\n";
                    }
                    ++processed;
                    return true; // 继续
                }
            );
            std::cout << "Processed video frames: " << processed << "\n";
        }
    } catch (const std::filesystem::filesystem_error& fe) {
        std::cerr << "filesystem error: " << fe.what() << "\n";
        return 1;
    }
    std::cout << "Seat states appended to: " << out_states << "\n";
    std::cout << "Latest frame snapshot: " << latest_frame_file << "\n";
    return 0;
}