// Minimal demo executable entry point.
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
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <chrono>

using namespace vision;

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    std::string img_dir = "data/samples";
    if (argc > 1) img_dir = argv[1];

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
        std::cerr << "images directory not found: " << img_dir << "\n";
        std::cerr << "Hint: use data/samples (exists in repo).\n";
        return 1;
    }
    VisionA vision(cfg);
    std::cout << "Loaded seats from " << cfg.seats_json 
            << ": count=" << vision.processFrame(cv::Mat(10,10,CV_8UC3), now_ms(), 0).size() 
            << " (dummy frame)\n";
    Publisher pub;
    pub.setCallback([](const std::vector<SeatFrameState>& states){
        std::cout << "Callback batch size = " << states.size() << "\n";
    });
    vision.setPublisher(&pub);

    int64_t frame_index = 0;
    try {
        for (auto &entry : std::filesystem::directory_iterator(img_dir)) {
            if (!entry.is_regular_file()) continue;
            cv::Mat bgr = cv::imread(entry.path().string());
            if (bgr.empty()) continue;
            auto states = vision.processFrame(bgr, now_ms(), frame_index++);
            for (auto &s : states) {
                std::cout << s.seat_id << " "
                          << toString(s.occupancy_state)
                          << " pc=" << s.person_conf
                          << " oc=" << s.object_conf
                          << " fg=" << s.fg_ratio
                          << " snap=" << (s.snapshot_path.empty() ? "-" : s.snapshot_path)
                          << "\n";
            }
        }
    } catch (const std::filesystem::filesystem_error& fe) {
        std::cerr << "filesystem error: " << fe.what() << "\n";
        return 1;
    }
    return 0;
}