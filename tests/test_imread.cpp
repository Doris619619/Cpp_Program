// [Test] 采样读取目录下图像并调用 VisionA 识别，输出到 test_seat_states.jsonl
// 运行示例（PowerShell）：
//   ./build/test_imread.exe --dir ./data/frames/frames_v004 --fp100 20 --out ./runtime/test_seat_states.jsonl --max 500
// 所有输出以 [Test] 开头，便于识别。

#ifndef TEST_STANDALONE
#include "vision/VisionA.h"
#include "vision/Config.h"
#include "vision/Types.h"
#endif
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <fstream>

namespace fs = std::filesystem;
#ifndef TEST_STANDALONE
using namespace vision;
#else
// 最小化本地定义以避免链接 vision/onnxruntime
enum class SeatOccupancyState { FREE=0, PERSON=1, OBJECT_ONLY=2 };
struct SeatFrameState {
    int seat_id = 0;
    int64_t ts_ms = 0;
    int64_t frame_index = 0;
    SeatOccupancyState occupancy_state = SeatOccupancyState::FREE;
    cv::Rect seat_roi;
};
static std::string make_jsonl_line(const std::vector<SeatFrameState>& states,
                                   int64_t ts, int64_t frame_idx,
                                   const std::string& src_path) {
    // 简化 JSONL 行：仅含必要字段，结构与原版近似
    std::ostringstream oss;
    oss << "{\"ts_ms\":" << ts << ",\"frame_index\":" << frame_idx
        << ",\"src\":\"" << src_path << "\",\"states\":[";
    for (size_t i=0;i<states.size();++i) {
        const auto& s = states[i];
        oss << "{\"seat_id\":" << s.seat_id
            << ",\"occupancy\":" << (int)s.occupancy_state
            << ",\"roi\":[" << s.seat_roi.x << "," << s.seat_roi.y << "," << s.seat_roi.width << "," << s.seat_roi.height << "]}";
        if (i+1<states.size()) oss << ",";
    }
    oss << "]}";
    return oss.str();
}
#endif

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static int stepsize_from_fp100(size_t total, int fp100) {
    if (fp100 <= 0) return (total <= 500 ? 5 : (total <= 1000 ? 10 : 50));
    if (fp100 > 100) fp100 = 100;
    int step = static_cast<int>(std::floor(100.0 / fp100)) + 1;
    return std::max(step, 1);
}

static void test_log(const std::string& msg) { std::cout << "[Test] " << msg << std::endl; }

int main(int argc, char** argv) {
    // 禁用并行线程，避免 oneTBB/并行后端初始化导致的 DLL 依赖问题
    cv::setNumThreads(1);
    cv::setUseOptimized(false);
    // ------------------ 参数解析 ------------------
    std::string image_dir = "./data/frames/frames_v004";
    std::string out_jsonl = "./runtime/test_seat_states.jsonl";
    int fp100 = 20;
    size_t max_process = 500;
    bool fake_mode = true; // 默认 FAKE 模式，绕过 VisionA/ONNX，确保链路可跑通；可用 --no-fake 关闭

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--dir" || arg == "-d") && i+1 < argc) image_dir = argv[++i];
        else if ((arg == "--fp100" || arg == "-f") && i+1 < argc) { try { fp100 = std::stoi(argv[++i]); } catch(...) { fp100 = 20; } }
        else if ((arg == "--out" || arg == "-o") && i+1 < argc) out_jsonl = argv[++i];
        else if ((arg == "--max" || arg == "-m") && i+1 < argc) { try { max_process = static_cast<size_t>(std::stoll(argv[++i])); } catch(...) { max_process = 500; } }
        else if (arg == "--fake") { fake_mode = true; }
        else if (arg == "--no-fake") { fake_mode = false; }
        else if (arg == "--help" || arg == "-h") { test_log("Usage: test_imread --dir <image_dir> --fp100 <N> --out <jsonl> --max <M>"); return 0; }
    }

    test_log(std::string("CWD: ") + fs::current_path().string());
    test_log(std::string("Dir: ") + image_dir);
    test_log(std::string("Out: ") + out_jsonl);
    test_log(std::string("fp100: ") + std::to_string(fp100));
    test_log(std::string("max: ") + std::to_string(max_process));
    test_log(std::string("Absolute dir: ") + fs::absolute(image_dir).string());
    test_log(std::string("Mode: ") + (fake_mode ? "FAKE" : "REAL"));

    // ------------------ 路径与输出准备 ------------------
    if (!fs::exists(image_dir)) { std::cerr << "[Test] Image directory NOT EXISTS: " << image_dir << std::endl; return 1; }
    if (!fs::is_directory(image_dir)) { std::cerr << "[Test] Image path is NOT a directory: " << image_dir << std::endl; return 1; }
    fs::path out_path(out_jsonl); fs::path parent = out_path.parent_path();
    if (!parent.empty()) { std::error_code ec; fs::create_directories(parent, ec); if (ec) { std::cerr << "[Test] create parent failed: " << parent.string() << " : " << ec.message() << std::endl; return 1; } }
    std::ofstream ofs(out_jsonl, std::ios::app); if (!ofs) { std::cerr << "[Test] open out file failed: " << out_jsonl << std::endl; return 1; }
    test_log(std::string("Opened out file: ") + fs::absolute(out_jsonl).string());

    // ------------------ 初始化 VisionA ------------------
#ifndef TEST_STANDALONE
    VisionA* vision_ptr = nullptr;
    VisionConfig cfg;
    if (!fake_mode) {
        test_log("Loading config: config/vision.yml");
        cfg = VisionConfig::fromYaml("config/vision.yml");
        test_log(std::string("Config seats_json: ") + cfg.seats_json);
        if (!fs::exists("config/vision.yml")) { std::cerr << "[Test] config/vision.yml not found." << std::endl; return 1; }
        test_log(std::string("Seats json: ") + cfg.seats_json);
        if (!fs::exists(cfg.seats_json)) { std::cerr << "[Test] seats json not found: " << cfg.seats_json << std::endl; return 1; }
        test_log("Constructing VisionA...");
        vision_ptr = new VisionA(cfg);
        test_log(std::string("Loaded seats: ") + std::to_string(vision_ptr->seatCount()));
    } else {
        test_log("FAKE mode: skip VisionA construction.");
    }
#endif

    // ------------------ 遍历与采样 ------------------
    size_t total_images = 0; 
    try {
        for (auto& e : fs::directory_iterator(image_dir)) { 
            if (!e.is_regular_file()) continue; 
            auto ext = e.path().extension().string(); 
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower); 
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") ++total_images; 
        }
    } catch (const std::exception& ex) {
        std::cerr << "[Test] directory iteration exception: " << ex.what() << std::endl; return 1;
    } catch (...) {
        std::cerr << "[Test] directory iteration unknown exception" << std::endl; return 1;
    }
    int step = stepsize_from_fp100(total_images, fp100); test_log(std::string("Images: ") + std::to_string(total_images) + ", step=" + std::to_string(step));

    size_t processed = 0, errors = 0; int frame_index = 0; cv::Mat last_vis;
    int effective_index = 0; // 仅对有效图计数
    for (auto& entry : fs::directory_iterator(image_dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string(); std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (!(ext == ".jpg" || ext == ".jpeg" || ext == ".png")) continue;
        test_log(std::string("Reading: ") + entry.path().string());
        cv::Mat bgr = cv::imread(entry.path().string());
        test_log(std::string("Read done: ") + std::to_string(!bgr.empty()) + 
                 ", size=" + std::to_string(bgr.cols) + "x" + std::to_string(bgr.rows));
        if (!bgr.empty() && bgr.cols == 0) { std::cerr << "[Test] unexpected zero cols" << std::endl; }
        if (bgr.empty()) { ++errors; std::cerr << "[Test] imread empty: " << entry.path().string() << std::endl; continue; }
        if (step > 1 && (effective_index % step != 0)) { ++effective_index; continue; }
        ++effective_index;

        try {
            std::vector<SeatFrameState> states;
#ifndef TEST_STANDALONE
            if (!fake_mode) {
                states = vision_ptr->processFrame(bgr, now_ms(), frame_index);
                if (states.empty()) { test_log("Frame produced 0 states (empty detection)." ); }
            } else {
                SeatFrameState s; s.seat_id = 0; s.ts_ms = now_ms(); s.frame_index = frame_index; s.occupancy_state = SeatOccupancyState::FREE; states.push_back(s);
            }
#else
            // 独立模式仅生成一条 FREE 状态，便于链路验证
            {
                SeatFrameState s; s.seat_id = 0; s.ts_ms = now_ms(); s.frame_index = frame_index; s.occupancy_state = SeatOccupancyState::FREE;
                s.seat_roi = cv::Rect(bgr.cols/4, bgr.rows/4, bgr.cols/2, bgr.rows/2);
                states.push_back(s);
            }
#endif
            int64_t ts = states.empty() ? now_ms() : states.front().ts_ms;
            last_vis = bgr.clone();
            for (auto &s : states) {
                cv::Scalar color(0, 255, 0);
                if (s.occupancy_state == SeatOccupancyState::PERSON) color = cv::Scalar(0,0,255);
                else if (s.occupancy_state == SeatOccupancyState::OBJECT_ONLY) color = cv::Scalar(0,255,255);
                cv::rectangle(last_vis, s.seat_roi, color, 2);
            }
            std::string line;
#ifndef TEST_STANDALONE
            line = seatFrameStatesToJsonLine(states, ts, frame_index, entry.path().string(), "");
#else
            line = make_jsonl_line(states, ts, frame_index, entry.path().string());
#endif
            ofs << line << "\n";
            ++processed; ++frame_index;
            if (processed % 5 == 0) test_log(std::string("processed=") + std::to_string(processed) + ", errors=" + std::to_string(errors));
            if (processed >= max_process) { test_log("Reached max limit."); break; }
        } catch (const std::exception& ex) {
            ++errors; std::cerr << "[Test] exception: " << ex.what() << " src=" << entry.path().string() << std::endl;
        } catch (...) {
            ++errors; std::cerr << "[Test] unknown exception src=" << entry.path().string() << std::endl;
        }
    }

    test_log(std::string("Summary: processed=") + std::to_string(processed) + ", errors=" + std::to_string(errors));

    // ------------------ 显示最后一张 ------------------
    if (!last_vis.empty()) {
        cv::imshow("[Test] Last Frame", last_vis);
        test_log("Press 'q' to close window.");
        while (true) { int key = cv::waitKey(30); if (key == 'q' || key == 'Q') break; }
        cv::destroyAllWindows();
    } else {
        test_log("No frame to show.");
    }

#ifndef TEST_STANDALONE
    if (vision_ptr) { delete vision_ptr; vision_ptr = nullptr; }
#endif
    return 0;
}