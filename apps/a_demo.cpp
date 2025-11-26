/*  
*   Name:  a_demo.cpp
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
#include "vision/FrameProcessor.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <fstream>
#include <cstddef>
//#include <limits>

using namespace vision;

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}



int main(int argc, char** argv) {
    // CLI args settings (parts)
    std::string img_dir = "data/frames";               // --framesrc 输入图像目录或视频文件
    std::string override_out_states;                   // --out 覆盖输出文件
    size_t max_process_frames = SIZE_MAX;              // --max 处理帧数上限
    double extract_fps = 2.0;                          // --fps 视频抽帧频率
    bool stream_video = false;                         // --stream false: 直接逐帧, 不落盘抽帧目录; true: 抽帧落盘后处理
    
    // anal args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if   (arg == "--framesrc" && i + 1 < argc) {   // --framesrc [img_dir]        // input image directory or video file
            try { img_dir = argv[++i]; } catch (...) { std::cerr << "Error in arg --framesrc\n"; }
        } else if (arg == "--out" && i + 1 < argc) {   // --out [override_out_states] // output states file directory
            try { override_out_states = argv[++i]; } catch (...) { std::cerr << "Error in arg --out\n"; }
        } else if (arg == "--max" && i + 1 < argc) {   // --max [N]                   // set max process frames
            try { max_process_frames = static_cast<size_t>(std::stoll(argv[++i])); } catch (...) { std::cerr << "Error in arg --max\n"; }
        } else if (arg == "--fps" && i + 1 < argc) {   // --fps [F]                   // set extract fps for video input
            try { extract_fps = std::stod(argv[++i]); } catch (...) { std::cerr << "Error in arg --fps\n"; }
        } else if (arg == "--stream" && i + 1 < argc) {// --stream                    // set stream video mode
            try { stream_video = static_cast<bool>(std::stoi(argv[++i])); } catch (...) { std::cerr << "Error in arg --stream\n"; }
        } else if (arg == "-h" || arg == "--help") {   // --help                      // help
            std::cout << "\n"
                      << "Usage: a_demo [input_path] [--out states.jsonl] [--max N] [--fps F] [--stream false/true]\n"
                      << "Or:    a_demo -h \n"
                      << "       a_demo --help \n"
                      << "for help.\n"
                      << "For file type, if pure images are to be processed, fill the \"--out\" term with the directory to the images; otherwise, fill in the video(.mp4) relative to current CWD, with .mp4 postfix ending. \n"
                      << "For the image processing mode, fill the \"--fps\" term with number of images you want to process per 100 images in the directory. (20 fp100 is recommended, which is also the default setting.)\n"
                      << "For the video processing mode, fill the \"--fps\" term with the desired extraction framerate (e.g. 2.0 for 2 frames per second)(the lower the fps is, the less frame it will extract and process from one second of video, and thus less stressful for the program to work.)\n"
                      << "For video input mode, to process video without intermediate frame extraction to disk, set \"--stream true\". Otherwise, \"--stream false\" will start the bulk extraction process.\n"
                      << "For maximum process frames, set \"--max N\" with N as the upper limit of frames to be processed. \n\n";
            return 0;
        } else if (arg.rfind("--", 0) == 0) {          // --                          // unknown option, Error occurs
            std::cerr << "Unknown option: " << arg << "\n"
                      << "Usage: a_demo [input_path or video.mp4] [--out path/to/states.jsonl] [--max N] [--fps F] [--stream false/true]\n"
                      << "Or:    a_demo -h \n"
                      << "       a_demo --help \n"
                      << "for help.\n";
        //} else if (img_dir == "data/frames") {         // 第一个非选项参数视为输入路径
        //    img_dir = arg;
        } else if (override_out_states.empty()) {
            override_out_states = arg;                 // 兼容旧的第二位置参数
            if (!std::filesystem::is_directory(override_out_states)) { // in case if not a directory provided
                override_out_states = "runtime/seat_states.jsonl";
            }
        }
    }

    std::cout << "a_demo starting...\n";
    std::cout << "CWD: " << std::filesystem::current_path().string() << "\n";         // current working directory
    std::cout << "input: " << img_dir << "\n";
    std::cout << "options: max=" << (max_process_frames==SIZE_MAX? -1 : (long long)max_process_frames) << "\n"
              << "         fps=" << extract_fps << "\n"
              << "         stream=" << (stream_video ? true : false) << "\n";
    std::cout.flush();

    // Load config with internal exception-safety (fromYaml keeps defaults on failure)
    VisionConfig cfg = VisionConfig::fromYaml("config/vision.yml");
    if (!std::filesystem::exists("config/vision.yml")) {  // yaml existence
        std::cerr << "config/vision.yml not found relative to CWD.\n";
        return 1;
    }
    if (!std::filesystem::exists(cfg.seats_json)) {       // seat json existence
        std::cerr << "seats json not found: " << cfg.seats_json << "\n";
        return 1;
    }
    if (!std::filesystem::exists(img_dir)) {              // input path existence
        std::cerr << "input path not found: " << img_dir << "\n";
        std::cerr << "Hint: use a directory of images or a video file path.\n";
        return 1;
    }
    // 使用配置中的 states_output，若用户cli提供则覆盖
    std::string out_states = override_out_states.empty() ? cfg.states_output : override_out_states;
    
    VisionA vision(cfg);
    std::cout << "Loaded seats from " << cfg.seats_json 
              << ": count=" << vision.seatCount() 
              << "\n";
    
    // Publish part
    Publisher pub;
    pub.setCallback([](const std::vector<SeatFrameState>& states){
        std::cout << "Callback batch size = " << states.size() << "\n";
    });
    vision.setPublisher(&pub);
    // note: 上面在count之后的processFrame会在count输出之前就输出一堆info，
    //        其中包含不少failed的情况，疑似与.dll有关，需要纠正；后续输出都正常进行

    // create output path (skip if no parent path included)
    int64_t frame_index = 0;                                             // index of img handled this patch
    auto parent_path = std::filesystem::path(out_states).parent_path();  // parent path of the output file
    if (!parent_path.empty()) {     // error check: find file's parent path
        std::error_code error_code_;
        std::filesystem::create_directories(parent_path, error_code_);
        if (error_code_) {
            std::cerr << "Failed to create directory: " << parent_path.string() << " : " << error_code_.message() << "\n";
        }
    }
    
    // open output states file (append mode)
    std::ofstream ofs(out_states, std::ios::app);
    if (!ofs) {                     // error check: open output states file
        std::cerr << "Failed to open output states file: " << out_states << "\n";
        return 1;
    }
    std::cout << "States output file: " << std::filesystem::absolute(out_states).string() << "\n";
    
    // latest frame file directory
    std::string latest_frame_file = (parent_path.empty() ? std::string("last_frame.json") : (parent_path / "last_frame.json").string());
    
    // frame annotated directory
    std::error_code error_code_marker;
    std::filesystem::create_directories(cfg.annotated_frames_dir, error_code_marker);

    // Formal process
    size_t total_processed = 0;
    size_t total_errors = 0;
    int print_every = 100; // 进度打印间隔
    int annotated_save_cnt = 0;
    int annotated_save_freq = cfg.annotated_save_freq;
    try {    
        auto input_path = std::filesystem::path(img_dir); // here the input_path can be video file or directory of img
        
        // Images Processing (Directly, NO Extraction)
        if (std::filesystem::is_directory(input_path)) {  // process frame from img directory
            std::cout << "Image directory mode. Iterating files...\n";
            
            // use directory iterator to iterate the files / folders
            for (auto &entry : std::filesystem::directory_iterator(input_path)) {    
                // find nonEmpty bgr img
                if (!entry.is_regular_file()) continue;
                std::string src_path = entry.path().string();
                cv::Mat bgr = cv::imread(src_path);
                if (bgr.empty()) continue;
                
                try {    // process frames directly
                    // process frames
                    auto states = vision.processFrame(bgr, now_ms(), frame_index++);
                    
                    // output in CLI
                    int64_t ts = states.empty() ? now_ms() : states.front().ts_ms;
                    for (auto &s : states) {
                        std::cout << s.seat_id << " " << toString(s.occupancy_state)
                                  << " pc = " << s.person_conf_max
                                  << " oc = " << s.object_conf_max
                                  << " fg = " << s.fg_ratio
                                  << " snap = " << (s.snapshot_path.empty() ? "-" : s.snapshot_path)
                                  << "\n";
                    }

                    // visualize person/object detecting results in every frame
                    std::vector<BBox> all_persons, all_objects;
                    vision.getLastDetections(all_persons, all_objects);
                    cv::Mat vis = bgr.clone();
                    for (auto &p : all_persons) {    // viusalize person boxes
                        cv::rectangle(vis, p.rect, cv::Scalar(255,0,255), 2);
                        std::string label = "person " + std::to_string(int(p.conf * 100)) + "%";
                        cv::putText(vis, label, cv::Point(p.rect.x, p.rect.y - 5),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,255), 1);
                    }
                    for (auto &o : all_objects) {    // visualize object boxes
                        cv::rectangle(vis, o.rect, cv::Scalar(255,255,0), 2);
                        std::string label = o.cls_name + " " + std::to_string(int(o.conf * 100)) + "%";
                        cv::putText(vis, label, cv::Point(o.rect.x, o.rect.y - 5),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,0), 1);
                    }
                    
                    // visualize seat states
                    auto color_for_state = [](SeatOccupancyState state){
                        switch(state){
                            case SeatOccupancyState::PERSON:        return cv::Scalar(0,0,255);
                            case SeatOccupancyState::OBJECT_ONLY:   return cv::Scalar(0,255,255);
                            case SeatOccupancyState::FREE:          return cv::Scalar(0,255,0);
                            default:                                return cv::Scalar(200,200,200);
                        }
                    };
                    for (auto &state : states) {
                        auto color = color_for_state(state.occupancy_state);
                        if (state.seat_poly.size() >= 3) {  // draw polygon contour
                            std::vector<std::vector<cv::Point>> contours = { state.seat_poly };  // contour of polygon
                            cv::polylines(vis, contours, true, color, 3);
                            cv::Moments moment = cv::moments(state.seat_poly);
                            if (moment.m00 != 0) {
                                cv::Point center(static_cast<int>(moment.m10 / moment.m00), static_cast<int>(moment.m01 / moment.m00));
                                std::string seat_label = "S" + std::to_string(state.seat_id) + " " + toString(state.occupancy_state);
                                cv::putText(vis, seat_label, center,
                                           cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
                            }
                        } else {         // fallback to rectangle if no effective polygon
                            cv::rectangle(vis, state.seat_roi, color, 3);
                            std::string seat_label = "Seat " + std::to_string(state.seat_id) + " " + toString(state.occupancy_state);
                            cv::putText(vis, seat_label, cv::Point(state.seat_roi.x, state.seat_roi.y - 10),
                                       cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
                        }
                    }
                    
                    // save annotated img
                    std::string file_name = entry.path().filename().string();
                    std::string annotated_path = (std::filesystem::path(cfg.annotated_frames_dir) / file_name).string();
                    
                    // skipping saving imgs
                    if (annotated_save_cnt < max_process_frames 
                        && annotated_save_cnt % annotated_save_freq == 0) {
                        ++annotated_save_cnt;
                        cv::imwrite(annotated_path, vis);    // write annotated img to specific path
                    } else if (annotated_save_cnt >= max_process_frames) {
                        std::cout << "Count of saved annotated images reached max saving frames limits: " << max_process_frames << "\n";
                    }
                    
                    std::string line = seatFrameStatesToJsonLine(states, ts, frame_index-1, src_path, annotated_path);
                    ofs << line << "\n";
                    { // update latest frame to .jsonl file
                        std::ofstream lf(latest_frame_file, std::ios::trunc);
                        if (lf) lf << line << "\n";
                    }
                    ++total_processed;
                    if (total_processed % print_every == 0) {  // processed progress & error cnt print
                        std::cout << "[progress] processed=" << total_processed << " errors=" << total_errors << "\n";
                    }
                    if (total_processed >= max_process_frames) { // max process frames cnt check
                        std::cout << "Reached --max limit " << max_process_frames << "; stopping directory iteration.\n";
                        break;
                    }

                } catch (const std::exception &exception) {  // catch exceptions per frame process
                    ++total_errors;
                    std::cerr << "Frame error: " << exception.what() << " src=" << src_path << "\n";
                } catch (...) {
                    ++total_errors;
                    std::cerr << "Frame error: unknown src=" << src_path << "\n";
                }
            }
        
        } else {// Video Processing
            
            // Streaming Process video: Extract and Process Frame-by-Frame from Video
            if (stream_video) { 
                /* Note: Here the iterate func. will conduct the extraction,
                    onFrame is only an arg which has a type of boolean func. here.
                    The onFrame func. is implemented here, and the iterate func. is
                    implemneted in FrameExtraction.cpp. So the iterate func. need arg
                    stream_video to decide its extraction way. The onFrame only 
                    responsible for processing and decide whether continue sampling
                */

                // 文件：按视频处理并逐帧送入 VisionA
                std::cout << "Input is a video file, mode streaming, iterating frames...\n";
                auto src_path = input_path.string();
                size_t processed = 0;

                // iterate over the video to extract & process frames 1-by-1
                FrameProcessor::streamProcess(
                    src_path,                                                         // video path
                    [&](int frame_idx, const cv::Mat& bgr, double /*tsec*/) -> bool { // conduct onFrame process
                        return FrameProcessor::onFrame(
                                        frame_idx,
                                        bgr, 
                                        /*t_sec*/0.0, 
                                        now_ms(), 
                                        input_path, 
                                        src_path, 
                                        ofs, 
                                        cfg, 
                                        vision, 
                                        latest_frame_file, 
                                        processed);
                    },                        
                    extract_fps,                                                      // extract fps
                    0,                                                                // start frame idx
                    -1                                                                // ending frame idx (-1: all frames) 
                );

                // output processed frames cnt
                std::cout << "Processed video frames: " << processed << "\n";

                    /*  old logic, just omit it  
                        // extract and process frame-by-frame from video
                        auto states = vision.processFrame(bgr, now_ms(), frame_index++);
                    
                        // output in CLI
                        int64_t ts = states.empty() ? now_ms() : states.front().ts_ms;
                        for (auto &s : states) {
                            std::cout << s.seat_id << " " << toString(s.occupancy_state)
                                      << " pc = " << s.person_conf_max
                                      << " oc = " << s.object_conf_max
                                      << " fg = " << s.fg_ratio
                                      << " snap = " << (s.snapshot_path.empty() ? "-" : s.snapshot_path)
                                      << "\n";
                        }
                    
                        // visualize person/object detecting results in every frame
                        std::vector<BBox> all_persons, all_objects;
                        vision.getLastDetections(all_persons, all_objects);
                        cv::Mat vis = bgr.clone();
                        for (auto &p : all_persons) {   // person boxes
                            cv::rectangle(vis, p.rect, cv::Scalar(255,0,255), 2);
                            std::string label = "person " + std::to_string(int(p.conf * 100)) + "%";
                            cv::putText(vis, label, cv::Point(p.rect.x, p.rect.y - 5),
                                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,255), 1);
                        }
                        for (auto &o : all_objects) {   // object boxes
                            cv::rectangle(vis, o.rect, cv::Scalar(255,255,0), 2);
                            std::string label = o.cls_name + " " + std::to_string(int(o.conf * 100)) + "%";
                            cv::putText(vis, label, cv::Point(o.rect.x, o.rect.y - 5),
                                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,0), 1);
                        }
                    
                        // record annotated frame
                        auto stem = input_path.stem().string();
                        char buf[64];
                        std::snprintf(buf, sizeof(buf), "%s_%06d.jpg", stem.c_str(), frame_idx);
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
                    },*/
            
            } else { /* Extract frames from video then process frames in directory
                
            */


            }

            // 视频：先提取帧到 data/frames/frames_vNNN，再统一按目录流程处理
            std::cout << "Input is a video file, " << (stream_video ? "streaming frames..." : "extracting frames then processing...") << "\n";
            std::filesystem::path frames_root = "data/frames";
            std::error_code error_code_frame_root;
            std::filesystem::create_directories(frames_root, error_code_frame_root);
            int next_idx = 1;
            for (auto &d : std::filesystem::directory_iterator(frames_root)) {
                if (d.is_directory()) {
                    auto name = d.path().filename().string();
                    if (name.rfind("frames_v", 0) == 0) {
                        try { int v = std::stoi(name.substr(8)); if (v >= next_idx) next_idx = v + 1; } catch (...) {}
                    }
                }
            }
            
            // extract frames from video
            char bufFolder[32];
            std::snprintf(bufFolder, sizeof(bufFolder), "frames_v%03d", next_idx);
            auto extractDir = frames_root / bufFolder;
            std::cout << "Extracting frames to: " << extractDir.string() << "\n";
            
            // 使用抽帧: 2 fps，避免一次性写出全部帧导致耗时/IO压力
            // 2fps too freq
            size_t processed_frames = 0;
            if (stream_video) {
                // 直接迭代视频帧
                FrameProcessor::streamProcess(
                    input_path.string(),
                    [&](int frameIdx, const cv::Mat& bgr, double /*tsec*/) -> bool {
                        if (bgr.empty()) return true;
                        try {
                            
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


                            char buf[64];
                            std::snprintf(buf, sizeof(buf), "stream_%06d.jpg", frameIdx);
                            std::string annotated_path = (std::filesystem::path(cfg.annotated_frames_dir) / buf).string();
                            cv::imwrite(annotated_path, vis);
                            std::string line = seatFrameStatesToJsonLine(states, ts, frame_index-1, input_path.string(), annotated_path);
                            ofs << line << "\n";
                            {
                                std::ofstream lf(latest_frame_file, std::ios::trunc);
                                if (lf) lf << line << "\n";
                            }
                            ++processed_frames; ++total_processed;
                            if (processed_frames % print_every == 0) {
                                std::cout << "[progress] processed=" << processed_frames << " errors=" << total_errors << "\n";
                            }
                            if (processed_frames >= max_process_frames) {
                                std::cout << "Reached --max limit " << max_process_frames << "; stopping video stream.\n";
                                return false; // stop
                            }
                        } catch (const std::exception &ex) {
                            ++total_errors;
                            std::cerr << "Video frame error: " << ex.what() << " idx=" << frameIdx << "\n";
                        } catch (...) {
                            ++total_errors;
                            std::cerr << "Video frame error: unknown idx=" << frameIdx << "\n";
                        }
                        return true;
                    },
                    extract_fps,            // extract fps
                    0                      // start frame idx
                    //-1                      // ending frame idx (-1: all frames)
                );
            } else {
                // 先抽帧到目录再统一处理
                size_t saved = FrameProcessor::extractToDir(input_path.string(), extractDir.string(), extract_fps, 95);
                std::cout << "Extracted " << saved << " frames from video." << "\n";
                
                
                for (auto &entry : std::filesystem::directory_iterator(extractDir)) {
                    if (!entry.is_regular_file()) continue;
                    if (entry.path().extension() != ".jpg" && entry.path().extension() != ".png") continue;
                    std::string src_path = entry.path().string();
                    cv::Mat bgr = cv::imread(src_path);
                    if (bgr.empty()) continue;
                    
                    
                    try {
                        
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


                        std::string fname = entry.path().filename().string();
                        std::string annotated_path = (std::filesystem::path(cfg.annotated_frames_dir) / fname).string();
                        cv::imwrite(annotated_path, vis);
                        std::string line = seatFrameStatesToJsonLine(states, ts, frame_index-1, src_path, annotated_path);
                        ofs << line << "\n";
                        {
                            std::ofstream lf(latest_frame_file, std::ios::trunc);
                            if (lf) lf << line << "\n";
                        }
                        ++processed_frames; ++total_processed;
                        if (processed_frames % print_every == 0) {
                            std::cout << "[progress] processed=" << processed_frames << " errors=" << total_errors << "\n";
                        }
                        if (processed_frames >= max_process_frames) {
                            std::cout << "Reached --max limit " << max_process_frames << "; stopping extracted frame iteration.\n";
                            break;
                        }
                    } catch (const std::exception &ex) {
                        ++total_errors;
                        std::cerr << "Extracted frame error: " << ex.what() << " src=" << src_path << "\n";
                    } catch (...) {
                        ++total_errors;
                        std::cerr << "Extracted frame error: unknown src=" << src_path << "\n";
                    }
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& fe) {
        std::cerr << "filesystem error: " << fe.what() << "\n";
        return 1;
    } catch (const std::exception &ex) {
        std::cerr << "Unhandled exception: " << ex.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Unhandled unknown exception.\n";
        return 1;
    }
    std::cout << "Seat states appended to: " << out_states << "\n";
    std::cout << "Summary: processed=" << total_processed << " errors=" << total_errors << "\n";
    std::cout << "Latest frame snapshot: " << latest_frame_file << "\n";
    return 0;
}