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
#include "vision/Types.h"
#include "vision/FrameProcessor.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <algorithm>
#include <set>
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
    std::string img_dir;                               // --framesrc 输入图像目录或视频文件
    std::string override_out_states;                   // --out 覆盖输出文件
    size_t max_process_frames = SIZE_MAX;              // --max 处理帧数上限
    double extract_fps = 2.0;                          // --fps 视频抽帧频率
    bool stream_video = false;                         // --stream false: 直接逐帧, 不落盘抽帧目录; true: 抽帧落盘后处理
    std::set<std::string> on_bag = {"1", "true", "True", "t", "T", "yes", "y", "on", "On", "ON", "v", "V"};
    std::set<std::string> off_bag = {"0", "false", "False", "f", "F", "no", "n", "off", "Off", "OFF", "x", "X"};
    
    //cv::setNumThreads(1);  // disable multithreading to avoid oneTBB/parallel backend DLL dependency issues
    //cv::setUseOptimized(false);  // disable OpenCV optimizations to avoid oneTBB/parallel backend DLL dependency issues

    // anal args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::cout << "[Main] Parsing arg: " << arg << "\n";
        if        (std::filesystem::is_directory(std::filesystem::path(arg)) || arg.find(".mp4") != std::string::npos) {
            try { img_dir = arg; } catch (...) { std::cerr << "[Main] Error in first arg (input file / directory) identification\n"; }                             // 输入路径为图像目录
        } else if (arg == "--framesrc" && i + 1 < argc) {   // --framesrc [img_dir]        // input image directory or video file
            try { img_dir = argv[i + 1]; } catch (...) { std::cerr << "[Main] Error in arg --framesrc\n"; }
        } else if (arg == "--out" && i + 1 < argc) {   // --out [override_out_states] // output states file directory
            try { override_out_states = argv[i + 1]; } catch (...) { std::cerr << "[Main] Error in arg --out\n"; }
        } else if (arg == "--max" && i + 1 < argc) {   // --max [N]                   // set max process frames
            try { max_process_frames = static_cast<size_t>(std::stoll(argv[i + 1])); } catch (...) { std::cerr << "Error in arg --max\n"; }
        } else if (arg == "--fps" && i + 1 < argc) {   // --fps [F]                   // set extract fps for video input
            try { extract_fps = std::stod(argv[i + 1]); } catch (...) { std::cerr << "[Main] Error in arg --fps\n"; }
        } else if (arg == "--stream" && i + 1 < argc) {// --stream                    // set stream video mode
            try { stream_video = (on_bag.count(argv[i + 1]) > 0) ? true : (off_bag.count(argv[i + 1]) > 0 ? false : false); } catch (...) { std::cerr << "Error in arg --stream\n" << i << "\n"; }
        } else if (arg == "-h" || arg == "--help") {   // --help                      // help
            std::cout << "\n"
                      << "Usage: a_demo [input_path] [--out states.jsonl] [--max N] [--fps F] [--stream false/true]\n"
                      << "       ./build/a_demo.exe [input_path] [--out states.jsonl] [--max N] [--fps F] [--stream false/true]\n"
                      << "Or:    a_demo -h \n"
                      << "       a_demo --help \n"
                      << "for help.\n\n"
                      << "For file type, if pure images are to be processed, fill the \"--out\" term with the directory to the images; \n"
                      << "      otherwise, fill in the video(.mp4) relative to current CWD, with .mp4 postfix ending. \n"
                      << "For the image processing mode, fill the \"--fps\" term with number of images you want to process per 100 images\n" 
                      << "      in the directory. (20 fp100 is recommended, which is also the default setting.)\n"
                      << "For the video processing mode, fill the \"--fps\" term with the desired extraction framerate (e.g. 2.0 for 2 \n" 
                      << "      frames per second)(the lower the fps is, the less frame it will extract and process from one second of\n"
                      << "      video, and thus less stressful for the program to work.)\n"
                      << "For video input mode, to process video without intermediate frame extraction to disk, set \"--stream true\". \n" 
                      << "      Otherwise, \"--stream false\" will start the bulk extraction process.\n"
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
                override_out_states = "runtime/seat_states.jsonl";  // parent directory!! NOT include seat_states.jsonl";
                std::cout << "[Main] override_out_states = " << override_out_states << "\n";
                std::cout << "[Main] Warning: second argument (for output state file directory) provided is not a directory, using default ./runtime/ instead for output states file.\n";
            }
        }
    }

    std::cout << "[Main] a_demo starting...\n";
    std::cout << "[Main] CWD: " << std::filesystem::current_path().string() << "\n";         // current working directory
    std::cout << "[Main] input: " << img_dir << "\n";
    std::cout << "[Main] output states file: " << (override_out_states.empty() ? "None" : override_out_states) << "\n";
    std::cout << "[Main] options: max=" << (max_process_frames==SIZE_MAX? -1 : (long long)max_process_frames) << "\n"
              << "                fps=" << extract_fps << "\n"
              << "                stream=" << (stream_video ? true : false) << "\n";
    std::cout.flush();

    // Load config with internal exception-safety (fromYaml keeps defaults on failure)
    VisionConfig cfg = VisionConfig::fromYaml("config/vision.yml");
    if (!std::filesystem::exists("config/vision.yml")) {  // yaml existence
        std::cerr << "[Main] config/vision.yml not found relative to CWD.\n";
        return 1;
    }
    if (!std::filesystem::exists(cfg.seats_json)) {       // seat json existence
        std::cerr << "[Main] seats json not found: " << cfg.seats_json << "\n";
        return 1;
    }
    if (!std::filesystem::exists(img_dir)) {              // input path existence
        std::cerr << "[Main] Input path not found: " << img_dir << "\n";
        std::cerr << "       Hint: use a directory of images or a video file path.\n";
        return 1;
    }
    
    // 使用配置中的 states_output 输出座位状态记录(runtime/seat_states.jsonl)，若用户cli提供则覆盖
    std::string out_states_path = (override_out_states.empty() || override_out_states != "runtime/seat_states.jsonl") ? cfg.states_output : override_out_states;
    std::cout << "[Main] Output states file: " << out_states_path << "\n";
    
    VisionA vision(cfg);
    std::cout << "[Main] Loaded seats from " << cfg.seats_json 
              << ": count=" << vision.seatCount() 
              << "\n";
    
    // Publish part
    Publisher pub;
    pub.setCallback([](const std::vector<SeatFrameState>& states){
        std::cout << "Callback batch size = " << states.size() << "\n";
    });
    vision.setPublisher(&pub);

    // create output path (skip if no parent path included)
    int64_t frame_index = 0;                                             // index of img handled this patch
    auto output_state_parent_path = std::filesystem::path(out_states_path).parent_path();  // parent path of the output file

    // std::error_code error_code;
    // if (!output_state_parent_path.empty()) {
    //     std::filesystem::create_directories(output_state_parent_path, error_code);
    //     if (error_code) {
    //         std::cerr << "[Main] Failed to create output state directory: "
    //                   << output_state_parent_path.string() << " : " << error_code.message() << "\n";
    //         return 1;
    //     }
    // }

    if (output_state_parent_path.empty()) {     
        std::error_code error_code_;
        std::filesystem::create_directories(output_state_parent_path, error_code_);
        if (error_code_) {
            std::cerr << "[Main] Failed to create output state directory: " << output_state_parent_path.string() << " : " << error_code_.message() << "\n"
                      << "       Hint: check if the path is valid. By default the path for saving output states is ./runtime/seat_states.jsonl \n";
            return 1;
        }
    }
    
    // open output states file (append mode)
    std::ofstream ofs(out_states_path, std::ios::app);

    if (!ofs) {                     // error check: open output states file
        std::cerr << "[Main] Failed to open output states file: " << out_states_path << "\n";
        return 1;
    }
    std::cout << "[Main] States output file: " << std::filesystem::absolute(out_states_path).string() << "\n";
    
    // frame annotated directory
    std::error_code error_code_marker;
    std::filesystem::create_directories(cfg.annotated_frames_dir, error_code_marker);

    // judge input type
    std::string input_type_str;
    switch (vision::FrameProcessor::judgeInputType(img_dir)) {
        case vision::InputType::DIRECTORY_IMAGE:
            input_type_str = "IMAGE";
            break;
        case vision::InputType::VIDEO_FILE:
            input_type_str = "VIDEO";
            break;
        case vision::InputType::IMAGE_FILE:
            input_type_str = "IMAGE";
            break;
        case vision::InputType::NOT_EXISTS:
            std::cerr << "[Main] Input path not exists: " << img_dir << "\n";
            return 1;
        case vision::InputType::UNKNOWN:
            input_type_str = "UNKNOWN";
            std::cerr << "[Main] Input path type UNKNOWN: " << img_dir << "\n"
                      << "       Maybe not a supported format. .jpg/.png are recommendde for images; .mp4 for video.\n"
                      << "       Hint: use a directory of images or a video file path.\n";
            return 1;
            break;
        default:
            input_type_str = "UNKNOWN";
            break;
    }
    std::cout << "[Main] Input type identified as: " << input_type_str << "\n";

    // Formal process
    size_t total_processed = 0;
    size_t total_errors = 0;
    int print_every = 100; // 进度打印间隔
    int annotated_save_cnt = 0;
    int annotated_save_freq = cfg.annotated_save_freq;
    
    try {    // Extract and Process   
        auto input_path = std::filesystem::path(img_dir); // here the input_path can be video file or directory of img
        auto input_path_string = input_path.string();

        // criterion cannot use "is_directory": video can be identified as directory

        // judging images or video to be processed
        
        // Images Processing (Directly, NO Extraction)
        if (input_type_str == "IMAGE") {  // process frame from img directory

            // conduct processing via imageProcess
            size_t processed = vision::FrameProcessor::imageProcess(
                input_path_string,                  // images directory path string
                output_state_parent_path.string(),  // output states parent path
                ofs,                                // output file stream
                cfg,                                // VisionConfig
                vision,                             // VisionA
                max_process_frames,                 // max process frames
                100,                                 // frames to sample per 100 images
                0                                   // original total index offset (will recheck cnt of all images in directory if 0 provided)
            );

            // output processed frames cnt
            std::cout << "[Main] Processed image frames: " << processed << "\n";

            // publish record

            // receive frame idx to be annotated

            // annotation

            // publish annotation

        } else if (input_type_str == "VIDEO") {// Video Processing
            
            std::cout << "Input is a video file, " << (stream_video ? "streaming frames..." : "extracting frames then processing...") << "\n";

            // Streaming Process video: Extract and Process Frame-by-Frame from Video
            if (stream_video) { 

                // iterate over the video to extract & process frames 1-by-1
                size_t processed = vision::FrameProcessor::streamProcess(
                    input_path_string,                 // video path
                    output_state_parent_path.string(), // output states parent path                        
                    vision,                            // VisionA
                    cfg,                               // VisionConfig
                    ofs,                               // output file stream
                    extract_fps,                       // extract fps
                    0,                                 // start frame idx
                    -1,                                // ending frame idx (-1: all frames) 
                    max_process_frames                 // maximum frames to process
                );

                // output processed frames cnt
                std::cout << "[Main] Processed video frames: " << processed << "\n";

                // publish record

                // receive frame idx to be annotated

                // annotation

                // publish annotation
            
            } else {  /* Bulk Process Video: Extract all sampled frames to directory then process them */

                // extract all out then process
                size_t processed = vision::FrameProcessor::bulkProcess(
                    input_path_string,                        // video_path
                    output_state_parent_path.string(),        // latest_frame_dir (for last_frame.json)
                    cfg,                                      // VisionConfig
                    ofs,                                      // output file stream
                    vision,                                   // VisionA
                    extract_fps,                               // sample_fps
                    0,                                        // start_frame_idx
                    -1,                                       // end_frame_idx (-1: all frames)
                    std::string("./data/frames"),             // img_dir to store extracted frames
                    max_process_frames,                       // max_process_frames
                    95,                                       // jpg_quality
                    "f_"                                      // filename_prefix
                );

                // output processed frames cnt
                std::cout << "[Main] Processed video frames: " << processed << "\n";

                // publish record

                // receive frame idx to be annotated

                // annotation

                // publish annotation

            }
        } else {
            std::cerr << "[Main] Unsupported input type for path: " << img_dir << "\n";
            return 1;
        }
              
    } catch (const std::filesystem::filesystem_error& filesystem_error) {  // Filesystem exception
        std::cerr << "[Main] Filesystem error: " << filesystem_error.what() << "\n";
        return 1;
    } catch (const std::exception &exception) { // General exception
        std::cerr << "[Main] Unhandled exception: " << exception.what() << "\n";
        return 1;
    } catch (...) {  // Unknow exception
        std::cerr << "[Main] Unhandled unknown exception.\n";
        return 1;
    }
    std::cout << "[Main] Seat states appended to: " << out_states_path << "\n";
    std::cout << "[Main] Summary: processed=" << total_processed << " errors=" << total_errors << "\n";
    std::cout << "[Main] Latest frame snapshot: " << output_state_parent_path.string() + "/last_frame.json" << "\n";
    return 0;
}