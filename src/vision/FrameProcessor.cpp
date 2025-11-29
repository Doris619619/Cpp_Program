#include "vision/FrameProcessor.h"
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <set>
#include "vision/VisionA.h"
#include "vision/Publish.h"
#include "vision/Config.h"
#include "vision/Types.h"
#include <opencv2/opencv.hpp>
//#include <filesystem>
#include <chrono>
#include <fstream>
#include <cstddef>

namespace fs = std::filesystem;

namespace vision {

// ==================== Core: Processing ===========================

/* onFrame 对每帧照片处理   

  @param frame_index         current frame index  
  @param bgr:                current frame image in BGR format
  @param t_sec:              current frame timestamp in seconds
  @param now_ms:             current system time in milliseconds
  @param input_path:         path of the input image or video
  @param video_src_path:     source path of the video file
  @param ofs:                output file stream for recording annotated frames
  @param cfg:                configuration settings for vision processing
  @param vision:             instance of VisionA for processing frames
  @param latest_frame_file:  path to the file storing the latest frame information
  @param processed:          reference to a counter for processed frames

  @note Logic
  @note - process frame by vision.processFrame() and receive the state
  @note - based on the state, output the content in the cli
  @note - DO NOT conduct visualization here, hand it over to the other method
  @note - record annotated frame and output in the jsonl file
  @note - DO NOT responsible for judging whether to save-in-disk, return bool for handled or not
*/
bool vision::FrameProcessor::onFrame(
    int frame_index, 
    const cv::Mat& bgr, 
    double /*t_sec*/, 
    int64_t now_ms,
    const std::filesystem::path& input_path,   // path (img path / video file)
    const std::string& annotated_frames_dir,
    std::ofstream& ofs,
    VisionA& vision,
    const std::string& latest_frame_file,
    size_t& processed
) {
    // report onFrame
    std::cout << "[FrameProcessor] onFrame called for frame index: " << frame_index << ". Start processing..." << "\n";

    // process frame
    auto states = vision.processFrame(bgr, now_ms, frame_index++);

    // output in CLI
    int64_t ts = states.empty() ? now_ms : states.front().ts_ms;
    for (auto &s : states) {
        std::cout << "[FrameProcessor] Processed Frame " << frame_index << " @ " << ts << " ms: "
                  << " seat = " << s.seat_id << " " << toString(s.occupancy_state)
                  << " pc = " << s.person_conf_max
                  << " oc = " << s.object_conf_max
                  << " fg = " << s.fg_ratio
                  << " snap = " << (s.snapshot_path.empty() ? "-" : s.snapshot_path)
                  << "\n";
    }

    // annotation imlementation

    // record annotated frame
    auto stem = input_path.stem().string();
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s_%06d.jpg", stem.c_str(), frame_index);
    std::string annotated_path = (std::filesystem::path(annotated_frames_dir) / buf).string();
    //cv::imwrite(annotated_path, vis);
    std::string line = seatFrameStatesToJsonLine(states, ts, frame_index-1, input_path.string(), annotated_path);
    ofs << line << "\n";
    {
        std::ofstream lf(latest_frame_file, std::ios::trunc);
        if (lf) lf << line << "\n";
    }
    ++processed;

    // when will return false? 
    return true;
}

// Stream Processing Video
size_t vision::FrameProcessor::streamProcess(
    const std::string& video_path,         // video file path
    const std::string& latest_frame_dir,   // output states parent path                        
    VisionA& vision,                       // VisionA
    const VisionConfig& cfg,               // VisionConfig
    std::ofstream& ofs,                    // output file stream
    double sample_fps,                     // sampling_freq
    int start_frame,                       // first_frame index = 0
    int end_frame,                         // end_frame index = -1 (the ending frame)
    size_t max_process_frames              // maximum frames to process
) {    
    // frame capturer initialization
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {  // open video failed
        std::cerr << "[FrameProcessor] Failed to open video: " << video_path << "\n";
        return false;
    }

    // derive frame sample args
    const double original_fps = cap.get(cv::CAP_PROP_FPS);
    int original_total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int t_ori_spf = static_cast<int>(1 / original_fps);
    if (end_frame < 0 && original_total_frames > 0) end_frame = original_total_frames - 1;                          

    // sample args setting
    const bool do_sample = (sample_fps > 0.0);
    int sample_stepsize = do_sample ? (sample_fps >= original_fps ? original_fps * 5 : 1 * sample_fps) : 0; // sampleing stepsize is of frames index / cnt, not time interval
    double t_next_sample = 0.0;                                         // next sampling time thresh (seconds)
    int sample_cnt_ub = 0;
    size_t processed_cnt = 0;
    size_t total_errors = 0;
    const std::string annotated_frames_dir = !cfg.annotated_frames_dir.empty() ? cfg.annotated_frames_dir : "data/annotated_frames"; // directory to save annotated frames
    auto input_path = std::filesystem::path(video_path);

    // sampling fps safety check
    if (original_total_frames > 6000) {
        sample_cnt_ub = 600;
        sample_stepsize = std::max(sample_stepsize, static_cast<int>(original_total_frames / sample_cnt_ub));
    } else if (original_total_frames > 4000) {
        sample_cnt_ub = static_cast<int>(original_total_frames / 20) + 1;
        sample_stepsize = std::max(sample_stepsize, static_cast<int>(original_total_frames / sample_cnt_ub));
    } else if (original_total_frames > 1000) {
        sample_cnt_ub = static_cast<int>(original_total_frames / 50) + 1;
        sample_stepsize = std::max(sample_stepsize, static_cast<int>(original_total_frames / sample_cnt_ub));
    } else if (original_total_frames > 200) {
        sample_cnt_ub = static_cast<int>(original_total_frames / 20) + 1;
        sample_stepsize = std::max(sample_stepsize, static_cast<int>(original_total_frames / sample_cnt_ub));
    } else {
        sample_cnt_ub = original_total_frames;
        sample_stepsize = std::max(sample_stepsize, static_cast<int>(original_total_frames / sample_cnt_ub));
    }

    std::cout << "[FrameProcessor] Streaming video mode. Iterating frames...\n";

    // streaming video: Extract and Process Frame-by-Frame from Video
    for (int idx = start_frame, sample_cnt = 0; idx < original_total_frames, sample_cnt < sample_cnt_ub; idx += sample_stepsize, sample_cnt++) {
        
        // test iteration
        std::cout << "[FrameProcessor] Processing frame index: " << idx << "\n";
        
        // skip-sampling
        cap.set(cv::CAP_PROP_POS_FRAMES, idx); // skip frames according to stepsize

        // read-in frame
        cv::Mat bgr;  // in loop, everytime call cap.read(bgr), it will automatically move to next frame
        if (!cap.read(bgr)) {  // EOF
            std::cerr << "[FrameProcessor] Reached end of video or read error at frame index " << idx << "\n";
            break; 
        }

        // process frame with exception handling
        try {
            // derive current timestamp ms and s (sec)
            double t_ms = cap.get(cv::CAP_PROP_POS_MSEC);
            double t_sec = (t_ms > 1e-6) ? (t_ms / 1000.0) : (original_fps > 0.0 ? (static_cast<double>(idx) / original_fps) : 0.0);
            int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();

            // process frame
            bool continue_process = FrameProcessor::onFrame(
                idx,
                bgr,
                t_sec,
                now_ms,
                input_path.string(),
                annotated_frames_dir,
                ofs,
                vision,
                (fs::path(latest_frame_dir) / "last_frame.jsonl").string(),
                processed_cnt
            );
            processed_cnt++;

            // ending check
            if (end_frame >= 0 && idx >= end_frame) break;
            if (!continue_process || processed_cnt >= max_process_frames) {// termination 
                std::cout << "[FrameProcessor] Stopping at frame " << idx << "\n"
                          << "                 onFrame reported: " << (continue_process ? ("frame " + std::to_string(idx) + " handled, max process amount reached.") 
                                                                                        : "truncation requested at frame " + std::to_string(idx) + ".") << "\n";
                break;
            }

            // sample saving logic
            /* not urgent */
        } catch (const std::exception& exception) {
            total_errors++;
            std::cerr << "[FrameProcessor] Exception at frame index " << idx << ": " << exception.what() << "\n";
        } catch (...) {
            total_errors++;
            std::cerr << "[FrameProcessor] Unknown Exception at frame index " << idx << "\n";
        }
    }

    // final output
    std::cout << "[FrameProcessor] streamProcess completed: processed=" << processed_cnt << "\n                 "
              << "errors=" << total_errors << "\n                 "
              << "original total frames=" << original_total_frames 
              << ", original fps=" << std::fixed << std::setprecision(2) << original_fps << "\n";

    return processed_cnt;
}

// Bulk Extracting Video Frames 批量提取视频帧为图像文件
size_t vision::FrameProcessor::bulkExtraction(
    const std::string& video_path,
    double sample_fps,                             // sampling fps = extract_fps, which is the program input arg
    const std::string& out_dir,                    // extract to data/frames/frames_vNNN/
    int start_frame,
    int end_frame,
    int jpeg_quality,
    const std::string& filename_prefix
) {
    // find or create output directory
    std::string actual_out_dir = FrameProcessor::getExtractionOutDir(out_dir);

    // init frame capturer and open video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) { // open video failed
        std::cerr << "[FrameProcessor] bulkExtraction open failed: " << video_path << "\n";
        return 0;
    }

    // arg check
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double original_fps = cap.get(cv::CAP_PROP_FPS);
    if (end_frame < 0 && total_frames > 0) end_frame = total_frames - 1;
    if (start_frame < 0) start_frame = 0;
    if (end_frame >= 0 && end_frame < start_frame) end_frame = start_frame;

    // set sample stepsize (of frame index / cnt, not time interval)
    int sample_stempsize = 1;
    if (sample_fps > 0.0 && original_fps > 0.0) {
        sample_stempsize = getStepsize(static_cast<size_t>(total_frames), (sample_fps / original_fps <= 1 ? static_cast<int>(100 * sample_fps / original_fps) : 20));
        if (sample_stempsize < 1) sample_stempsize = 1;
    }

    std::cout << "[FrameProcessor] Bulk extracting and processing video mode. Extracting frames...\n";

    // extract frames
    size_t extracted_cnt = 0;
    int consecutive_failures_cnt = 0;
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, std::clamp(jpeg_quality, 1, 100) };
    for (int idx = start_frame; end_frame < 0 ? true : (idx <= end_frame); idx += sample_stempsize) {
        // skip-sampling
        cap.set(cv::CAP_PROP_POS_FRAMES, idx);
        cv::Mat bgr;
        if (!cap.read(bgr) || bgr.empty()) { // read failed
            std::cerr << "[FrameProcessor] bulkExtraction read failed at frame index " << idx << "\n";
            ++consecutive_failures_cnt;
            if (consecutive_failures_cnt >= 3) {  // consecutive 3 failures
                std::cerr << "[FrameProcessor] bulkExtraction stopping due to 3 consecutive read failures.\n";
                break;
            }
            continue;
        }
        consecutive_failures_cnt = 0;

        // derive output file path
        std::ostringstream oss;
        oss << filename_prefix << std::setw(6) << std::setfill('0') << idx << ".jpg"; // img: prefix + 000000 + .jpg
        fs::path out_path = fs::path(actual_out_dir) / oss.str();

        // write check
        std::cout << "[FrameProcessor] bulkExtraction writing frame index " << idx << " to " << out_path.string() << "\n";

        if (!cv::imwrite(out_path.string(), bgr, params)) {
            std::cerr << "[FrameProcessor] bulkExtraction write failed: " << actual_out_dir << " at frame index " << idx << "\n";
        } else {
            ++extracted_cnt;
        }
    }

    std::cout << "[FrameProcessor] bulkExtraction completed: extracted=" << extracted_cnt << "\n"
              << "                 from video: " << video_path << "\n"
              << "                 to directory: " << out_dir << "\n"
              << "                 total frames in video: " << total_frames
              << ", original fps: " << std::fixed << std::setprecision(2) << original_fps << "\n"
              << "                 sampling stepsize: " << sample_stempsize << " (unit: frames)\n";

    return extracted_cnt;
}

/*  @brief Bulk Processing Video  批量处理视频帧
*
*   @param video_path:        视频文件路径
*   @param lastest_frame_dir: 最新帧文件目录 
*   @param cfg:              VisionConfig配置
*   @param ofs:              输出文件流
*   @param vision:           VisionA实例
*   @param sample_fps:       采样帧率
*   @param start_frame:      起始帧索引
*   @param end_frame:        结束帧索引
*   @param img_dir:           图像输出目录
*   @param max_process_frames: 最大处理帧数
*   @param jpeg_quality:      JPEG图像质量
*   @param filename_prefix:   输出文件名前缀
*
*   @return number of frames processed 处理的帧数
*/
size_t vision::FrameProcessor::bulkProcess(
    const std::string& video_path,                   // 
    const std::string& latest_frame_dir,             // 
    const VisionConfig& cfg,                         // VisionConfig used by onFrame
    std::ofstream& ofs,                              // output file stream
    VisionA& vision,                                 // VisionA
    double sample_fps,                               // 
    int start_frame,                                 //           
    int end_frame,                                   // 
    const std::string& img_dir,                      //
    size_t max_process_frames,                       // use user input --max
    int jpeg_quality,                                // 
    const std::string& filename_prefix               // 
) {
    std::cout << "[FrameProcessor] Bulk extracting and processing video mode. Extraction to be begin...\n";
    
    // check img dir
    if (!std::filesystem::exists(img_dir)) {
        std::filesystem::create_directories(img_dir);
        std::cout << "[FrameProcessor] bulkProcess created img_dir: " << img_dir << "\n";
    } else if (!std::filesystem::is_directory(img_dir)) {
        std::cerr << "[FrameProcessor] bulkProcess img_dir is not a directory: " << img_dir << "\n";
        return 1;
    }
    
    std::string actual_img_dir = "./runtime/frames";
    if (img_dir != "./runtime/frames") {
        std::string actual_img_dir = "./runtime/frames";
        std::cerr << "[FrameProcessor] bulkProcess default img_dir should be 'runtime/frames' to avoid conflicts: " << "\n";
    }

    // bulk extract frames (with sample)
    size_t extracted = bulkExtraction(video_path, sample_fps, actual_img_dir, start_frame, end_frame, jpeg_quality, filename_prefix);
    if (extracted == 0) return 0;

    // set stepsize 
    int process_stepsize = 1;  // difer from extraction sampling stepsize
    int total_errors = 0;
    int original_total_frames = static_cast<int>(cv::VideoCapture(video_path).get(cv::CAP_PROP_FRAME_COUNT));
    double original_fps = cv::VideoCapture(video_path).get(cv::CAP_PROP_FPS);
    /* since sampling has been conducted during extraction, 
       it's not urgent to add another sampling here.
       if still needed after consideration, add later.

       needed to mention that the stepsize here should be determined by 
       total img cnt and consider idx diff between each img, as they 
       have been sampled and thus idx name of imgs may not be consecutive,
       follow the rule found.
       
    */

    std::cout << "[FrameProcessor] Bulk extracting and processing video mode. Processing to be begin...\n";

    // process extracted frames
    size_t processed_cnt = 0;
    int consecutive_failures_cnt = 0;
    // for (auto& entry : std::filesystem::directory_iterator(img_dir)) {
    //     // check
    for (size_t idx = 0; idx < extracted; idx+=process_stepsize) {
        
        // check if img dir exists before processing
        if (!std::filesystem::exists(std::filesystem::path(img_dir))) {  // img: prefix + 000000 + .jpg (idx at 000000)
            std::cerr << "[FrameProcessor] bulkProcess imread failed: " << std::filesystem::path(img_dir).string() << "\n";
            ++consecutive_failures_cnt;
        if (consecutive_failures_cnt >= 3) {
            std::cerr << "[FrameProcessor] bulkProcess stopping due to 3 consecutive read failures.\n";
            break;
        }
            continue;
        }
        consecutive_failures_cnt = 0;

        // skip-sampling (detailed logic added later if needed)

        // process with failure handling
        try {
            // process frame via imageProcess
            FrameProcessor::imageProcess(
                img_dir, 
                latest_frame_dir,
                ofs, 
                cfg,
                vision,
                max_process_frames - processed_cnt,
                20,                                   // sample_fp100
                original_total_frames
            );

            processed_cnt++;

        } catch (const std::exception& exception){
            std::cerr << "[FrameProcessor] bulkProcess exception at image index " << idx << ": " << exception.what() << "\n";
            ++consecutive_failures_cnt;
        } catch (...) {
            std::cerr << "[FrameProcessor] bulkProcess unknown exception at image index " << idx << "\n";
            ++consecutive_failures_cnt;
        }
        
    }

    // final output
    std::cout << "[FrameProcessor] streamProcess completed: processed=" << processed_cnt << "\n                 "
              << "errors=" << total_errors << "\n                 "
              << "original total frames=" << original_total_frames 
              << ", original fps=" << std::fixed << std::setprecision(2) << original_fps << "\n";

    return processed_cnt;
}

/* imageProcess   
* 批量图像处理: 遍历目录下的所有图像文件并通过回调处理
* 
*  @param image_dir:            图像所在目录
*  @param latest_frame_dir:     最新帧文件路径
*  @param ofs:                  输出文件流
*  @param cfg:                  VisionConfig配置
*  @param vision:               VisionA实例
*  @param max_process_frames:   最大处理帧数
*  @param sample_fp100:         每100张图像采样的帧数
*  @param original_total_frames:原始总帧数索引偏移（如果提供0，则重新检查目录中所有图像的数量）
*  
*  @return  number of frames processed 处理的帧数
*/
size_t vision::FrameProcessor::imageProcess(
    const std::string& image_path,              // images directory path string
    const std::string& latest_frame_dir,        // output states parent path
    std::ofstream& ofs,                         // output file stream
    const VisionConfig& cfg,                    // VisionConfig
    VisionA& vision,                            // VisionA
    size_t max_process_frames,                  // max process frames
    int sample_fp100,                           // frames to sample per 100 images
    int original_total_frames                   // original total index offset (will recheck cnt of all images in directory if 0 provided)
) {
    // basic args
    size_t total_processed = 0;
    size_t total_errors = 0;
    int frame_index = 0;                        // differ from original_img_idx below: this is idx recorded in jsonl as the idx of frame processed
    size_t total_frames = (original_total_frames > 0) ? original_total_frames : countImageFilesInDir(image_path);
    const std::string annotated_frames_dir = !cfg.annotated_frames_dir.empty() ? cfg.annotated_frames_dir : "data/annotated_frames"; // directory to save annotated frames
    
    //sample_fp100 = 20;                                         // default sampling fps100 if needed later
    int sample_stepsize = getStepsize(total_frames, sample_fp100); // stepsize for sampling during processing
    int original_img_idx = 0;                                      // original image index during iteration
    
    // input path check
    if (std::filesystem::path(image_path).empty()) {
        std::cerr << "[FrameProcessor] imageProcess: empty image path provided.\n";
        std::error_code error_code;
        std::filesystem::create_directories(image_path, error_code);
        if (error_code) {
            std::cerr << "[FrameProcessor] imageProcess: create directory failed: " << image_path << " : " << error_code.message() << "\n";
            return 0;
        }
    }
    if (!std::filesystem::is_directory(image_path)) {  // open directory failed
        std::cerr << "[FrameProcessor] imageProcess: not a directory: " << image_path << "\n"
                  << "                 Hint: to process images, use directory to images as argument.\n"
                  << "                 e.g. --input /path/to/images/\n";
        return 0;
    }
    
    std::cout << "[FrameProcessor] Image directory mode. Iterating files...\n";
    
    // iteration on the imgs (y no sampling here? needed!!! )
    for (auto &entry : std::filesystem::directory_iterator(image_path)) {
        
        // image check
        auto ext = entry.path().extension().string();                   // extension name of the entry file with dot
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (!entry.is_regular_file() || (ext != ".jpg" && ext != ".jpeg" && ext != ".png")) {
            continue;                                                   // not a regular image file
        }

        // sampling logic
        if (original_img_idx % sample_stepsize != 0) {
            ++original_img_idx;
            continue;
        }
        ++original_img_idx; 

        // basic checks
        if (!entry.is_regular_file()) continue;
        try {
            std::cout << "[FrameProcessor] Processing image: " << entry.path().string() << "\n";
            cv::Mat bgr = cv::imread(entry.path().string());
        } catch (...) {
            ++total_errors;
            cv::Mat bgr;
            std::cerr << "[FrameProcessor] imageProcess imread exception: " << entry.path().string() << "\n";
            continue;
        }

        // conduct imread
        std::cout << "[FrameProcessor] Reading image: " << entry.path().string() << "\n";

        cv::Mat bgr = cv::imread(entry.path().string());
        // ================== DIAG (ADDED ONLY) ==================

        if (bgr.empty()) {
            std::cerr << "[FrameProcessor][diag] imread empty: " << entry.path().string() << "\n";
        }
        // =======================================================
        
        // process frame with exception handling
        try {
            // derive current system time in ms
            int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            // report processing
            std::cout << "[FrameProcessor] Processing frame index: " << frame_index << "\n";

            // process frame via onFrame
            bool continue_process = FrameProcessor::onFrame(
                frame_index,
                bgr,
                0.0,  // t_sec
                now_ms,
                entry.path(),    //::filesystem::path(image_path), // 建议改为 entry.path() (见下方注释)
                annotated_frames_dir,
                ofs,
                vision,
                (std::filesystem::path(latest_frame_dir) / "last_frame.jsonl").string(),
                total_processed
            );
            
            ++frame_index;
            ++total_processed;
            
            // report success in processing current image! 
            std::cout << "[FrameProcessor] Processed image: " << entry.path().string() << ", total processed: " << total_processed << "\n";

            if (!continue_process || total_processed >= max_process_frames) {  // termination 
                std::cout << "[FrameProcessor] Stopping at frame " << frame_index << " to be processed (the " << original_img_idx << " image in the directory)" << "\n"
                          << "                 onFrame reported: " << (continue_process ? ("frame " + std::to_string(frame_index) + " handled, max process amount reached.") 
                                                                                        : "truncation requested at frame " + std::to_string(frame_index) + ".") << "\n";
                break;
            }
            
        } catch (const std::exception &exception) { // exception handling
            ++total_errors;
            std::cerr << "[FrameProcessor] Frame error: " << exception.what() << " src=" << image_path << "\n";
        } catch (...) {    // unknown exception
            ++total_errors;
            std::cerr << "[FrameProcessor] Frame error: unknown src=" << image_path << "\n";
        }
    }

    std::cout << "[FrameProcessor] imageProcess completed: processed=" << total_processed << "\n                 "
              << "errors=" << total_errors << "\n                 "
              << "original total frames=" << total_frames << "\n                 "
              << "stepsize to process the images: " << sample_stepsize << "\n"; 
    
    return total_processed;
} 

// ==================== Utils: Sampling, Counting, and Mapping ===========================

static double safe_fps(cv::VideoCapture& cap) {
    double original_fps = cap.get(cv::CAP_PROP_FPS);  // video original fps
    if (original_fps < 1e-3 || std::isnan(original_fps)) return 0.0;
    else if (original_fps > 2.0)                         return 2.0; // cap the fps to 2.0 to avoid too dense sampling
    return original_fps;
}

// count files in specific directory
size_t vision::FrameProcessor::countFilesInDir(const std::string& dir_path) {
    std::error_code error_code;
    if (!std::filesystem::exists(dir_path, error_code) || !std::filesystem::is_directory(dir_path, error_code)) return 0;  // no dire exists || not a path
    size_t cnt = 0;
    for (auto& entry : std::filesystem::directory_iterator(dir_path, error_code)) {
        if (error_code) break;
        if (!entry.is_regular_file()) continue;
        ++cnt;
    }
    return cnt;
}

// count images files in specific directory (.jpg/.jpeg/.png)
size_t vision::FrameProcessor::countImageFilesInDir(const std::string& dir_path) {
    std::error_code error_code;
    if (!std::filesystem::exists(dir_path, error_code) || !std::filesystem::is_directory(dir_path, error_code)) return 0;   // no dire exists || not a path
    size_t cnt = 0;
    for (auto& entry : std::filesystem::directory_iterator(dir_path, error_code)) {
        if (error_code) break;
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();                   // extension name of the entry file with dot
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") ++cnt;
    }
    return cnt;
}

// get stepsize based on img cnt
int vision::FrameProcessor::getStepsize(size_t image_count) {
    if (image_count <= 500) return 5;
    if (image_count <= 1000) return 10;
    return 50;
}

// get stepsize based on img cnt and sample_fps100
int vision::FrameProcessor::getStepsize(size_t image_count, int sample_fp100) {
    if (sample_fp100 <= 0) return getStepsize(image_count);
    if (sample_fp100 > 100) sample_fp100 = 20; // 超过100则按安全默认20 fp100
    int step = static_cast<int>(std::floor(100 / sample_fp100)) + 1;
    return std::max(step, 1);
}

// get extraction output directory
std::string FrameProcessor::getExtractionOutDir(const std::string& out_dir){
    std::filesystem::path frames_root = std::filesystem::path((out_dir.empty()) ? "./runtime/frames" : out_dir);
    std::error_code error_code;
    if (!std::filesystem::exists(out_dir)) { // output directory not exists
        std::filesystem::create_directories(frames_root, error_code);
        if (error_code) {     // create directory failed
            std::cerr << "[FrameProcessor] bulkExtraction create out dir failed: " << out_dir << " : " << error_code.message() << "\n";
            return frames_root.string();
        }
    }
    
    // int next_idx = 1;
    // for (auto &d : std::filesystem::directory_iterator(frames_root)) {
    //     if (d.is_directory()) {
    //         auto name = d.path().filename().string();
    //         if (name.rfind("frames_v", 0) == 0) {
    //             try { 
    //                 int idx = std::stoi(name.substr(8)); 
    //                 if (idx >= next_idx) next_idx = idx + 1; 
    //             } catch (...) {
    //                 std::cerr << "[FrameProcessor] getExtractionOutDir: invalid directory name found: " << name << "\n";
    //                 return frames_root.string();
    //             }
    //         }
    //     }
    // }

    //char buf_folder[32];
    //std::snprintf(buf_folder, sizeof(buf_folder), "frames_v%03d", next_idx);
    //auto extract_dir = frames_root / buf_folder;
    //std::cout << "[FrameProcessor] Extracting frames to: " << extract_dir.string() << "\n";
    std::cout << "[FrameProcessor] Extracting frames to: " << frames_root.string() << "\n";

    //return extract_dir.string();
    return frames_root.string();
}


// judge if is video by extension
bool vision::FrameProcessor::isVideoFile(const std::string& file_path_string) {
    auto file_path = std::filesystem::path(file_path_string);
    auto ext = file_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    static const std::set<std::string> exts = {
        ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".mpg", ".mpeg", ".m4v", ".ts", ".mts"
    };
    return exts.count(ext) > 0;
}

// judge if is image by extension
bool vision::FrameProcessor::isImageFile(const std::string& file_path_string) {
    auto file_path = std::filesystem::path(file_path_string);
    auto ext = file_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    static const std::set<std::string> exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
    };
    return exts.count(ext) > 0;
}

// judge input type (directory, video file, image file, unknown)
InputType vision::FrameProcessor::judgeInputType(const std::string& file_path_string) {
    auto file_path = std::filesystem::path(file_path_string);
    std::error_code error_code;
    if (!std::filesystem::exists(file_path, error_code))            return InputType::NOT_EXISTS;
    if (std::filesystem::is_regular_file(file_path, error_code)) {
        if (isVideoFile(file_path_string))                          return InputType::VIDEO_FILE;
        if (isImageFile(file_path_string))                          return InputType::IMAGE_FILE;
        else                                                        return InputType::UNKNOWN;
    }
    if (std::filesystem::is_directory(file_path, error_code))       return InputType::DIRECTORY_IMAGE;
    else                                                            return InputType::UNKNOWN;
}


} // namespace vision
 