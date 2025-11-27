#include "vision/FrameProcessor.h"
#include <filesystem>
#include <iostream>
#include <iomanip>
#include "vision/VisionA.h"
#include "vision/Publish.h"
#include "vision/Config.h"
//#include "vision/FrameProcessor.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
//#include <iostream>
#include <chrono>
#include <fstream>
#include <cstddef>

namespace fs = std::filesystem;

namespace vision {

    /* logic draft
        iterate will extract frames from video derived from the path;
        for the extraction mode, 
            if stream_video is true, 
            then read in frames and process them one-by-one without saving to disk, 
                only save several img to the disk if needed (save_sample_frame = true)
                (here the "several" implies that the number should not be larger than
                10 and should be less than 1% of the total amount of fraames; sve after
                all the frames processed, only save the frames s.t. conf >= 50/75/80%? 
                use a queue (maybe) to implement filtering the highest conf annotated 
                frames; saving directory is the same data/frames/frames_vNNN);
            else, save all extracted frames to the guided data/frames and later the 
                iterate will use onFrame to process all the saved frames.
            
            that is:
            if stream_video:
                for each frame in video:
                    extract frame
                    onFrame(frame)
                    
                    // saving logic
                    if save_sample_frame and meet save condition:
                        save frame to disk
                        maintain map s.t. frame_idx(name) -> conf and order descendingly
                        ending one will be discarded if new-in one better than it
                    else:
                        only save the last several frames to be processed (if nearly 
                        all the frames mismatch the cond.)
            else:
                extract all frames to disk data/frames/frames_vNNN
                for each frame in data/frames/frames_vNNN:
                    onFrame(frame)
                    
                    // saving logic
                    if save_sample_frame and meet save condition:
                        save frame to disk
                        maintain map s.t. frame_idx(name) -> conf and order descendingly
                        ending one will be discarded if new-in one better than it
                    else:
                        only save the last several frames to be processed (if nearly 
                        all the frames mismatch the cond.)

    */


static double safe_fps(cv::VideoCapture& cap) {
    double original_fps = cap.get(cv::CAP_PROP_FPS);  // video original fps
    if (original_fps < 1e-3 || std::isnan(original_fps)) return 0.0;
    else if (original_fps > 2.0)                         return 2.0; // cap the fps to 2.0 to avoid too dense sampling
    return original_fps;
}

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
bool static onFrame(
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
bool FrameProcessor::streamProcess(
    const std::string& video_path,
    const std::function<bool(int, const cv::Mat&, double)> &onFrame_, // callback func, implemented by user/demo with onFrame called
    double sample_fps,                                                // sampling_freq
    int start_frame,                                                  // first_frame index = 0
    int end_frame                                                     // end_frame index = -1 (the ending frame)
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
    int processed_cnt = 0;

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

    // streaming video: Extract and Process Frame-by-Frame from Video
    for (int idx = start_frame, sample_cnt = 0; idx < original_total_frames, sample_cnt < sample_cnt_ub; idx += sample_stepsize, sample_cnt++) {
        
        // skip-sampling
        cap.set(cv::CAP_PROP_POS_FRAMES, idx); // skip frames according to stepsize

        // read-in frame
        cv::Mat bgr;  // in loop, everytime call cap.read(bgr), it will automatically move to next frame
        if (!cap.read(bgr)) {  // EOF
            std::cerr << "[FrameProcessor] Reached end of video or read error at frame index " << idx << "\n";
            break; 
        }

        // derive current timestamp ms and s (sec)
        double t_ms = cap.get(cv::CAP_PROP_POS_MSEC);
        double t_sec = (t_ms > 1e-6) ? (t_ms / 1000.0) : (original_fps > 0.0 ? (static_cast<double>(idx) / original_fps) : 0.0);

        // process frame
        if (!onFrame_(idx, bgr, t_sec)) {  // process and check termination
            std::cerr << "[FrameProcessor] onFrame_ callback requested termination at frame index " << idx << "\n";
            break;
        }
        processed_cnt++;

        // ending check
        if (end_frame >= 0 && idx >= end_frame) break;

        // sample saving logic
        /* not urgent */
    }

    // final output
    std::cout << "[FrameProcessor] streamProcess completed: processed=" << processed_cnt << "\n                 "
              << "original total frames=" << original_total_frames 
              << ", original fps=" << std::fixed << std::setprecision(2) << original_fps << "\n";

    return true;

    /*  old logic, just omit it   
    //// y onFrame used
    int idx = start_frame;
    for (;;) {
        cv::Mat bgr;
        if (!cap.read(bgr)) {
            std::cerr << "[FrameProcessor] Reached end of video or read error at frame index " << idx << "\n";
            break; // EOF
        }
        double t_ms = cap.get(cv::CAP_PROP_POS_MSEC);
        double t_sec = (t_ms > 1e-6) ? (t_ms / 1000.0) : (oringinal_fps > 0.0 ? (static_cast<double>(idx) / oringinal_fps) : 0.0);

        bool take = !do_sample;
        if (!take) {
            if (t_sec + 1e-9 >= t_next_sample) {
                take = true;
                while (t_next_sample <= t_sec) t_next_sample += sample_stepsize;
            }
        }
        if (take) {
            if (!onFrame(idx, bgr, t_sec)) break;
        }
        if (end_frame >= 0 && idx >= end_frame) break;
        ++idx;
    }
    */
}

// Bulk Extracting Video Frames 批量提取视频帧为图像文件
size_t FrameProcessor::bulkExtraction(
    const std::string& video_path,
    const std::string& out_dir,
    double sample_fps,                   // sampling fps = extract_fps, which is the program input arg
    int start_frame = 0,
    int end_frame = -1,
    int jpeg_quality,
    const std::string& filename_prefix
) {
    // find or create output directory
    std::error_code error_code;
    if (!std::filesystem::exists(out_dir)) { // output directory not exists
        std::filesystem::create_directories(out_dir, error_code);
        if (error_code) {     // create directory failed
            std::cerr << "[FrameProcessor] bulkExtraction create dir failed: " << out_dir << " : " << error_code.message() << "\n";
            return 0;
        }
    }

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
        sample_stempsize = getStepsize(static_cast<size_t>(total_frames), (sample_fps / original_fps <= 1 ? static_cast<int>(100 * sample_fps / original_fps) : 20), original_fps);
        if (sample_stempsize < 1) sample_stempsize = 1;
    }

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
        fs::path out_path = fs::path(out_dir) / oss.str();
        if (!cv::imwrite(out_path.string(), bgr, params)) {
            std::cerr << "[FrameProcessor] bulkExtraction write failed: " << out_path.string() << " at frame index " << idx << "\n";
        } else {
            ++extracted_cnt;
        }
    }
    return extracted_cnt;
}

// Bulk Processing Video
size_t FrameProcessor::bulkProcess(
    const std::string& video_path,
    const std::string& img_dir,
    const std::string& lastest_frame_dir,
    double sample_fps,
    int start_frame,
    int end_frame,
    const VisionConfig& cfg,                      // used by onFrame
    std::ofstream& ofs,
    VisionA& vision,
    size_t max_process_frames = 500,              // use user input --max
    int jpeg_quality = 95,
    const std::string& filename_prefix = "f_"
    
) {
    // bulk extract frames (with sample)
    size_t extracted = bulkExtraction(video_path, img_dir, sample_fps, start_frame, end_frame, jpeg_quality, filename_prefix);
    if (extracted == 0) return 0;

    // set stepsize 
    int process_stepsize = 1;  // difer from extraction sampling stepsize
    /* since sampling has been conducted during extraction, 
       it's not urgent to add another sampling here.
       if still needed after consideration, add later.

       needed to mention that the stepsize here should be determined by 
       total img cnt and consider idx diff between each img, as they 
       have been sampled and thus idx name of imgs may not be consecutive,
       follow the rule found.
       
    */

    // process extracted frames
    size_t processed_cnt = 0;
    int consecutive_failures_cnt = 0;
    for (size_t idx = 0; idx < extracted; idx+=process_stepsize) {
        
        // check if exists before processing
        if (!std::filesystem::exists(std::filesystem::path(img_dir) / (filename_prefix + std::to_string(idx) + ".jpg"))) {  // img: prefix + 000000 + .jpg (idx at 000000)
            std::cerr << "[FrameProcessor] bulkProcess imread failed: " << (std::filesystem::path(img_dir) / (filename_prefix + std::to_string(idx) + ".jpg")).string() << "\n";
            ++consecutive_failures_cnt;
            if (consecutive_failures_cnt >= 3) {
                std::cerr << "[FrameProcessor] bulkProcess stopping due to 3 consecutive read failures.\n";
                break;
            }
            continue;
        }
        consecutive_failures_cnt = 0;

        // skip-sampling (detailed logic added later if needed)

        // process frame via imageProcess
        FrameProcessor::imageProcess(
            img_dir, 
            ofs, 
            cfg,
            vision,
            lastest_frame_dir,
            max_process_frames - processed_cnt
        );

        processed_cnt++;
    }
    return processed_cnt;
}

/* imageProcess   
* 批量图像处理: 遍历目录下的所有图像文件并通过回调处理
* 
*  @param image_dir:            图像所在目录
*  @param ofs:                  输出文件流
*  @param cfg:                  VisionConfig配置
*  @param vision:               VisionA实例
*  @param latest_frame_file:    最新帧文件路径
*  @param max_process_frames:   最大处理帧数
*  
*  @return  number of frames processed 处理的帧数
*/
size_t FrameProcessor::imageProcess(
    const std::string& image_dir,           // 
    std::ofstream& ofs,                     //
    const VisionConfig& cfg,                //
    VisionA& vision,                        //
    const std::string& latest_frame_file,   // 
    size_t max_process_frames               // 
) {
    // basic args
    size_t total_processed = 0;
    size_t total_errors = 0;
    int frame_index = 0;
    const std::string annotated_frames_dir = !cfg.annotated_frames_dir.empty() ? cfg.annotated_frames_dir : "data/annotated_frames"; // directory to save annotated frames
    
    // input path check
    auto input_path = std::filesystem::path(image_dir);
    if (!std::filesystem::is_directory(input_path)) {  // open directory failed
        std::cerr << "[FrameProcessor] imageProcess: not a directory: " << image_dir << "\n"
                  << "                 Hint: to process images, use directory to images as argument.\n"
                  << "                 e.g. --input /path/to/images/\n";
        return 0;
    }
    
    std::cout << "[FrameProcessor] Image directory mode. Iterating files...\n";
    
    // iteration on the imgs (y no sampling here? needed!!! )
    for (auto &entry : std::filesystem::directory_iterator(input_path)) {
        
        // basic checks
        if (!entry.is_regular_file()) continue;
        cv::Mat bgr = cv::imread(entry.path().string());
        if (bgr.empty()) continue;
        
        // process frame with exception handling
        try {
            // derive current system time in ms
            int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            // process frame via onFrame
            bool continue_process = FrameProcessor::onFrame(
                frame_index,
                bgr,
                0.0,  // t_sec
                now_ms,
                input_path,
                annotated_frames_dir,
                ofs,
                vision,
                latest_frame_file,
                total_processed
            );
            
            ++frame_index;
            ++total_processed;
            
            if (!continue_process || total_processed >= max_process_frames) {  // termination 
                std::cout << "[FrameProcessor] Stopping at frame " << frame_index << "\n"
                          << "                 onFrame reported: " << (continue_process ? ("frame " + std::to_string(frame_index) + " handled, max process amount reached.") 
                                                                                        : "truncation requested at frame " + std::to_string(frame_index) + ".") << "\n";
                break;
            }
            
        } catch (const std::exception &exception) { // exception handling
            ++total_errors;
            std::cerr << "[FrameProcessor] Frame error: " << exception.what() << " src=" << input_path.string() << "\n";
        } catch (...) {    // unknown exception
            ++total_errors;
            std::cerr << "[FrameProcessor] Frame error: unknown src=" << input_path.string() << "\n";
        }
    }
    
    std::cout << "[FrameProcessor] imageProcess completed: processed=" << total_processed << ", errors=" << total_errors << "\n";
    return total_processed;
} 

// count files in specific directory
static size_t countFilesInDir(const std::string& dir_path) {
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
static size_t countImageFilesInDir(const std::string& dir_path) {
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
static int getStepsize(size_t image_count) {
    if (image_count <= 500) return 5;
    if (image_count <= 1000) return 10;
    return 50;
}

// get stepsize based on img cnt and sample_fps100
static int getStepsize(size_t image_count, int sample_fp100, double original_fps) {
    if (sample_fp100 <= 0) return getStepsize(image_count);
    //if (samplle_fp100 > original_fps)
    if (sample_fp100 > 100) sample_fp100 = 20; // 超过100则按安全默认20 fp100
    int step = static_cast<int>(std::floor(100 / sample_fp100)) + 1;
    return std::max(step, 1);
}

/* draft
bulk extraction: 

imgprocessing: if come from video, no need to extract again.
*/

// ============== to be discarded =============

size_t FrameProcessor::extractToDir(
    const std::string& video_path,
    const std::string& out_dir,
    double extract_fps,
    int jpeg_quality,
    int start_frame,
    int end_frame,
    const std::string& filename_prefix)
{
    std::error_code error_code;
    fs::create_directories(out_dir, error_code);
    if (error_code) {
        std::cerr << "[FrameProcessor] Failed to create dir: " << out_dir << " : " << error_code.message() << "\n";
        return 0;
    }

    size_t saved = 0;
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 
                                std::clamp(jpeg_quality, 1, 100) };

    streamProcess(
        video_path,
        [&](int frame_idx, const cv::Mat& bgr, double /*t_sec*/) -> bool {
            std::ostringstream oss;
            oss << filename_prefix << std::setw(6) << std::setfill('0') << frame_idx << ".jpg";
            fs::path out_path = fs::path(out_dir) / oss.str();
            if (cv::imwrite(out_path.string(), bgr, params)) {
                ++saved;
            } else {
                std::cerr << "[FrameProcessor] imwrite failed: " << out_path.string() << "\n";
            }
            return true; // continue
        },
        extract_fps,
        start_frame,
        end_frame);

    return saved;
}

// extract frames and return single cv::Mat
cv::Mat FrameProcessor::extractFrame(
    const std::string& video_path,
    int target_frame_idx
) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {  // open failed
        std::cerr << "[FrameProcessor] extractFrame open failed: " << video_path << "\n";
        return cv::Mat();
    }
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (target_frame_idx < 0 || (total_frames > 0 && target_frame_idx >= total_frames)) {  // invalid target index
        std::cerr << "[FrameProcessor] extractFrame invalid target frame index: " << target_frame_idx << "\n";
        return cv::Mat();
    }
    
    // seek to target frame
    cap.set(cv::CAP_PROP_POS_FRAMES, target_frame_idx);
    cv::Mat frame;
    if (!cap.read(frame)) {  // read failed
        std::cerr << "[FrameProcessor] extractFrame read failed at frame index " << target_frame_idx << "\n";
        return cv::Mat();
    }
    return frame;
}

// what's t_sec and now_ms? diff? 



} // namespace vision

// =============================== 新增: 批量提取与目录采样处理 ===============================
// 仅新增, 不修改/删除现有代码与注释。以下方法位于第一个命名空间内。

namespace vision {

// 统计目录下文件数量(可按需过滤图片扩展名)
static size_t countFilesInDir(const std::string& dir_path) {
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

// 统计目录下图片文件数量(.jpg/.jpeg/.png)
static size_t countImageFilesInDir(const std::string& dir_path) {
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

// 根据总图片数确定默认采样步长: <=500 => 5; <=1000 => 10; 否则 50
static int getStepsize(size_t image_count) {
    if (image_count <= 500) return 5;
    if (image_count <= 1000) return 10;
    return 50;
}

// 将 sample_fp100 (每100张取多少) 转为步长: step = floor(100 / sample_fp100), 最小为1
static int step_from_sample_fp100(int sample_fp100) {
    if (sample_fp100 <= 0) return 1;
    if (sample_fp100 > 100) sample_fp100 = 20; // 超过100则按安全默认20 fp100
    int step = static_cast<int>(std::floor(100 / sample_fp100));
    return step < 1 ? 1 : step;
}

// 批量提取: 从视频按 fps 抽帧写入目录; 读取失败累计3次则中止
size_t FrameProcessor::bulkExtraction(
    const std::string& video_path,
    const std::string& out_dir,
    double extract_fps,
    int start_frame,
    int end_frame,
    int jpeg_quality,
    const std::string& filename_prefix
) {
    std::error_code error_code;
    std::filesystem::create_directories(out_dir, error_code);
    if (error_code) {
        std::cerr << "[FrameProcessor] bulkExtraction create dir failed: " << out_dir << " : " << error_code.message() << "\n";
        return 0;
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[FrameProcessor] bulkExtraction open failed: " << video_path << "\n";
        return 0;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double original_fps = cap.get(cv::CAP_PROP_FPS);
    if (end_frame < 0 && total_frames > 0) end_frame = total_frames - 1;
    if (start_frame < 0) start_frame = 0;
    if (end_frame >= 0 && end_frame < start_frame) end_frame = start_frame;

    int sample_stepsize = 1;
    if (extract_fps > 0.0 && original_fps > 0.0) {
        sample_stepsize = static_cast<int>(std::floor(original_fps / extract_fps));
        if (sample_stepsize < 1) sample_stepsize = 1;
    }

    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, std::clamp(jpeg_quality, 1, 100) };

    size_t saved = 0;
    int consecutive_failures = 0;
    for (int idx = start_frame; end_frame < 0 ? true : (idx <= end_frame); idx += sample_stepsize) {
        cap.set(cv::CAP_PROP_POS_FRAMES, idx);
        cv::Mat bgr;
        if (!cap.read(bgr) || bgr.empty()) {
            std::cerr << "[FrameProcessor] bulkExtraction read failed at frame index " << idx << "\n";
            ++consecutive_failures;
            if (consecutive_failures >= 3) {
                std::cerr << "[FrameProcessor] bulkExtraction stopping due to 3 consecutive read failures.\n";
                break;
            }
            continue;
        }
        consecutive_failures = 0;

        std::ostringstream oss;
        oss << filename_prefix << std::setw(6) << std::setfill('0') << idx << ".jpg";
        fs::path out_path = fs::path(out_dir) / oss.str();
        if (!cv::imwrite(out_path.string(), bgr, params)) {
            std::cerr << "[FrameProcessor] bulkExtraction imwrite failed: " << out_path.string() << "\n";
        } else {
            ++saved;
        }
    }
    return saved;
}

// 批量处理: 先抽帧, 再按规则对目录内帧进行采样处理
size_t FrameProcessor::bulkProcess(
    const std::string& video_path,
    const std::string& out_dir,
    double extract_fps,
    int start_frame,
    int end_frame,
    int jpeg_quality,
    const std::string& filename_prefix,
    int sample_fp100,
    const std::function<bool(int, const cv::Mat&, double)>& onFrame_simple,
    size_t max_process_frames
) {
    size_t saved = bulkExtraction(video_path, out_dir, extract_fps, start_frame, end_frame, jpeg_quality, filename_prefix);
    if (saved == 0) return 0;

    // 计算目录内图片数量
    size_t image_count = countImageFilesInDir(out_dir);

    // 目录采样步长
    int step = 1;
    if (sample_fp100 > 0) step = step_from_sample_fp100(sample_fp100);
    else step = getStepsize(image_count);

    // 构建文件列表(排序保证顺序处理)
    std::vector<fs::path> files;
    files.reserve(image_count);
    for (auto &e : fs::directory_iterator(out_dir)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());

    size_t processed = 0;
    int consecutive_read_failures = 0;
    for (size_t i = 0; i < files.size(); i += static_cast<size_t>(step)) {
        if (max_process_frames > 0 && processed >= max_process_frames) break;
        cv::Mat bgr = cv::imread(files[i].string());
        if (bgr.empty()) {
            std::cerr << "[FrameProcessor] bulkProcess imread failed: " << files[i].string() << "\n";
            ++consecutive_read_failures;
            if (consecutive_read_failures >= 3) {
                std::cerr << "[FrameProcessor] bulkProcess stopping due to 3 consecutive read failures.\n";
                break;
            }
            continue;
        }
        consecutive_read_failures = 0;

        if (onFrame_simple) {
            // 这里 t_sec 用 0.0, 如需精准时间可在文件名或外部映射中提供
            if (!onFrame_simple(static_cast<int>(i), bgr, 0.0)) break;
        }
        ++processed;
    }
    return processed;
}

// 目录图像处理(仅目录, 可指定 sample_fp100); 若 sample_fp100<=0 则按数量规则自动取步长
size_t FrameProcessor::imageProcess(
    const std::string& image_dir,
    int sample_fp100,
    const std::function<bool(int, const cv::Mat&, double)>& onFrame_simple,
    size_t max_process_frames
) {
    size_t image_count = countImageFilesInDir(image_dir);
    int step = (sample_fp100 > 0) ? step_from_sample_fp100(sample_fp100)
                                  : getStepsize(image_count);

    std::vector<fs::path> files;
    files.reserve(image_count);
    std::error_code ec;
    for (auto &e : fs::directory_iterator(image_dir, ec)) {
        if (ec) break;
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());

    size_t processed = 0;
    int consecutive_read_failures = 0;
    for (size_t i = 0; i < files.size(); i += static_cast<size_t>(step)) {
        if (max_process_frames > 0 && processed >= max_process_frames) break;
        cv::Mat bgr = cv::imread(files[i].string());
        if (bgr.empty()) {
            std::cerr << "[FrameProcessor] imageProcess imread failed: " << files[i].string() << "\n";
            ++consecutive_read_failures;
            if (consecutive_read_failures >= 3) {
                std::cerr << "[FrameProcessor] imageProcess stopping due to 3 consecutive read failures.\n";
                break;
            }
            continue;
        }
        consecutive_read_failures = 0;

        if (onFrame_simple) {
            if (!onFrame_simple(static_cast<int>(i), bgr, 0.0)) break;
        }
        ++processed;
    }
    return processed;
}

} // namespace vision (added methods)

// =============================== 新增: 采样映射与按需回取功能 ===============================
// 说明: 为满足后续 B 返回关键采样序号后, 从原视频中重新获取对应原始帧用于延迟绘制的需求。
// 仅新增, 不改动/删除现有用户代码与注释。

namespace vision {

// 计算单个采样序号对应的原视频帧号
// sample_index: 第几个被保留的采样(从0开始)
// original_fps: 原视频实际 fps (若被 safe_fps() 截断为2, 回取逻辑应仍使用真实 fps, 可由调用侧传入)
// sample_fps:   采样频率 (<=0 表示未采样, 直接返回 sample_index)
static int mapSampleIndexToOriginalFrame(int sample_index, double original_fps, double sample_fps) {
    if (sample_index < 0) return 0;
    if (sample_fps <= 0.0 || original_fps <= 0.0) return sample_index; // 全帧或无有效fps
    // 目标时间点 = sample_index / sample_fps 秒; 原帧号 ≈ time * original_fps
    double target_time_sec = static_cast<double>(sample_index) / sample_fps;
    int original_frame = static_cast<int>(std::llround(target_time_sec * original_fps));
    if (original_frame < 0) original_frame = 0;
    return original_frame;
}

// 批量映射采样序号到原视频帧号
static std::vector<int> mapSampleIndicesToOriginalFrames(const std::vector<int>& sample_indices,
                                                         double original_fps,
                                                         double sample_fps) {
    std::vector<int> original_frames;
    original_frames.reserve(sample_indices.size());
    for (int si : sample_indices) {
        original_frames.push_back(mapSampleIndexToOriginalFrame(si, original_fps, sample_fps));
    }
    return original_frames;
}

// 根据“采样阶段的采样序号”列表回取对应原视频帧; 若采样频率为0(表示未采样, 全帧), 则采样序号即原始帧号
// video_path:   原视频
// sample_fps:   当初采样使用的频率
// original_fps: 原视频真实 fps (不使用 safe_fps 截断的值); 若 <=0 则从视频重新获取
// sample_indices: 需要回取的采样序号列表(非原始帧号)
// out_frames:   输出的图像帧 (按 sample_indices 顺序)
// out_original_indices: 输出对应的原始帧号 (与 out_frames 一一对应)
bool fetchFramesBySampleIndices(const std::string& video_path,
                                double sample_fps,
                                double original_fps,
                                const std::vector<int>& sample_indices,
                                std::vector<cv::Mat>& out_frames,
                                std::vector<int>& out_original_indices) {
    out_frames.clear();
    out_original_indices.clear();
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[FrameProcessor] fetchFramesBySampleIndices open failed: " << video_path << "\n";
        return false;
    }
    if (original_fps <= 0.0) {
        double raw_fps = cap.get(cv::CAP_PROP_FPS);
        if (raw_fps > 0.0 && !std::isnan(raw_fps)) original_fps = raw_fps; else original_fps = 0.0;
    }
    // 先映射到原始帧号
    std::vector<int> original_indices = mapSampleIndicesToOriginalFrames(sample_indices, original_fps, sample_fps);
    // 为提升顺序访问性能, 创建一个排序副本再按原顺序恢复
    std::vector<int> sorted_indices = original_indices;
    std::vector<int> order_map(sorted_indices.size());
    for (size_t i = 0; i < sorted_indices.size(); ++i) order_map[i] = static_cast<int>(i);
    std::stable_sort(order_map.begin(), order_map.end(), [&](int a, int b){ return sorted_indices[a] < sorted_indices[b]; });
    // 逐个读取
    std::vector<cv::Mat> temp_frames(sorted_indices.size());
    for (int ord : order_map) {
        int target_frame = sorted_indices[ord];
        if (target_frame < 0) target_frame = 0;
        cap.set(cv::CAP_PROP_POS_FRAMES, target_frame);
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "[FrameProcessor] read failed at original frame " << target_frame << "\n";
            // 允许失败继续, 输出空帧占位
        }
        temp_frames[ord] = frame;
    }
    // 恢复用户请求顺序
    for (size_t i = 0; i < original_indices.size(); ++i) {
        out_frames.push_back(temp_frames[i]);
        out_original_indices.push_back(original_indices[i]);
    }
    return true;
}

// 直接按原始帧号列表回取 (适用于 B 已经返回的是原视频帧号而非采样序号)
bool fetchFramesByOriginalIndices(const std::string& video_path,
                                  const std::vector<int>& original_indices,
                                  std::vector<cv::Mat>& out_frames) {
    out_frames.clear();
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[FrameProcessor] fetchFramesByOriginalIndices open failed: " << video_path << "\n";
        return false;
    }
    for (int idx : original_indices) {
        int target = idx < 0 ? 0 : idx;
        cap.set(cv::CAP_PROP_POS_FRAMES, target);
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "[FrameProcessor] read failed at original frame " << target << "\n";
        }
        out_frames.push_back(frame);
    }
    return true;
}

} // end extra namespace vision block

// =============================== 新增: 单帧提取与包装流式/批量处理 ===============================
// 说明: 依据用户在文件中的笔记需求, 我们不修改既有代码与注释,
// 仅在此追加新的 helper 与包装方法, 命名与变量风格遵循用户约定。

namespace vision {

// 单帧提取 helper: 通过原视频帧号读取一帧并返回
// video_path: 原视频路径
// frame_index: 目标原始帧号 (>=0)
// out_bgr: 输出 BGR 图像
// 返回: 是否读取成功
bool extractSingleFrame(const std::string& video_path, int frame_index, cv::Mat& out_bgr) {
    out_bgr.release();
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[FrameProcessor] extractSingleFrame open failed: " << video_path << "\n";
        return false;
    }
    int target = frame_index < 0 ? 0 : frame_index;
    cap.set(cv::CAP_PROP_POS_FRAMES, target);
    cv::Mat bgr;
    if (!cap.read(bgr)) {
        std::cerr << "[FrameProcessor] extractSingleFrame read failed at frame " << target << "\n";
        return false;
    }
    out_bgr = bgr;
    return true;
}

// 流式处理包装: 使用单帧提取 helper, 然后交给 onFrame
// 注意: 不修改原阈值/采样设定; 只调用现有的采样频率变量约定 (由调用侧传入)
// 参数:
// - video_path: 原视频
// - onFrame: 与现有 iterate 相同签名的回调 (idx, bgr, t_sec)
// - sample_fps: 采样频率 (<=0 表示每帧)
// - start_frame, end_frame: 起止原始帧号, 与现有语义一致
// 返回: 处理的帧数
size_t streamProcess(const std::string& video_path,
                               //const std::function<bool(int, const cv::Mat&, double)>& onFrame,
                               bool static onFrame(
                                int frame_index, 
                                const cv::Mat& bgr, 
                                double /*t_sec*/, 
                                int64_t now_ms,
                                const std::filesystem::path& input_path,
                                const std::string& video_src_path,
                                std::ofstream& ofs,
                                const VisionConfig& cfg,
                                VisionA& vision,
                                const std::string& latest_frame_file,
                                size_t& processed)
                               double sample_fps,
                               int start_frame,
                               int end_frame) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[FrameProcessor] streamProcess open failed: " << video_path << "\n";
        return 0;
    }
    double raw_fps = cap.get(cv::CAP_PROP_FPS);
    bool do_sample = sample_fps > 0.0;
    double sample_interval = do_sample ? (1.0 / sample_fps) : 0.0;
    double next_sample_t = 0.0;

    int idx = start_frame < 0 ? 0 : start_frame;
    size_t processed = 0;
    for (;;) {
        if (end_frame >= 0 && idx > end_frame) break;
        // 单帧提取
        cv::Mat bgr;
        if (!extractSingleFrame(video_path, idx, bgr)) break;
        // 时间戳 (优先 POS_MSEC, 否则 idx/raw_fps)
        cap.set(cv::CAP_PROP_POS_FRAMES, idx);
        double t_ms = cap.get(cv::CAP_PROP_POS_MSEC);
        double t_sec = (!std::isnan(t_ms) && t_ms > 0.0) ? (t_ms / 1000.0)
                        : ((raw_fps > 0.0) ? (static_cast<double>(idx) / raw_fps) : 0.0);

        bool take = !do_sample;
        if (!take) {
            if (t_sec + 1e-9 >= next_sample_t) {
                take = true;
                while (next_sample_t <= t_sec) next_sample_t += sample_interval;
            }
        }
        if (take) {
            ++processed;
            if (!onFrame_(int frame_index, const cv::Mat& bgr, double /*t_sec*/, int64_t now_ms, 
                const std::filesystem::path& input_path, const std::string& video_src_path, std::ofstream& ofs,
                const VisionConfig& cfg, VisionA& vision, const std::string& latest_frame_file, size_t& processed)) 
                break;
        }
        ++idx;
    }
    return processed;
}

// 批量处理包装: 按采样频率批量选择原始帧号, 用单帧 helper 读取后再 onFrame
// 注意: 计算机内存/IO 限制, 此方法逐帧读取并立即回调, 不做大规模缓冲; 不修改原阈值
// 参数与返回与上面一致
size_t bulkExtractWithSampling(const std::string& video_path,
                               const std::function<bool(int, const cv::Mat&, double)>& onFrame,
                               double sample_fps,
                               int start_frame,
                               int end_frame) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[FrameProcessor] bulkExtractWithSampling open failed: " << video_path << "\n";
        return 0;
    }
    double raw_fps = cap.get(cv::CAP_PROP_FPS);
    bool do_sample = sample_fps > 0.0;
    double sample_interval = do_sample ? (1.0 / sample_fps) : 0.0;
    double next_sample_t = 0.0;

    int idx = start_frame < 0 ? 0 : start_frame;
    size_t processed = 0;
    for (;;) {
        if (end_frame >= 0 && idx > end_frame) break;
        // 单帧提取
        cv::Mat bgr;
        if (!extractSingleFrame(video_path, idx, bgr)) break;
        // 时间戳 (优先 POS_MSEC, 否则 idx/raw_fps)
        cap.set(cv::CAP_PROP_POS_FRAMES, idx);
        double t_ms = cap.get(cv::CAP_PROP_POS_MSEC);
        double t_sec = (!std::isnan(t_ms) && t_ms > 0.0) ? (t_ms / 1000.0)
                        : ((raw_fps > 0.0) ? (static_cast<double>(idx) / raw_fps) : 0.0);

        bool take = !do_sample;
        if (!take) {
            if (t_sec + 1e-9 >= next_sample_t) {
                take = true;
                while (next_sample_t <= t_sec) next_sample_t += sample_interval;
            }
        }
        if (take) {
            ++processed;
            if (!onFrame(idx, bgr, t_sec)) break;
        }
        ++idx;
    }
    return processed;
}

} // namespace vision
