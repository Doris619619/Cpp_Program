#include "vision/FrameExtractor.h"
#include <filesystem>
#include <iostream>
#include <iomanip>

namespace fs = std::filesystem;

namespace vision {

static double safe_fps(cv::VideoCapture& cap) {
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps < 1e-3 || std::isnan(fps)) return 0.0;
    else if (fps > 2.0) return 2.0; // cap the fps to 2.0 to avoid too dense sampling
    return fps;
}

bool FrameExtractor::iterate(
    const std::string& video_path,
    const std::function<bool(int, const cv::Mat&, double)> &onFrame,  // callback func., truncate sampling if return false
    double sample_fps,                                                // sampling_freq
    int start_frame,                                                  // first_frame index = 0
    int end_frame,                                                    // end_frame index = -1 (the ending frame)
    bool stream_video                                                 // whether stream the video
) {
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
    
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[FrameExtractor] Failed to open video: " << video_path << "\n";
        return false;
    }

    // derive frame range and set start accordingly
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (end_frame < 0 && totalFrames > 0) end_frame = totalFrames - 1;
    if (start_frame > 0) cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);

    const double fps = safe_fps(cap);
    const bool do_sample = (sample_fps > 0.0);
    const double sample_interval = do_sample ? (1.0 / sample_fps) : 0.0;// sampleing interval = 1 / f
    double next_sample_T = 0.0;                                         // next sampling time seconds thresh

    if (stream_video) { // streaming video: Extract and Process Frame-by-Frame from Video
        int idx = start_frame;
        for (;;) {
            cv::Mat bgr;
            if (!cap.read(bgr)) {  // EOF
                std::cerr << "[FrameExtractor] Reached end of video or read error at frame index " << idx << "\n";
                break; 
            }

            // derive timestamp ms and s (sec)
            double t_ms = cap.get(cv::CAP_PROP_POS_MSEC);
            double t_sec = (t_ms > 1e-6) ? (t_ms / 1000.0) : (fps > 0.0 ? (static_cast<double>(idx) / fps) : 0.0);

            bool take = !do_sample;   // if not do sampling, check every time
            if (!take) {              // if starts sampling, take = true iff. 
                if (t_sec + 1e-9 >= next_sample_T) {  // sample only if current time exceeds next_sample_T
                    take = true;
                    while (next_sample_T <= t_sec) next_sample_T += sample_interval; // skip accumulated thresh, avoid miss-sampling
                }
            }
            if (take) {
                if (!onFrame(idx, bgr, t_sec)) break;
            }
            if (end_frame >= 0 && idx >= end_frame) break;
            ++idx;
        }
    }

    //// y onFrame used
    int idx = start_frame;
    for (;;) {
        cv::Mat bgr;
        if (!cap.read(bgr)) {
            std::cerr << "[FrameExtractor] Reached end of video or read error at frame index " << idx << "\n";
            break; // EOF
        }
        double t_ms = cap.get(cv::CAP_PROP_POS_MSEC);
        double t_sec = (t_ms > 1e-6) ? (t_ms / 1000.0) : (fps > 0.0 ? (static_cast<double>(idx) / fps) : 0.0);

        bool take = !do_sample;
        if (!take) {
            if (t_sec + 1e-9 >= next_sample_T) {
                take = true;
                while (next_sample_T <= t_sec) next_sample_T += sample_interval;
            }
        }
        if (take) {
            if (!onFrame(idx, bgr, t_sec)) break;
        }
        if (end_frame >= 0 && idx >= end_frame) break;
        ++idx;
    }
    return true;
}

size_t FrameExtractor::extractToDir(
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
        std::cerr << "[FrameExtractor] Failed to create dir: " << out_dir << " : " << error_code.message() << "\n";
        return 0;
    }

    size_t saved = 0;
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 
                                std::clamp(jpeg_quality, 1, 100) };

    iterate(
        video_path,
        [&](int frame_idx, const cv::Mat& bgr, double /*t_sec*/) -> bool {
            std::ostringstream oss;
            oss << filename_prefix << std::setw(6) << std::setfill('0') << frame_idx << ".jpg";
            fs::path out_path = fs::path(out_dir) / oss.str();
            if (cv::imwrite(out_path.string(), bgr, params)) {
                ++saved;
            } else {
                std::cerr << "[FrameExtractor] imwrite failed: " << out_path.string() << "\n";
            }
            return true; // continue
        },
        extract_fps,
        start_frame,
        end_frame);

    return saved;
}

} // namespace vision

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
        std::cerr << "[FrameExtractor] fetchFramesBySampleIndices open failed: " << video_path << "\n";
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
            std::cerr << "[FrameExtractor] read failed at original frame " << target_frame << "\n";
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
        std::cerr << "[FrameExtractor] fetchFramesByOriginalIndices open failed: " << video_path << "\n";
        return false;
    }
    for (int idx : original_indices) {
        int target = idx < 0 ? 0 : idx;
        cap.set(cv::CAP_PROP_POS_FRAMES, target);
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "[FrameExtractor] read failed at original frame " << target << "\n";
        }
        out_frames.push_back(frame);
    }
    return true;
}

} // end extra namespace vision block
