#pragma once
#include <string>
#include <functional>
#include <vector>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <set>
#include <opencv2/opencv.hpp>
#include "vision/VisionA.h"
#include "vision/Publish.h"
#include "vision/Config.h"
#include "vision/Types.h"

namespace vision {

// 轻量帧提取器：
// - iterate: 迭代视频帧并回调，不落盘；可按 N fps 采样
// - extractToDir: 将视频帧批量导出为 jpg 文件
class FrameProcessor {
public:

    explicit FrameProcessor() = default;
    ~FrameProcessor() = default;

    // ==================== Core: Processing ===========================

    /* onFrame 对每帧照片处理   

    @param frame_index           current frame index  
    @param bgr:                  current frame image in BGR format
    @param t_sec:                current frame timestamp in seconds
    @param now_ms:               current system time in milliseconds
    @param input_path:           path of the input image or video
    @param annotated_frames_dir: directory for saving annotated frames
    @param ofs:                  output file stream for recording annotated frames
    @param vision:               instance of VisionA for processing frames
    @param latest_frame_file:    path to the file storing the latest frame information
    @param processed:            reference to a counter for processed frames

    @note Logic
    @note - process frame by vision.processFrame() and receive the state
    @note - based on the state, output the content in the cli
    @note - DO NOT conduct visualization here, hand it over to the other method
    @note - record annotated frame and output in the jsonl file
    @note - DO NOT responsible for judging whether to save-in-disk, return bool for handled or not
    */
    static bool onFrame(
        int, // frame_index
        const cv::Mat&, // bgr 
        double /*t_sec*/, 
        int64_t, // now_ms
        const std::filesystem::path&, // input_path (img path / video file)
        const std::string&, // cfg.annotated_frames_dir
        std::ofstream&, // ofs
        VisionA&, // vision
        const std::string&, // latest_frame_file
        size_t& // processed
    );

    /*  @brief streamProcess 流式处理视频帧   
    *  
    *  参考 sample_fps 边抽帧边处理，不入库
    * 
    *  @param videoPath:           视频路径
    *  @param latest_frame_dir:    最新帧文件目录 
    *  @param sampleFps:           采样帧率 (<=0 表示全帧)
    *  @param startFrame/endFrame: 帧区间（包含），endFrame<0 表示到视频结束
    *  @param vision:              VisionA实例
    *  @param cfg:                 VisionConfig配置
    *  @param ofs:                 输出文件流
    *  @param max_process_frames:  最大处理帧数
    * 
    *  @return bool: 处理是否成功
    */
    static size_t streamProcess(
        const std::string& videoPath,
        const std::string& latest_frame_dir,
        VisionA& vision,
        const VisionConfig& cfg,
        std::ofstream& ofs,
        double sampleFps = 0.0,
        int startFrame = 0,
        int endFrame = -1,
        size_t max_process_frames = 500
    );

    /*  @brief bulkExtraction 批量提取视频帧  
    *  
    *  @param video_path:        视频文件路径
    *  @param out_dir:           输出目录
    *  @param sample_fps:        采样帧率
    *  @param start_frame:       起始帧索引
    *  @param end_frame:         结束帧索引(-1 implies till the end)
    *  @param jpeg_quality:      jpeg 图像质量
    *  @param filename_prefix:   输出文件名前缀
    * 
    *  @return number of frames extracted 提取的帧数
    */ 
    static size_t bulkExtraction(
        const std::string& video_path,
        double sample_fps, 
        const std::string& out_dir = "./data/frames/",
        int start_frame = 0,
        int end_frame = -1,
        int jpeg_quality = 95,
        const std::string& filename_prefix = "f_"
    );

    /*  @brief Bulk Processing Video  批量处理视频帧
    *
    *   @param video_path:         视频文件路径
    *   @param img_dir:            图像输出目录
    *   @param latest_frame_dir:   最新帧文件目录 
    *   @param sample_fps:         采样帧率
    *   @param start_frame:        起始帧索引
    *   @param end_frame:          结束帧索引
    *   @param cfg:                VisionConfig配置
    *   @param ofs:                输出文件流
    *   @param vision:             VisionA实例
    *   @param max_process_frames: 最大处理帧数
    *   @param jpeg_quality:       jpeg 图像质量
    *   @param filename_prefix:    输出文件名前缀
    *
    *   @return number of frames processed 处理的帧数
    */
    static size_t bulkProcess(
        const std::string& video_path,
        const std::string& latest_frame_dir,
        const VisionConfig& cfg,                      // used by onFrame
        std::ofstream& ofs,
        VisionA& vision,
        double sample_fps,                            // frames args
        int start_frame = 0,
        int end_frame = -1,
        const std::string& img_dir = "./data/frames/",
        size_t max_process_frames = 500,              // use user input --max
        int jpeg_quality = 95,
        const std::string& filename_prefix = "f_"
    );

    /* @brief Image Processing 批量图像处理方法
    *  批量图像处理: 遍历目录下的所有图像文件并通过回调处理
    * 
    *  @param image_path:            图像所在目录
    *  @param ofs:                   输出文件流
    *  @param cfg:                   VisionConfig配置
    *  @param vision:                VisionA实例
    *  @param latest_frame_file:     最新帧文件路径
    *  @param max_process_frames:    最大处理帧数
    *  @param sample_fp100:          采样频率 (每100帧采样数, default = 20)
    *  @param original_total_frames: 原始总帧数 (用于采样计算)
    *  
    *  @return  number of frames processed 处理的帧数
    */
    static size_t imageProcess(
        const std::string& image_path,
        const std::string& latest_frame_dir,
        std::ofstream& ofs,
        const VisionConfig& cfg,
        VisionA& vision,
        size_t max_process_frames = 500,
        int sample_fp100 = 20,
        int original_total_frames = 0
    );

    // ==================== Utils: Sampling, Counting, Checking and Mapping ===========================

    // count files in specific directory
    static size_t countFilesInDir(const std::string& dir_path);

    // count images files in specific directory (.jpg/.jpeg/.png)
    static size_t countImageFilesInDir(const std::string& dir_path);

    // get stepsize based on img cnt
    static int getStepsize(size_t image_count);

    // get stepsize based on img cnt and sample_fps100
    static int getStepsize(size_t image_count, int sample_fp100);

    // get extraction output directory (create if not exists) (for bulk extraction mainly)
    static std::string getExtractionOutDir(const std::string& out_dir);

    // judge if is video by extension
    static bool isVideoFile(const std::string& file_path);

    // judge if is image by extension
    static bool isImageFile(const std::string& file_path);

    // judge input type (directory, video file, image file, unknown)
    static InputType judgeInputType(const std::string& file_path_string);

//private:

};

} // namespace vision
