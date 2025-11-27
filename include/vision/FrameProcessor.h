#pragma once
#include <string>
#include <functional>
#include <vector>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "vision/VisionA.h"
#include "vision/Publish.h"
#include "vision/Config.h"

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
    static bool FrameProcessor::onFrame(
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
    static size_t FrameProcessor::streamProcess(
        const std::string& videoPath,
        const std::string& latest_frame_dir,
        double sampleFps = 0.0,
        int startFrame = 0,
        int endFrame = -1,
        VisionA& vision,
        const VisionConfig& cfg,
        std::ofstream& ofs,
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
    static size_t FrameProcessor::bulkExtraction(
        const std::string& video_path,
        const std::string& out_dir = "./data/frames/",
        double sample_fps, 
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
    static size_t FrameProcessor::bulkProcess(
        const std::string& video_path,
        const std::string& img_dir = "./data/frames/",
        const std::string& latest_frame_dir,
        double sample_fps,                            // frames args
        int start_frame = 0,
        int end_frame = -1,
        const VisionConfig& cfg,                      // used by onFrame
        std::ofstream& ofs,
        VisionA& vision,
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
    static size_t FrameProcessor::imageProcess(
        const std::string& image_path,
        const std::string& latest_frame_dir,
        std::ofstream& ofs,
        const VisionConfig& cfg,
        VisionA& vision,
        size_t max_process_frames = 500,
        int sample_fp100 = 20,
        int original_total_frames = 0
    );

    // ==================== Utils: Sampling, Counting, and Mapping ===========================

    // count files in specific directory
    static size_t countFilesInDir(const std::string& dir_path);

    // count images files in specific directory (.jpg/.jpeg/.png)
    static size_t countImageFilesInDir(const std::string& dir_path);

    // get stepsize based on img cnt
    static int getStepsize(size_t image_count);

    // get stepsize based on img cnt and sample_fps100
    static int getStepsize(size_t image_count, int sample_fp100);

    // get extraction output directory (create if not exists) (for bulk extraction mainly)
    static std::string FrameProcessor::getExtractionOutDir(const std::string& out_dir);

    /* temp refs: map from seat index to seat id  

// 计算单个采样序号对应的原视频帧号
// sample_index: 第几个被保留的采样(从0开始)
// original_fps: 原视频实际 fps (若被 safe_fps() 截断为2, 回取逻辑应仍使用真实 fps, 可由调用侧传入)
// sample_fps:   采样频率 (<=0 表示未采样, 直接返回 sample_index)
static int mapSampleIndexToOriginalFrame(int sample_index, double original_fps, double sample_fps)

// 直接按原始帧号列表回取 (适用于已经返回的是原视频帧号而非采样序号)
bool fetchFramesByOriginalIndices(const std::string& video_path,
                                  const std::vector<int>& original_indices,
                                  std::vector<cv::Mat>& out_frames)

// 批量映射采样序号到原视频帧号
static std::vector<int> mapSampleIndicesToOriginalFrames(const std::vector<int>& sample_indices,
                                                         double original_fps,
                                                         double sample_fps)

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
                                std::vector<int>& out_original_indices)



    */

//private:


};

} // namespace vision
