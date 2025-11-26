    #pragma once
#include <string>
#include <functional>
#include <vector>
#include <opencv2/opencv.hpp>

namespace vision {

// 轻量帧提取器：
// - iterate: 迭代视频帧并回调，不落盘；可按 N fps 采样
// - extractToDir: 将视频帧批量导出为 jpg 文件
class FrameProcessor {
public:
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
        int, // frame_index
        const cv::Mat&, // bgr 
        double /*t_sec*/, 
        int64_t, // now_ms
        const std::filesystem::path&, // input_path
        const std::string&, // video_src_path
        std::ofstream&, // ofs
        const VisionConfig&, // cfg
        VisionA&, // vision
        const std::string&, // latest_frame_file
        size_t& // processed
    );

    /*  @brief streamProcess 流式处理视频帧
    *  
    *  参考 sample_fps 边抽帧边处理，不入库
    * 
    *  @param videoPath:           视频路径
    *  @param onFrame:             回调函数，参数为 (int frameIdx, const cv::Mat& bgr, double t_sec)
    *  @param sampleFps:           采样帧率 (<=0 表示全帧)
    *  @param startFrame/endFrame: 帧区间（包含），endFrame<0 表示到视频结束
    *  @param stream_video:        是否以流式方式处理视频 (逐帧读取处理, 不落盘抽帧目录)
    * 
    *  @return bool: 处理是否成功
    */
    static bool streamProcess(
        const std::string& videoPath,
        const std::function<bool(int /*frameIdx*/, const cv::Mat& /*bgr*/, double /*t_sec*/)> &onFrame,
        double sampleFps = 0.0,
        int startFrame = 0,
        int endFrame = -1
    );

    /*  @brief bulkExtraction 批量提取视频帧  
    *  
    *  @param video_path:        视频文件路径
    *  @param out_dir:           输出目录
    */ 
    size_t FrameProcessor::bulkExtraction(
        const std::string& video_path,
        const std::string& out_dir,
        double sample_fps, 
        int start_frame = 0,
        int end_frame = -1,
        int jpeg_quality,
        const std::string& filename_prefix
    );

    // Bulk Processing Video  批量处理视频帧
    size_t FrameProcessor::bulkProcess(
        const std::string& video_path,
        const std::string& out_dir,
        const std::function<bool(int, const cv::Mat&, double)>& onFrame_,
        double sample_fps,
        int start_frame,
        int end_frame,
        int jpeg_quality,
        const std::string& filename_prefix,
        size_t max_process_frames
    );

    /*static size_t imageProcess(
        const std::filesystem::path& imagePath,
        
    );*/

    // no a good one, fix or throw
    // 批量图像处理方法
    static size_t imageProcess(
        const std::string& image_dir,
        const std::function<bool(
            int, 
            const cv::Mat&, 
            double, 
            int64_t, 
            const std::filesystem::path&, 
            const std::string&, 
            std::ofstream&, 
            const VisionConfig&, 
            VisionA&, 
            const std::string&, 
            size_t&
        )>& onFrame_,
        const VisionConfig& cfg,
        VisionA& vision,
        std::ofstream& ofs,
        const std::string& latest_frame_file,
        size_t max_process_frames
    );

    // count files in specific directory
    static size_t countFilesInDir(const std::string& dir_path);

    // count images files in specific directory (.jpg/.jpeg/.png)
    static size_t countImageFilesInDir(const std::string& dir_path);

    // get stepsize based on img cnt
    static int getStepsize(size_t image_count);

    // get stepsize based on img cnt and sample_fps100
    static int getStepsize(size_t image_count, int sample_fp100, double original_fps);

    /*size_t streamProcess(const std::string& video_path,
                        const std::function<bool(int, const cv::Mat&, double)>& onFrame_,
                        /*bool static onFrame(
                                int frame_index, 
                                const cv::Mat& bgr, 
                                double /*t_sec*//*, 
                                int64_t now_ms,
                                const std::filesystem::path& input_path,
                                const std::string& video_src_path,
                                std::ofstream& ofs,
                                const VisionConfig& cfg,
                                VisionA& vision,
                                const std::string& latest_frame_file,
                                size_t& processed
                               )*/
                        /*double sample_fps,
                        int start_frame,
                        int end_frame);*/

    // 将视频帧导出为目录内的一系列 .jpg 图片，返回成功导出的数量
    static size_t extractToDir(
        const std::string& videoPath,
        const std::string& outDir,
        double outFps = 0.0,
        int jpegQuality = 95,
        int startFrame = 0,
        int endFrame = -1,
        const std::string& filenamePrefix = "frame_");

    // 提取单帧图像，返回 cv::Mat (失败时返回空 Mat)
    static cv::Mat extractFrame(
        const std::string& video_path,
        int target_frame_idx
    );


};

} // namespace vision
