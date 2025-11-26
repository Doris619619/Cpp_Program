#pragma once
#include <string>
#include <functional>
#include <vector>
#include <opencv2/opencv.hpp>

namespace vision {

// 轻量帧提取器：
// - iterate: 迭代视频帧并回调，不落盘；可按 N fps 采样
// - extractToDir: 将视频帧批量导出为 jpg 文件
class FrameExtractor {
public:
    /* 
    *  遍历视频，按 sampleFps 采样 (<=0 表示全帧), 回调返回 false 可提前停止
    *  args:
    *  - videoPath:           视频路径
    *  - onFrame:             回调函数，参数为 (int frameIdx, const cv::Mat& bgr, double t_sec)
    *  - sampleFps:           采样帧率 (<=0 表示全帧)
    *  - startFrame/endFrame: 帧区间（包含），endFrame<0 表示到视频结束
    *  - stream_video:        是否以流式方式处理视频 (逐帧读取处理, 不落盘抽帧目录)
    */
    static bool streamProcess(
        const std::string& videoPath,
        const std::function<bool(int /*frameIdx*/, const cv::Mat& /*bgr*/, double /*t_sec*/)> &onFrame,
        //bool onFrame,
        double sampleFps = 0.0,
        int startFrame = 0,
        int endFrame = -1
        //bool stream_video = true // 
    );

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
