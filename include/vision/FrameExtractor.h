#pragma once
#include <string>
#include <functional>
#include <vector>
#include <opencv2/opencv.hpp>

namespace vision {

// 轻量帧提取器：
// - iterate: 迭代视频帧并回调，不落盘；可采样到目标 FPS
// - extractToDir: 将视频帧批量导出为 jpg 文件
class FrameExtractor {
public:
    // 遍历视频，按 sampleFps 采样（<=0 表示全帧），回调返回 false 可提前停止
    // startFrame/endFrame: 帧区间（包含），endFrame<0 表示到视频结束
    static bool iterate(
        const std::string& videoPath,
        const std::function<bool(int /*frameIdx*/, const cv::Mat& /*bgr*/, double /*t_sec*/)> &onFrame,
        double sampleFps = 0.0,
        int startFrame = 0,
        int endFrame = -1);

    // 将视频帧导出为目录内的一系列 JPG 图片，返回成功导出的数量
    static size_t extractToDir(
        const std::string& videoPath,
        const std::string& outDir,
        double outFps = 0.0,
        int jpegQuality = 95,
        int startFrame = 0,
        int endFrame = -1,
        const std::string& filenamePrefix = "frame_");
};

} // namespace vision
