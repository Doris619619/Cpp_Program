#include "vision/FrameExtractor.h"
#include <filesystem>
#include <iostream>
#include <iomanip>

namespace fs = std::filesystem;

namespace vision {

static double safe_fps(cv::VideoCapture& cap) {
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 1e-3 || std::isnan(fps)) return 0.0;
    return fps;
}

bool FrameExtractor::iterate(
    const std::string& videoPath,
    const std::function<bool(int, const cv::Mat&, double)> &onFrame,
    double sampleFps,
    int startFrame,
    int endFrame)
{
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[FrameExtractor] Failed to open video: " << videoPath << "\n";
        return false;
    }

    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (endFrame < 0 && totalFrames > 0) endFrame = totalFrames - 1;
    if (startFrame > 0) cap.set(cv::CAP_PROP_POS_FRAMES, startFrame);

    const double fps = safe_fps(cap);
    const bool doSample = (sampleFps > 0.0);
    const double sampleInterval = doSample ? (1.0 / sampleFps) : 0.0;
    double nextSampleT = 0.0; // seconds

    int idx = startFrame;
    for (;;) {
        cv::Mat bgr;
        if (!cap.read(bgr)) break; // EOF
        double t_ms = cap.get(cv::CAP_PROP_POS_MSEC);
        double t_sec = (t_ms > 1e-6) ? (t_ms / 1000.0) : (fps > 0.0 ? (static_cast<double>(idx) / fps) : 0.0);

        bool take = !doSample;
        if (!take) {
            if (t_sec + 1e-9 >= nextSampleT) {
                take = true;
                while (nextSampleT <= t_sec) nextSampleT += sampleInterval;
            }
        }
        if (take) {
            if (!onFrame(idx, bgr, t_sec)) break;
        }
        if (endFrame >= 0 && idx >= endFrame) break;
        ++idx;
    }
    return true;
}

size_t FrameExtractor::extractToDir(
    const std::string& videoPath,
    const std::string& outDir,
    double outFps,
    int jpegQuality,
    int startFrame,
    int endFrame,
    const std::string& filenamePrefix)
{
    std::error_code ec;
    fs::create_directories(outDir, ec);
    if (ec) {
        std::cerr << "[FrameExtractor] Failed to create dir: " << outDir << " : " << ec.message() << "\n";
        return 0;
    }

    size_t saved = 0;
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, std::clamp(jpegQuality, 1, 100) };

    iterate(
        videoPath,
        [&](int frameIdx, const cv::Mat& bgr, double /*t_sec*/) -> bool {
            std::ostringstream oss;
            oss << filenamePrefix << std::setw(6) << std::setfill('0') << frameIdx << ".jpg";
            fs::path outPath = fs::path(outDir) / oss.str();
            if (cv::imwrite(outPath.string(), bgr, params)) {
                ++saved;
            } else {
                std::cerr << "[FrameExtractor] imwrite failed: " << outPath.string() << "\n";
            }
            return true; // continue
        },
        outFps,
        startFrame,
        endFrame);

    return saved;
}

} // namespace vision
