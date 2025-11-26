#include "vision/FrameExtractor.h"
#include <filesystem>
#include <iostream>
#include <string>

// Usage:
//   frame_extract <video_path> [out_dir] [--fps N] [--quality Q] [--start-frame S] [--end-frame E] [--prefix P]
// Examples:
//   frame_extract data/video/sample.mp4 runtime/frames --fps 5 --quality 92 --prefix f_

static bool hasFlag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) if (flag == argv[i]) return true; return false;
}
static const char* getOpt(int argc, char** argv, const std::string& key, const char* defv=nullptr) {
    for (int i = 1; i + 1 < argc; ++i) if (key == argv[i]) return argv[i+1]; return defv;
}

int main(int argc, char** argv) {
    if (argc < 2 || hasFlag(argc, argv, "-h") || hasFlag(argc, argv, "--help")) {
        std::cout << "Usage: frame_extract <video_path> [out_dir] [--fps N] [--quality Q] [--start-frame S] [--end-frame E] [--prefix P]\n";
        return 0;
    }
    std::string videoPath = argv[1];
    std::string outDir = (argc >= 3 && argv[2][0] != '-') ? argv[2] : std::string();
    if (outDir.empty()) {
        std::filesystem::path p(videoPath);
        std::string stem = p.stem().string();
        outDir = (std::filesystem::path("data/frames") / stem).string();
    }

    double fps = 0.0; if (const char* v = getOpt(argc, argv, "--fps"))         fps = std::atof(v);
    int quality = 95; if (const char* v = getOpt(argc, argv, "--quality"))     quality = std::atoi(v);
    int sframe = 0;   if (const char* v = getOpt(argc, argv, "--start-frame")) sframe = std::atoi(v);
    int eframe = -1;  if (const char* v = getOpt(argc, argv, "--end-frame"))   eframe = std::atoi(v);
    std::string prefix = "frame_"; if (const char* v = getOpt(argc, argv, "--prefix")) prefix = v;

    std::cout << "Extracting frames...\n"
              << "  video : " << videoPath << "\n"
              << "  out   : " << outDir << "\n"
              << "  fps   : " << fps << (fps <= 0 ? " (all frames)" : "") << "\n"
              << "  qual  : " << quality << "\n"
              << "  range : [" << sframe << ", " << (eframe < 0 ? -1 : eframe) << "]\n";

    size_t n = vision::FrameProcessor::extractToDir(videoPath, outDir, fps, quality, sframe, eframe, prefix);
    std::cout << "Saved " << n << " frames to " << outDir << "\n";
    return 0;
}
