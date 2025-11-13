
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <set>


void saveScreenshot(const cv::Mat& frame, int frameCount, double currentTime, const std::string& outputDir) {
    std::filesystem::create_directories(outputDir);


    std::stringstream filename;
    filename << outputDir << "/screenshot_"
        << std::fixed << std::setprecision(1) << currentTime
        << "s_" << frameCount << ".jpg";


    if (cv::imwrite(filename.str(), frame)) {
        std::cout << "screenshots saved: " << filename.str() << std::endl;
    }
    else {
        std::cerr << "screenshot failed: " << filename.str() << std::endl;
    }
}

std::string formatTime(double seconds) {
    int hours = static_cast<int>(seconds) / 3600;
    int minutes = (static_cast<int>(seconds) % 3600) / 60;
    int secs = static_cast<int>(seconds) % 60;
    int milliseconds = static_cast<int>((seconds - static_cast<int>(seconds)) * 1000);

    std::stringstream ss;
    if (hours > 0) {
        ss << std::setw(2) << std::setfill('0') << hours << ":";
    }
    ss << std::setw(2) << std::setfill('0') << minutes << ":"
        << std::setw(2) << std::setfill('0') << secs << "."
        << std::setw(3) << std::setfill('0') << milliseconds;

    return ss.str();
}

void showHelp() {
    std::cout << "\n===控制命令 ===\n";
    std::cout << "q / ESC - 退出程序\n";
    std::cout << "空格 - 暂停/继续播放\n";
    std::cout << "l - 前进10帧\n";
    std::cout << "j - 后退10帧\n";
    std::cout << "i - 前进100帧\n";
    std::cout << "k - 后退100帧\n";
    std::cout << "s - 保存当前帧截图\n";
    std::cout << "g - 跳转到指定帧\n";
    std::cout << "t - 跳转到指定时间\n";
    std::cout << "a - 设置自动截图间隔\n";
    std::cout << "h - 显示此帮助信息\n";
    std::cout << "==========================\n\n";
}

int main() {
    std::string videoPath;
    std::cout << "视频文件路径 (或按回车使用摄像头): ";
    std::getline(std::cin, videoPath);

    cv::VideoCapture cap;
    bool isCameraMode = false;

    if (videoPath.empty()) {

        cap.open(0);
        std::cout << "摄像头" << std::endl;
        isCameraMode = true;
    }
    else {

        cap.open(videoPath);
        std::cout << "打开视频文件: " << videoPath << std::endl;
        isCameraMode = false;
    }

    if (!cap.isOpened()) {
        std::cerr << "无法打开视频" << std::endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double duration = totalFrames / fps;

    //摄像头
    if (isCameraMode) {
        fps = 30.0; 
        totalFrames = -1; 
        duration = -1;
    }

    std::cout << "视频信息: " << fps << ", 总帧数: " << totalFrames
        << ", 时长: " << formatTime(duration) << std::endl;

    cv::Mat frame;
    bool isPaused = false;
    int currentFrame = 0;
    double autoCaptureInterval = 10.0; // 默认10秒自动截图
    std::string outputDir = "screenshots";

    // 模式选择
    int mode = 1; 
    if (!isCameraMode) {
        std::cout << "\n请选择模式:\n";
        std::cout << "1. 自动截图模式 - 程序不播放，每隔" << autoCaptureInterval << "秒自动保存截图\n";
        std::cout << "2. 观看模式 - 播放控制\n";
        std::cout << "请输入选择 (1 或 2): ";

        std::string modeInput;
        std::getline(std::cin, modeInput);

        if (modeInput == "2") {
            mode = 2;
            std::cout << "观看模式" << std::endl;
        }
        else {
            mode = 1;
            std::cout << "自动截图模式" << std::endl;


            std::cout << "当前自动截图间隔: " << autoCaptureInterval << "秒\n";
            std::cout << "是否修改截图间隔? (y/n): ";
            std::string changeInterval;
            std::getline(std::cin, changeInterval);

            if (changeInterval == "y" || changeInterval == "Y") {
                std::cout << "请输入新的截图间隔 (秒): ";
                std::cin >> autoCaptureInterval;
                std::cin.ignore(); // 清除输入缓冲区
                std::cout << "自动截图间隔已设置为: " << autoCaptureInterval << "秒" << std::endl;
            }
        }
    }

    std::set<int> capturedMarkers;


    if (!isCameraMode && mode == 1) {
        std::cout << "\n开始自动截图处理..." << std::endl;
        std::cout << "将每隔 " << autoCaptureInterval << " 秒保存一张截图" << std::endl;

        // 遍历视频的每一帧
        for (currentFrame = 0; currentFrame < totalFrames; currentFrame++) {
            cap.set(cv::CAP_PROP_POS_FRAMES, currentFrame);
            cap >> frame;

            if (frame.empty()) {
                std::cout << "视频结束或无法读取帧" << std::endl;
                break;
            }

            double currentTime = static_cast<double>(currentFrame) / fps;


            int timeMarker = static_cast<int>(currentTime / autoCaptureInterval);


            if (capturedMarkers.find(timeMarker) == capturedMarkers.end() &&
                currentTime >= timeMarker * autoCaptureInterval) {

                saveScreenshot(frame, currentFrame, currentTime, outputDir);
                capturedMarkers.insert(timeMarker);
            }

            // 进度
            if (currentFrame % 500 == 0) {
                double currentTime = static_cast<double>(currentFrame) / fps;
                int progress = static_cast<int>((currentFrame * 100.0) / totalFrames);
                std::cout << "处理进度: " << currentFrame << "/" << totalFrames
                    << " 帧 (" << progress << "%) - 时间: "
                    << formatTime(currentTime) << "/" << formatTime(duration) << std::endl;
            }
        }

        std::cout << "自动截图处理完成，共保存 " << capturedMarkers.size() << " 张截图" << std::endl;
        cap.release();
        return 0;
    }


    showHelp();

    while (true) {
        if (!isPaused) {
            cap >> frame;
            currentFrame = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));

            if (frame.empty()) {
                std::cout << "视频结束或无法读取帧" << std::endl;
                break;
            }
        }


        double currentTime = static_cast<double>(currentFrame) / fps;


        std::string statusText = "frame: " + std::to_string(currentFrame);
        if (totalFrames > 0) {
            statusText += "/" + std::to_string(totalFrames);
        }

        statusText += " time: " + formatTime(currentTime);
        if (duration > 0) {
            statusText += "/" + formatTime(duration);
        }



        if (isCameraMode) {
            statusText += " [camera mode]";
        }
        else {
            statusText += " [watching mode]";
        }

        cv::putText(frame, statusText, cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("video player", frame);

        // 键盘，后期改成qt的选项？
        int key = cv::waitKey(isPaused ? 0 : 30) & 0xFF;

        switch (key) {
        case 'q': // 退出
        case 27:  // ESC
            goto end_loop;

        case ' ': // 空格键 - 暂停/继续
            isPaused = !isPaused;
            std::cout << (isPaused ? "播放暂停" : "继续播放") << std::endl;
            break;

        case 's': // 保存当前截图
            saveScreenshot(frame, currentFrame, currentTime, outputDir);
            break;

        case 'g': // 跳转
            if (!isCameraMode) {
                int targetFrame;
                std::cout << "请输入目标帧 (0-" << totalFrames << "): ";
                std::cin >> targetFrame;
                if (targetFrame >= 0 && targetFrame <= totalFrames) {
                    cap.set(cv::CAP_PROP_POS_FRAMES, targetFrame);
                    currentFrame = targetFrame;
                    std::cout << "跳转到帧: " << targetFrame << std::endl;
                }
                else {
                    std::cout << "无效帧" << std::endl;
                }
                std::cin.ignore(); 
            }
            else {
                std::cout << "摄像头模式不支持跳帧!" << std::endl;
            }
            break;

        case 't': 
            if (!isCameraMode && fps > 0) {
                double targetTime;
                std::cout << "请输入目标时间 (秒): ";
                std::cin >> targetTime;
                int targetFrame = static_cast<int>(targetTime * fps);
                if (targetFrame >= 0 && targetFrame <= totalFrames) {
                    cap.set(cv::CAP_PROP_POS_FRAMES, targetFrame);
                    currentFrame = targetFrame;
                    std::cout << "跳转到时间: " << formatTime(targetTime) << std::endl;
                }
                else {
                    std::cout << "无效的时间!" << std::endl;
                }
                std::cin.ignore();
            }
            else {
                std::cout << "摄像头模式不支持时间跳转!" << std::endl;
            }
            break;

        case 'a': // 设置自动截图间隔
            if (isCameraMode) {
                // 摄像头模式下可以修改间隔值，但不启用自动截图
                std::cout << "当前自动截图间隔: " << autoCaptureInterval << "秒\n";
                std::cout << "请输入新的截图间隔 (秒): ";
                std::cin >> autoCaptureInterval;
                std::cout << "自动截图间隔设置为: " << autoCaptureInterval << "秒" << std::endl;
                std::cin.ignore();
            }
            else {
                // 观看模式下不允许修改间隔
                std::cout << "观看模式下不能修改自动截图间隔!" << std::endl;
            }
            break;

        case 'j': //  后退10帧
            if (!isCameraMode) {
                currentFrame = std::max(0, currentFrame - 10);
                cap.set(cv::CAP_PROP_POS_FRAMES, currentFrame);
                cap >> frame;
            }
            break;

        case 'l': //  前进10帧
            if (!isCameraMode) {
                currentFrame = std::min(totalFrames, currentFrame + 10);
                cap.set(cv::CAP_PROP_POS_FRAMES, currentFrame);
                cap >> frame;
            }
            break;

        case 'i': //  前进100帧
            if (!isCameraMode) {
                currentFrame = std::min(totalFrames, currentFrame + 100);
                cap.set(cv::CAP_PROP_POS_FRAMES, currentFrame);
                cap >> frame;
            }
            break;

        case 'k': //  后退100帧
            if (!isCameraMode) {
                currentFrame = std::max(0, currentFrame - 100);
                cap.set(cv::CAP_PROP_POS_FRAMES, currentFrame);
                cap >> frame;
            }
            break;

        case 'h': // 显示帮助
            showHelp();
            break;
        }
    }

end_loop:
    cap.release();
    cv::destroyAllWindows();
    std::cout << "程序结束" << std::endl;
    return 0;
}