#include "vision/VisionA.h"
#include "vision/Publish.h"
#include "vision/Config.h"
#include "vision/Types.h"
#include "vision/FrameProcessor.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <algorithm>
#include <set>
#include <iostream>
#include <chrono>
#include <fstream>
#include <cstddef>
using namespace vision;

int main() {
    std::cout << "a_demo stub running.\n";

    //std::cout << "Loaded " << images.size() << " images. Controls: click=select, 1/2/3/4=set state, S=save+next, N=skip, ESC=quit." << std::endl;

    cv::namedWindow("annotator", cv::WINDOW_NORMAL);
    cv::resizeWindow("annotator", 1280, 720);

    cv::Mat img;
    img = cv::imread("data/frames/frames_v004/f_000000.jpg");
    cv::imshow("Demo Image", img);
    cv::waitKey(0);
    std::cout << "Image read successfully.\n";
}