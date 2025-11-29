#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "vision/SeatRoi.h"
#include "vision/Types.h"

/* 标注工具: annotate_seat_states
 * 用途: 为 data/frames 下的图片批量标注每个座位的占用状态 (PERSON / OBJECT_ONLY / FREE / UNKNOWN)
 * 使用座位表: config/poly_simple_seats.json (可传参覆盖)
 * 保存: 标注后的图片 + 座位状态 JSON (data/frames_train/<image_stem>.json)
 * 交互:
 *   - 左键点击座位多边形选择/高亮
 *   - 按键 1: FREE, 2: PERSON, 3: OBJECT_ONLY, 4: UNKNOWN (循环或直接赋值)
 *   - S: 保存当前图片标注并跳到下一张 (沿用上一张状态作为初值)
 *   - N: 跳过不保存直接下一张
 *   - ESC: 退出
 */

using namespace vision;

struct SeatAnnotState {
    int seat_id;
    std::vector<cv::Point> poly;
    SeatOccupancyState state = SeatOccupancyState::UNKNOWN;
};

static std::string toStringState(SeatOccupancyState st) {
    switch(st){
        case SeatOccupancyState::FREE: return "FREE";
        case SeatOccupancyState::PERSON: return "PERSON";
        case SeatOccupancyState::OBJECT_ONLY: return "OBJECT_ONLY";
        case SeatOccupancyState::UNKNOWN: default: return "UNKNOWN";
    }
}

static SeatOccupancyState fromKey(int key, SeatOccupancyState prev) {
    switch(key){
        case '1': return SeatOccupancyState::FREE;
        case '2': return SeatOccupancyState::PERSON;
        case '3': return SeatOccupancyState::OBJECT_ONLY;
        case '4': return SeatOccupancyState::UNKNOWN;
        default: return prev;
    }
}

static void saveAnnotation(const std::string& outJsonPath, const std::string& imagePath, const std::vector<SeatAnnotState>& states) {
    nlohmann::json root;
    root["image"] = imagePath;
    nlohmann::json arr = nlohmann::json::array();
    for (auto &s : states) {
        nlohmann::json obj;
        obj["seat_id"] = s.seat_id;
        nlohmann::json poly = nlohmann::json::array();
        for (auto &p : s.poly) poly.push_back({p.x, p.y});
        obj["poly"] = poly;
        obj["state"] = toStringState(s.state);
        arr.push_back(std::move(obj));
    }
    root["seats"] = std::move(arr);
    std::ofstream ofs(outJsonPath);
    if (ofs) ofs << root.dump(2);
}

static int hitTestPoly(const std::vector<cv::Point>& poly, cv::Point pt) {
    if (poly.size() < 3) return 0;
    double dist = cv::pointPolygonTest(poly, pt, false);
    return dist >= 0 ? 1 : 0;
}

int main(int argc, char** argv) {
    std::string framesDir = "data/frames";
    std::string seatsFile = "config/poly_simple_seats.json";
    std::string outDir = "data/frames_train";
    if (argc > 1) framesDir = argv[1];
    if (argc > 2) seatsFile = argv[2];
    if (argc > 3) outDir = argv[3];

    std::vector<SeatROI> seatDefs;
    if (!loadSeatsFromJson(seatsFile, seatDefs)) {
        std::cerr << "Failed to load seats file: " << seatsFile << "\n";
        return 1;
    }
    if (!std::filesystem::exists(framesDir)) {
        std::cerr << "Frames dir not found: " << framesDir << "\n";
        return 1;
    }
    std::filesystem::create_directories(outDir);

    // 初始化标注状态（全部 UNKNOWN）
    std::vector<SeatAnnotState> current;
    current.reserve(seatDefs.size());
    for (auto &s : seatDefs) {
        SeatAnnotState st; st.seat_id = s.seat_id; st.poly = s.poly; st.state = SeatOccupancyState::UNKNOWN; current.push_back(st);
    }
    int selectedIndex = -1;

    std::vector<std::filesystem::path> images;
    for (auto &e : std::filesystem::directory_iterator(framesDir)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        for (auto &c : ext) c = (char)std::tolower((unsigned char)c);
        if (ext == ".jpg" || ext == ".png" || ext == ".jpeg") images.push_back(e.path());
    }
    std::sort(images.begin(), images.end());
    if (images.empty()) {
        std::cerr << "No images found in " << framesDir << "\n";
        return 0;
    }

    std::cout << "Loaded " << images.size() << " images. Controls: click=select, 1/2/3/4=set state, S=save+next, N=skip, ESC=quit." << std::endl;

    cv::namedWindow("annotator", cv::WINDOW_NORMAL);
    cv::resizeWindow("annotator", 1280, 720);

    cv::Mat curImg;
    int imgIdx = 0;

    auto draw = [&](const cv::Mat& bgr){
        cv::Mat vis = bgr.clone();
        for (size_t i=0;i<current.size();++i) {
            auto &seat = current[i];
            cv::Scalar color;
            switch(seat.state){
                case SeatOccupancyState::PERSON: color = cv::Scalar(0,0,255); break;
                case SeatOccupancyState::OBJECT_ONLY: color = cv::Scalar(0,255,255); break;
                case SeatOccupancyState::FREE: color = cv::Scalar(0,255,0); break;
                default: color = cv::Scalar(200,200,200); break;
            }
            std::vector<std::vector<cv::Point>> contours = { seat.poly };
            cv::polylines(vis, contours, true, color, (int)(i==selectedIndex?4:2));
            // 中心文本
            cv::Moments m = cv::moments(seat.poly);
            if (m.m00 != 0) {
                cv::Point center((int)(m.m10/m.m00), (int)(m.m01/m.m00));
                std::string label = std::to_string(seat.seat_id) + ":" + toStringState(seat.state);
                cv::putText(vis, label, center, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            }
        }
        std::string info = images[imgIdx].filename().string() + "  [" + std::to_string(imgIdx+1) + "/" + std::to_string(images.size()) + "]";
        cv::putText(vis, info, {10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255),1);
        cv::imshow("annotator", vis);
    };

    bool need_red_raw = true;
    cv::setMouseCallback("annotator", [](int event, int x, int y, int, void* userdata){
        if (event != cv::EVENT_LBUTTONDOWN) return;
        auto pack = reinterpret_cast<std::pair<std::vector<SeatAnnotState>*, int*>*>(userdata);
        auto &arr = *pack->first; auto &sel = *pack->second;
        cv::Point pt(x,y);
        int found = -1;
        for (size_t i=0;i<arr.size();++i) {
            if (hitTestPoly(arr[i].poly, pt)) { found = (int)i; break; }
        }
        sel = found;
    }, new std::pair<std::vector<SeatAnnotState>*, int*>(&current, &selectedIndex));

    while (true) {
        if (need_red_raw) {
            curImg = cv::imread(images[imgIdx].string());
            if (curImg.empty()) { std::cerr << "Failed load: " << images[imgIdx] << "\n"; }
            need_red_raw = false;
        }
        draw(curImg);
        int k = cv::waitKey(30);
        if (k == 27) break; // ESC
        if (k == 's' || k == 'S') {
            // 保存标注
            std::string stem = images[imgIdx].stem().string();
            std::string outJson = (std::filesystem::path(outDir) / (stem + ".json")).string();
            std::string outImg  = (std::filesystem::path(outDir) / (stem + "_annotated.jpg")).string();
            saveAnnotation(outJson, images[imgIdx].string(), current);
            // 保存可视化图
            cv::Mat vis;
            draw(curImg); // 重绘到窗口
            vis = cv::imread(images[imgIdx].string()); // 原图重新读用于干净叠加
            if (!vis.empty()) {
                // 再绘一次得到标注图
                for (auto &seat : current) {
                    cv::Scalar color;
                    switch(seat.state){
                        case SeatOccupancyState::PERSON: color = cv::Scalar(0,0,255); break;
                        case SeatOccupancyState::OBJECT_ONLY: color = cv::Scalar(0,255,255); break;
                        case SeatOccupancyState::FREE: color = cv::Scalar(0,255,0); break;
                        default: color = cv::Scalar(200,200,200); break;
                    }
                    std::vector<std::vector<cv::Point>> contours = { seat.poly };
                    cv::polylines(vis, contours, true, color, 2);
                    cv::Moments m = cv::moments(seat.poly);
                    if (m.m00 != 0) {
                        cv::Point center((int)(m.m10/m.m00), (int)(m.m01/m.m00));
                        std::string label = std::to_string(seat.seat_id) + ":" + toStringState(seat.state);
                        cv::putText(vis, label, center, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                    }
                }
                cv::imwrite(outImg, vis);
            }
            std::cout << "Saved: " << outJson << "\n";
            // 下一张：沿用当前状态（需求中：下一张初值与上一张相同）
            imgIdx++;
            if (imgIdx >= (int)images.size()) {
                std::cout << "All images processed." << std::endl; break;
            }
            need_red_raw = true;
        } else if (k == 'n' || k == 'N') {
            imgIdx++;
            if (imgIdx >= (int)images.size()) { std::cout << "Done." << std::endl; break; }
            need_red_raw = true;
        } else if (k == '1' || k=='2' || k=='3' || k=='4') {
            if (selectedIndex >=0 && selectedIndex < (int)current.size()) {
                current[selectedIndex].state = fromKey(k, current[selectedIndex].state);
            }
        }
    }
    cv::destroyAllWindows();
    return 0;
}
