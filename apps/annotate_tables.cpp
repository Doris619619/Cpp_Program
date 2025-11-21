#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include "vision/SeatRoi.h"

using namespace vision;

static void draw_poly(cv::Mat& img, const std::vector<cv::Point>& poly, const cv::Scalar& color) {
    if (poly.size() < 2) return;
    for (size_t i = 0; i < poly.size(); ++i) {
        cv::line(img, poly[i], poly[(i+1)%poly.size()], color, 2);
        cv::circle(img, poly[i], 3, color, -1);
    }
}

int main(int argc, char** argv) {
    std::string img_path = argc > 1 ? argv[1]: "data/samples/annotate.jpg";
    std::string out_json = argc > 2 ? argv[2]: "config/seats.json";
    std::string layout   = argc > 3 ? argv[3]: "2x2"; // 2x2 默认四人桌

    cv::Mat img = cv::imread(img_path);
    if (img.empty()) { 
        std::cerr << "Failed to read: " << img_path << "\n"; 
        return 1; 
    }

    std::vector<std::vector<cv::Point>> tables; // polygons of tables
    std::vector<cv::Point> current;             // current polygon being drawn

    cv::namedWindow("annotate", cv::WINDOW_NORMAL);
    cv::setMouseCallback("annotate", [](int e, int x, int y, int, void* ud){
        auto cur = reinterpret_cast<std::vector<cv::Point>*>(ud);
        if (e == cv::EVENT_LBUTTONDOWN) cur->push_back({x,y});
    }, &current);

    std::cout << "Instructions:\n"
              << " - Left click: add point for current table polygon\n"
              << " - ENTER: finish current table polygon and add to list\n"
              << " - BACKSPACE: undo last point\n"
              << " - C: clear current polygon\n"
              << " - S: save seats.json (split each table by layout=" << layout << ") and exit\n";

    // drawing process
    while (true) {
        cv::Mat vis = img.clone();
        
        // NOTICE: table is blank at the beginning
        for (auto& t : tables) draw_poly(vis, t, {0,255,0});
        draw_poly(vis, current, {0,128,255});
        cv::imshow("annotate", vis);
        int k = cv::waitKey(30);
        if (k == 13 || k == 10) { // Enter
            if (current.size() >= 3) 
                tables.push_back(current);
            current.clear();
        } else if (k == 8 || k == 127) { // Backspace/Delete
            if (!current.empty()) 
                current.pop_back();
        } else if (k == 'c' || k == 'C') {
            current.clear();
        } else if (k == 's' || k == 'S') {
            std::vector<SeatROI> seats;
            int tid = 0;
            for (auto& t : tables) {
                auto rects = splitTablePolyToSeats(t, layout);
                for (size_t i = 0; i < rects.size(); ++i) {
                    SeatROI s; 
                    s.seat_id = (++tid); 
                    s.rect = rects[i]; 
                    seats.push_back(std::move(s));
                }
            }
            if (saveSeatsToJson(out_json, seats)) {
                std::cout << "Saved seats to " << out_json << " (" << seats.size() << " seats)\n";
            } else {
                std::cerr << "Failed to save seats to " << out_json << "\n";
            }
            break;
        } else if (k == 27) break; // ESC
    }
    return 0;
}
