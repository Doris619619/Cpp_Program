#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

// Simple interactive person bounding box annotation tool.
// Usage:
//   annotate_person <frames_dir> <output_root>
// Output structure:
//   <output_root>/images_manual/<frame>.jpg (copied original)
//   <output_root>/labels_manual/<frame>.txt (YOLO format: class x_center y_center w h normalized)
// Keys:
//   n : next frame (auto-save)
//   p : previous frame (auto-save)
//   s : save current
//   d : delete last box
//   c : clear all boxes
//   q/ESC : save & quit
//   Arrow keys: move selected box (pick by clicking inside before moving)
// Mouse:
//   Drag LMB on empty area: create new box
//   LMB press inside existing box then release at new position: move box (drag)
//   Right click inside box: select that box (highlight)
// Boxes persist to next frame for fast adjustment.

struct Box { int x1; int y1; int x2; int y2; };

static std::vector<std::filesystem::path> g_images;
static int g_index = 0;
static std::vector<Box> g_boxes; // current frame boxes (persist forward)
static int g_selected = -1;
static bool g_dragging_new = false;
static bool g_dragging_move = false;
static cv::Point g_drag_start;
static Box g_move_origin; // original position while moving
static std::string g_window = "annotate_person";
static std::filesystem::path g_out_images;
static std::filesystem::path g_out_labels;
static cv::Mat g_current;

static void ensureDir(const std::filesystem::path &p) {
    std::error_code ec; std::filesystem::create_directories(p, ec);
}

static void saveCurrent() {
    if(g_index < 0 || g_index >= (int)g_images.size()) return;
    auto src_path = g_images[g_index];
    auto stem = src_path.stem().string();
    // copy image
    auto dst_img = g_out_images / (stem + ".jpg");
    cv::imwrite(dst_img.string(), g_current);
    // write labels
    auto dst_lbl = g_out_labels / (stem + ".txt");
    std::ofstream ofs(dst_lbl);
    for(const auto &b: g_boxes) {
        int x1 = std::min(b.x1, b.x2); int x2 = std::max(b.x1, b.x2);
        int y1 = std::min(b.y1, b.y2); int y2 = std::max(b.y1, b.y2);
        float cx = (x1 + x2) / 2.f; float cy = (y1 + y2) / 2.f;
        float w = (x2 - x1) * 1.f; float h = (y2 - y1) * 1.f;
        float iw = g_current.cols; float ih = g_current.rows;
        ofs << 0 << ' ' << cx/iw << ' ' << cy/ih << ' ' << w/iw << ' ' << h/ih << '\n';
    }
    std::cout << "Saved " << dst_img << " and " << dst_lbl << " (" << g_boxes.size() << " boxes)\n";
}

static void draw() {
    cv::Mat vis = g_current.clone();
    for(size_t i=0;i<g_boxes.size();++i){
        const auto &b = g_boxes[i];
        cv::Scalar color = (int)i==g_selected? cv::Scalar(0,0,255): cv::Scalar(0,255,0);
        cv::rectangle(vis, cv::Point(b.x1,b.y1), cv::Point(b.x2,b.y2), color, 2);
        cv::putText(vis, std::to_string(i), cv::Point(std::min(b.x1,b.x2)+3,std::min(b.y1,b.y2)+15), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
    cv::putText(vis, (std::to_string(g_index+1)+"/"+std::to_string(g_images.size())), cv::Point(5,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,0), 2);
    cv::imshow(g_window, vis);
}

static int test(int x, int y) {
    for(size_t i=0;i<g_boxes.size();++i){
        int x1 = std::min(g_boxes[i].x1,g_boxes[i].x2);
        int x2 = std::max(g_boxes[i].x1,g_boxes[i].x2);
        int y1 = std::min(g_boxes[i].y1,g_boxes[i].y2);
        int y2 = std::max(g_boxes[i].y1,g_boxes[i].y2);
        if(x>=x1 && x<=x2 && y>=y1 && y<=y2) return (int)i;
    }
    return -1;
}

static void mouseCallback(int event, int x, int y, int flags, void*) {
    if(g_current.empty()) return;
    if(event == cv::EVENT_LBUTTONDOWN){
        int h = test(x,y);
        if(h>=0){
            g_selected = h;
            g_dragging_move = true;
            g_move_origin = g_boxes[h];
            g_drag_start = {x,y};
        }else{
            g_dragging_new = true;
            g_selected = -1;
            g_boxes.push_back({x,y,x,y});
        }
    } else if(event == cv::EVENT_MOUSEMOVE){
        if(g_dragging_new && !g_boxes.empty()){ g_boxes.back().x2 = x; g_boxes.back().y2 = y; }
        if(g_dragging_move && g_selected>=0){
            int dx = x - g_drag_start.x; int dy = y - g_drag_start.y;
            g_boxes[g_selected].x1 = g_move_origin.x1 + dx; g_boxes[g_selected].x2 = g_move_origin.x2 + dx;
            g_boxes[g_selected].y1 = g_move_origin.y1 + dy; g_boxes[g_selected].y2 = g_move_origin.y2 + dy;
        }
    } else if(event == cv::EVENT_LBUTTONUP){
        g_dragging_new = false; g_dragging_move = false;
    } else if(event == cv::EVENT_RBUTTONDOWN){
        int h = test(x,y); if(h>=0) g_selected = h; else g_selected = -1;
    }
    draw();
}

static void loadFrame(int idx){
    if(idx<0 || idx>=(int)g_images.size()) return;
    g_index = idx;
    g_current = cv::imread(g_images[g_index].string());
    if(g_current.empty()){
        std::cerr << "Failed to load: " << g_images[g_index] << "\n"; return; }
    draw();
}

int main(int argc, char** argv){
    if(argc < 3){
        std::cerr << "Usage: annotate_person <frames_dir> <output_root>\n";
        return 1;
    }
    std::filesystem::path frames_dir = argv[1];
    std::filesystem::path out_root = argv[2];
    if(!std::filesystem::exists(frames_dir)){
        std::cerr << "Frames dir not exist: " << frames_dir << '\n'; return 1; }
    // gather images (jpg/png)
    for(auto &p: std::filesystem::directory_iterator(frames_dir)){
        if(!p.is_regular_file()) continue;
        auto ext = p.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if(ext == ".jpg" || ext == ".jpeg" || ext == ".png"){
            g_images.push_back(p.path());
        }
    }
    std::sort(g_images.begin(), g_images.end());
    if(g_images.empty()){ std::cerr << "No images found in " << frames_dir << '\n'; return 1; }
    g_out_images = out_root / "images_manual";
    g_out_labels = out_root / "labels_manual";
    ensureDir(g_out_images); ensureDir(g_out_labels);

    cv::namedWindow(g_window, cv::WINDOW_NORMAL);
    cv::setMouseCallback(g_window, mouseCallback);
    loadFrame(0);

    while(true){
        int k = cv::waitKey(30);
        if(k == 'n'){ saveCurrent(); loadFrame(std::min(g_index+1,(int)g_images.size()-1)); }
        else if(k == 'p'){ saveCurrent(); loadFrame(std::max(g_index-1,0)); }
        else if(k == 's'){ saveCurrent(); }
        else if(k == 'd'){ if(!g_boxes.empty()){ g_boxes.pop_back(); draw(); } }
        else if(k == 'c'){ g_boxes.clear(); g_selected=-1; draw(); }
        else if(k == 27 || k=='q'){ saveCurrent(); break; }
        else if(k == 0x250000){ /* left arrow */ if(g_selected>=0){ g_boxes[g_selected].x1-=1; g_boxes[g_selected].x2-=1; draw(); } }
        else if(k == 0x270000){ /* right arrow */ if(g_selected>=0){ g_boxes[g_selected].x1+=1; g_boxes[g_selected].x2+=1; draw(); } }
        else if(k == 0x260000){ /* up arrow */ if(g_selected>=0){ g_boxes[g_selected].y1-=1; g_boxes[g_selected].y2-=1; draw(); } }
        else if(k == 0x280000){ /* down arrow */ if(g_selected>=0){ g_boxes[g_selected].y1+=1; g_boxes[g_selected].y2+=1; draw(); } }
    }
    std::cout << "Done.\n";
    return 0;
}
