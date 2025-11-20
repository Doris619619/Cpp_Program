// seat_yolo.cpp
// YOLO(person) + baseline 空桌比较 -> 区分 PERSON / OBJECT / EMPTY
// 需要：OpenCV >=4.5（包含 dnn） + yolov5s.onnx（或 yolov8n.onnx） + baseline 空桌截图

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <sstream>
#include <deque>
#include <iostream>

using namespace cv;
using namespace dnn;
using namespace std;

struct Seat {
    int id;
    Rect rect;
    int state = 0;      // 0 empty, 1 person, 2 object
    int empty_cnt = 0;
    int person_cnt = 0;
    int object_cnt = 0;
};


const string videoPath = R"(D:\3002code\c++\3002Project1\3002Project1\video2.mp4)";
const string roiPath = R"(D:\3002code\c++\3002Project1\3002Project1\rois.csv)";
const string outPath = R"(D:\3002code\c++\3002Project1\3002Project1\result_yolo_obj.mp4)";
const string yoloModelPath = R"(D:\3002code\c++\3002Project1\3002Project1\yolov5s.onnx)";
// baseline 空桌截图
const string baselinePath = R"(D:\3002code\c++\3002Project1\3002Project1\screenshots\screenshot_6.5s_195.jpg)";


// YOLO / 参数（
const float YOLO_CONF_THRESH = 0.25f;
const float YOLO_NMS_THRESH = 0.45f;
const int   YOLO_INPUT_SIZE = 640;
const int   YOLO_EVERY_N_FRAMES = 15;

// 状态判定窗口（抗抖动）
const int PERSON_CONFIRM = 2;   // 连续N帧检测到person才确认
const int OBJECT_CONFIRM = 4;   // 连续N帧检测到object才确认
const int EMPTY_CONFIRM = 18;  // 连续N帧都空才确认为空（防止静止被误判）

// baseline 比较阈值
const int DIFF_PIXEL_THRESH = 30;  // 像素灰度差超过此值计为变化像素
const double DIFF_RATIO_THRESH = 0.02; // 如果变化像素占 ROI 面积 > 2% 判为 有物品

// MOG2 备用阈值
const double FG_RATIO = 0.05;


vector<Seat> loadROIs(const string& path) {
    vector<Seat> seats;
    ifstream fin(path);
    if (!fin.is_open()) {
        cerr << "无法打开 ROI 文件: " << path << endl;
        return seats;
    }
    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        int id, x, y, w, h; char comma;
        ss >> id >> comma >> x >> comma >> y >> comma >> w >> comma >> h;
        seats.push_back({ id, Rect(x, y, w, h) });
    }
    return seats;
}

// letterbox 缩放（保持长宽比并 pad）
// 返回 pad 后的图，scale/dx/dy 会被设置用于把坐标从 padded 映射回原图
static Mat letterbox(const Mat& src, int target_size, float& scale, int& dx, int& dy) {
    int w = src.cols, h = src.rows;
    float r = min((float)target_size / (float)w, (float)target_size / (float)h);
    int nw = int(round(w * r)), nh = int(round(h * r));
    dx = (target_size - nw) / 2;
    dy = (target_size - nh) / 2;
    Mat resized; resize(src, resized, Size(nw, nh));
    Mat canvas = Mat::zeros(target_size, target_size, src.type());
    resized.copyTo(canvas(Rect(dx, dy, resized.cols, resized.rows)));
    scale = r;
    return canvas;
}

// YOLO 推理（使用 letterbox 映射回原图）
// 输出 boxes/conf/classId
void yoloDetect(Net& net, const Mat& frame, vector<Rect>& outBoxes, vector<float>& outConf, vector<int>& outClassIds) {
    outBoxes.clear(); outConf.clear(); outClassIds.clear();
    float scale; int dx, dy;
    Mat inputImg = letterbox(frame, YOLO_INPUT_SIZE, scale, dx, dy);

    Mat blob = blobFromImage(inputImg, 1.0 / 255.0, Size(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), Scalar(), true, false);
    net.setInput(blob);
    vector<Mat> outputs;
    vector<String> outNames = net.getUnconnectedOutLayersNames();
    net.forward(outputs, outNames);

    vector<Rect> boxes; vector<float> confs; vector<int> classIds;
    for (size_t i = 0; i < outputs.size(); ++i) {
        Mat pred = outputs[i];
        if (pred.dims == 3) {
            int rows = pred.size[1], cols = pred.size[2];
            const float* data = (float*)pred.data;
            for (int r = 0; r < rows; ++r) {
                const float* row = data + r * cols;
                float obj_conf = row[4];
                if (obj_conf <= 0) continue;
                float maxClassScore = 0; int cls = -1;
                for (int c = 5; c < cols; ++c) {
                    if (row[c] > maxClassScore) { maxClassScore = row[c]; cls = c - 5; }
                }
                float score = obj_conf * maxClassScore;
                if (score < YOLO_CONF_THRESH) continue;
                float cx = row[0], cy = row[1], bw = row[2], bh = row[3];
                // bbox in padded image scale; map back to original
                float x1 = cx - bw / 2.0f, y1 = cy - bh / 2.0f;
                float rx1 = (x1 - dx) / scale;
                float ry1 = (y1 - dy) / scale;
                float rw = bw / scale;
                float rh = bh / scale;
                Rect b = Rect(int(round(rx1)), int(round(ry1)), int(round(rw)), int(round(rh))) & Rect(0, 0, frame.cols, frame.rows);
                if (b.area() <= 0) continue;
                boxes.push_back(b);
                confs.push_back(score);
                classIds.push_back(cls);
            }
        }
        else if (pred.dims == 2) {
            int rows = pred.rows, cols = pred.cols;
            for (int r = 0; r < rows; ++r) {
                const float* row = pred.ptr<float>(r);
                float obj_conf = row[4];
                if (obj_conf <= 0) continue;
                float maxClassScore = 0; int cls = -1;
                for (int c = 5; c < cols; ++c) {
                    if (row[c] > maxClassScore) { maxClassScore = row[c]; cls = c - 5; }
                }
                float score = obj_conf * maxClassScore;
                if (score < YOLO_CONF_THRESH) continue;
                float cx = row[0], cy = row[1], bw = row[2], bh = row[3];
                float x1 = cx - bw / 2.0f, y1 = cy - bh / 2.0f;
                float rx1 = (x1 - dx) / scale;
                float ry1 = (y1 - dy) / scale;
                float rw = bw / scale;
                float rh = bh / scale;
                Rect b = Rect(int(round(rx1)), int(round(ry1)), int(round(rw)), int(round(rh))) & Rect(0, 0, frame.cols, frame.rows);
                if (b.area() <= 0) continue;
                boxes.push_back(b);
                confs.push_back(score);
                classIds.push_back(cls);
            }
        }
    }

    // NMS
    vector<int> idx;
    NMSBoxes(boxes, confs, YOLO_CONF_THRESH, YOLO_NMS_THRESH, idx);
    for (int i : idx) {
        outBoxes.push_back(boxes[i]);
        outConf.push_back(confs[i]);
        outClassIds.push_back(classIds[i]);
    }
}

// 以 ROI 为基准计算检测框与 ROI 的重叠占比
double overlapRatioROI(const Rect& roi, const Rect& box) {
    Rect inter = roi & box;
    if (inter.area() <= 0) return 0.0;
    return (double)inter.area() / (double)roi.area();
}

// baseline 对比：返回 ROI 内“差异像素比”（0..1）
double baselineDiffRatio(const Mat& roi, const Mat& baseline_roi) {
    // 将两者缩放到相同大小（如果必要）
    Mat a, b;
    if (roi.size() != baseline_roi.size()) {
        resize(baseline_roi, b, roi.size());
    }
    else b = baseline_roi;
    a = roi;

    Mat grayA, grayB;
    cvtColor(a, grayA, COLOR_BGR2GRAY);
    cvtColor(b, grayB, COLOR_BGR2GRAY);

    Mat diff;
    absdiff(grayA, grayB, diff);
    // 二值化：差异像素 = 灰度差 > DIFF_PIXEL_THRESH
    Mat mask = diff > DIFF_PIXEL_THRESH;
    double changed = (double)countNonZero(mask);
    double total = (double)(roi.rows * roi.cols);
    return changed / total;
}

// 备用：边缘 / 亮度检测（保留以防）
bool detect_edge_presence(const Mat& roi) {
    Mat gray, edges;
    cvtColor(roi, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 1.2);
    Canny(gray, edges, 50, 150);
    double edge_ratio = (double)countNonZero(edges) / (roi.rows * roi.cols);
    return edge_ratio > 0.015;
}
bool detect_color_change(const Mat& roi) {
    Mat hsv; cvtColor(roi, hsv, COLOR_BGR2HSV);
    vector<Mat> ch; split(hsv, ch);
    Scalar mean, stddev; meanStdDev(ch[2], mean, stddev);
    return stddev[0] > 18;
}

int main() {
    // load ROIs
    auto seats = loadROIs(roiPath);
    if (seats.empty()) { cerr << "未加载到 ROI，请检查 rois.csv\n"; return -1; }

    // baseline 图像
    Mat baselineFull = imread(baselinePath);
    if (baselineFull.empty()) {
        cerr << "警告：无法打开 baseline 图像: " << baselinePath << "\n"
            << "程序仍将运行，但无法进行基于 baseline 的物品检测（object）判定。\n";
    }

    // open video
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) { cerr << "无法打开视频: " << videoPath << endl; return -1; }
    int W = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS); if (fps <= 0) fps = 25.0;
    VideoWriter writer(outPath, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(W, H));
    if (!writer.isOpened()) cerr << "警告：VideoWriter 未打开，可能无法生成输出视频\n";

    // load yolo
    Net yolo;
    try {
        yolo = readNet(yoloModelPath);
    }
    catch (const std::exception& e) {
        cerr << "无法加载 YOLO 模型: " << yoloModelPath << "\nException: " << e.what() << endl;
        return -1;
    }
    yolo.setPreferableBackend(DNN_BACKEND_OPENCV);
    yolo.setPreferableTarget(DNN_TARGET_CPU);

    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(300, 16, true);

    vector<Rect> lastBoxes; vector<int> lastClassIds; vector<float> lastConfs;
    int frameIdx = 0;
    Mat frame, fgmask;

    cout << "开始处理（YOLO 每 " << YOLO_EVERY_N_FRAMES << " 帧运行一次）...\n";

    while (cap.read(frame)) {
        frameIdx++;
        // update background model (fallback)
        pMOG2->apply(frame, fgmask, 0.01);
        morphologyEx(fgmask, fgmask, MORPH_OPEN, Mat(), Point(-1, -1), 1);
        morphologyEx(fgmask, fgmask, MORPH_CLOSE, Mat(), Point(-1, -1), 1);

        if (frameIdx % YOLO_EVERY_N_FRAMES == 1) {
            lastBoxes.clear(); lastClassIds.clear(); lastConfs.clear();
            yoloDetect(yolo, frame, lastBoxes, lastConfs, lastClassIds);
        }

        Mat vis = frame.clone();

        for (auto& s : seats) {
            Rect r = s.rect & Rect(0, 0, frame.cols, frame.rows);
            if (r.area() <= 0) continue;

            // 1) YOLO 人体检测
            bool personDetected = false;
            for (size_t i = 0; i < lastBoxes.size(); ++i) {
                if (lastClassIds[i] == 0) { // COCO: person -> classId 0
                    double overlap = overlapRatioROI(r, lastBoxes[i]);
                    if (overlap > 0.12) { personDetected = true; break; }
                }
            }

            // 2) baseline 差异判定（当 baseline 有提供时）
            bool objectDetected = false;
            if (!baselineFull.empty()) {
                // 截取 baseline 对应 ROI 区域（需要 baseline 与视频分辨率一致或会自动 resize）
                Mat baseline_roi;
                if (r.x >= 0 && r.y >= 0 && r.x + r.width <= baselineFull.cols && r.y + r.height <= baselineFull.rows) {
                    baseline_roi = baselineFull(r);
                }
                else {
                    // baseline 分辨率可能不同 -> resize baselineFull to frame size first
                    Mat baseline_resized;
                    resize(baselineFull, baseline_resized, Size(frame.cols, frame.rows));
                    baseline_roi = baseline_resized(r);
                }

                Mat roi = frame(r);
                double diffRatio = baselineDiffRatio(roi, baseline_roi);
                if (diffRatio > DIFF_RATIO_THRESH) objectDetected = true;
            }
            else {
                // fallback: 使用 MOG2+edge+color 作为物品检测提示
                Mat fg = fgmask(r);
                double fg_ratio = (double)countNonZero(fg) / (double)r.area();
                bool edge_hit = detect_edge_presence(frame(r));
                bool color_hit = detect_color_change(frame(r));
                if (fg_ratio > FG_RATIO && (edge_hit || color_hit)) objectDetected = true;
            }

            // final per-frame occ decision:
            // priority: personDetected -> person; else objectDetected -> object; else empty
            if (personDetected) {
                s.person_cnt++; s.object_cnt = 0; s.empty_cnt = 0;
                if (s.person_cnt >= PERSON_CONFIRM) s.state = 1;
            }
            else if (objectDetected) {
                s.object_cnt++; s.person_cnt = 0; s.empty_cnt = 0;
                if (s.object_cnt >= OBJECT_CONFIRM) s.state = 2;
            }
            else {
                s.empty_cnt++; s.person_cnt = 0; s.object_cnt = 0;
                if (s.empty_cnt >= EMPTY_CONFIRM) s.state = 0;
            }

            Scalar col;
            string lab;
            if (s.state == 1) { col = Scalar(0, 200, 0); lab = "PERSON"; }
            else if (s.state == 2) { col = Scalar(0, 165, 255); lab = "OBJECT"; }
            else { col = Scalar(0, 0, 255); lab = "EMPTY"; }

            rectangle(vis, r, col, 2);
            putText(vis, lab, Point(r.x, r.y - 8), FONT_HERSHEY_SIMPLEX, 0.6, col, 2);
            putText(vis, "ID:" + to_string(s.id), Point(r.x, r.y + 18), FONT_HERSHEY_SIMPLEX, 0.5, col, 1);
        }

        // draw YOLO boxes for debugging
        for (size_t i = 0; i < lastBoxes.size(); ++i) {
            Rect b = lastBoxes[i] & Rect(0, 0, frame.cols, frame.rows);
            Scalar col = (lastClassIds[i] == 0) ? Scalar(0, 200, 0) : Scalar(200, 100, 0);
            rectangle(vis, b, col, 2);
            string txt = cv::format("%d %.2f", lastClassIds[i], lastConfs[i]);
            putText(vis, txt, Point(b.x, b.y - 6), FONT_HERSHEY_SIMPLEX, 0.5, col, 1);
        }

        if (writer.isOpened()) writer.write(vis);
        imshow("seat monitor (YOLO+baseline)", vis);
        char k = (char)waitKey(1);
        if (k == 27) break;
    }

    cap.release();
    if (writer.isOpened()) writer.release();
    destroyAllWindows();
    cout << "完成，输出: " << outPath << endl;
    return 0;
}
