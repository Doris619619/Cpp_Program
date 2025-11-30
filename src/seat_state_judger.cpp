#include "seat_state_judger.hpp"
#include "src/database/SeatDatabase.h"  
#include "data_structures.hpp"
#include <json.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <thread>
#include <climits> 
#include <opencv2/highgui.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

// 构造函数：初始化背景建模器
SeatStateJudger::SeatStateJudger() : db_(SeatDatabase::getInstance()) // 初始化数据库引用
{
    // 初始化数据库
    db_.initialize();
    
    mog2_ = createBackgroundSubtractorMOG2(500, 16.0, true);
    mog2_->setShadowValue(127);  // 阴影标记为 127，后续过滤
}

// 预处理前景掩码：过滤阴影+形态学去噪
Mat SeatStateJudger::preprocessFgMask(const Mat& frame, const Rect& roi) {
    Mat fg_mask;
    mog2_->apply(frame, fg_mask);  // 背景建模生成前景掩码
    // 过滤阴影（只保留前景=255，排除阴影=127）
    inRange(fg_mask, Scalar(255), Scalar(255), fg_mask);
    // 提取座位 ROI 区域的前景掩码
    Mat roi_fg_mask = fg_mask(roi);
    // 形态学开运算：先腐蚀再膨胀，去除小噪点
    Mat kernel = getStructuringElement(MORPH_RECT, Size(MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE));
    morphologyEx(roi_fg_mask, roi_fg_mask, MORPH_OPEN, kernel);
    return roi_fg_mask;
}

// 计算两个矩形的 IoU
float SeatStateJudger::calculateIoU(const Rect& rect1, const Rect& rect2) {
    int x1 = max(rect1.x, rect2.x);
    int y1 = max(rect1.y, rect2.y);
    int x2 = min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = min(rect1.y + rect1.height, rect2.y + rect2.height);
    int inter_area = max(0, x2 - x1) * max(0, y2 - y1);  // 交集面积
    int rect1_area = rect1.width * rect1.height;
    int rect2_area = rect2.width * rect2.height;
    int union_area = rect1_area + rect2_area - inter_area;  // 并集面积
    if (union_area == 0) return 0.0f;
    return static_cast<float>(inter_area) / union_area;
}

// 获取当前时间的 ISO8601 格式（如：2025-11-26T22:05:13.766）
string SeatStateJudger::getISO8601Timestamp() {
    auto now = chrono::system_clock::now();
    auto in_time_t = chrono::system_clock::to_time_t(now);
    struct tm local_tm{};
    localtime_r(&in_time_t, &local_tm);  // 线程安全的本地时间转换
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &local_tm);
    auto ms = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()) % 1000;
    stringstream ss;
    ss << buf << "." << setw(3) << setfill('0') << ms.count();
    return ss.str();
}

// 座位状态转字符串（用于输出）
string SeatStateJudger::stateToStr(B2CD_State::SeatStatus status) {
    if (status == B2CD_State::SEATED) return "Seated";
    else if (status == B2CD_State::UNSEATED) return "Unseated";
    else return "Anomaly";
}

// 核心函数：处理单条座位数据（完全适配结构体）
void SeatStateJudger::processAData(
    const A2B_Data& a_data,
    const json& seat_j,
    B2CD_State& state,
    vector<B2CD_Alert>& alerts,
    B2C_SeatSnapshot& out_snapshot,
    optional<B2C_SeatEvent>& out_event
) {
    // 初始化状态
    state.seat_id = a_data.seat_id;
    state.timestamp = a_data.timestamp;
    state.confidence = 0.90f;
    state.status_duration = 0;
    state.source_frame_id = a_data.frame_id;
    state.status = B2CD_State::UNSEATED;

    // 1. 从JSON获取真实时间戳（用ts_ms计算真实时间）
    int64_t current_ts_ms = seat_j.contains("ts_ms") ? seat_j["ts_ms"].get<int64_t>() : 0;
    static unordered_map<string, int64_t> last_seat_ts;  // 存储每个座位的上一帧时间戳
    static unordered_map<string, B2CD_State::SeatStatus> last_seat_status;  // 上一帧状态
    static unordered_map<string, int> anomaly_occupied_duration;  // 异常占座真实持续时间（秒）

    // 2. 计算当前帧与上一帧的真实时间差（秒）
    int time_diff_sec = 0;
    if (last_seat_ts.find(a_data.seat_id) != last_seat_ts.end() && current_ts_ms > 0) {
        time_diff_sec = static_cast<int>((current_ts_ms - last_seat_ts[a_data.seat_id]) / 1000);
        time_diff_sec = max(time_diff_sec, 0);  // 避免负数（帧乱序）
    }

    // 3. 提取核心数据
    int person_count = seat_j.contains("person_count") ? seat_j["person_count"].get<int>() : 0;
    int object_count = seat_j.contains("object_count") ? seat_j["object_count"].get<int>() : 0;
    string occupancy_state = seat_j.contains("occupancy_state") ? seat_j["occupancy_state"].get<string>() : "FREE";

    // 4. 状态判定（基于真实时间）
    B2CD_State::SeatStatus current_status = B2CD_State::UNSEATED;
    if (occupancy_state == "PERSON" || person_count > 0) {
        current_status = B2CD_State::SEATED;
        state.confidence = seat_j.contains("person_conf") ? seat_j["person_conf"].get<float>() : 0.95f;
        anomaly_occupied_duration[a_data.seat_id] = 0;  // 有人，重置异常占座计时
    } else if (occupancy_state == "OBJECT_ONLY" || (object_count > 0 && person_count == 0)) {
        // 异常占座：累加真实时间差
        anomaly_occupied_duration[a_data.seat_id] += time_diff_sec;
        if (anomaly_occupied_duration[a_data.seat_id] >= ANOMALY_THRESHOLD_SECONDS) {
            current_status = B2CD_State::ANOMALY_OCCUPIED;
            state.confidence = seat_j.contains("object_conf") ? seat_j["object_conf"].get<float>() : 0.85f;
            // 构造告警
            B2CD_Alert alert;
            alert.alert_id = a_data.seat_id + "_" + a_data.timestamp;
            alert.seat_id = a_data.seat_id;
            alert.alert_type = "AnomalyOccupied";
            alert.alert_desc = "座位被物品占用（无行人），持续" + to_string(anomaly_occupied_duration[a_data.seat_id]) + "秒";
            alert.timestamp = a_data.timestamp;
            alert.is_processed = false;
            alerts.push_back(alert);
        } else {
            current_status = B2CD_State::UNSEATED;  // 未到120秒，仍算空闲
        }
    } else {
        anomaly_occupied_duration[a_data.seat_id] = 0;  // 无物品，重置异常占座计时
        current_status = B2CD_State::UNSEATED;
    }

    // 5. 更新状态持续时间（真实时间累加）
    if (last_seat_status.find(a_data.seat_id) != last_seat_status.end() && 
        last_seat_status[a_data.seat_id] == current_status) {
        state.status_duration = last_seat_status_duration[a_data.seat_id] + time_diff_sec;
    } else {
        state.status_duration = time_diff_sec;  // 状态变化，重新计时
    }

    // 6. 更新当前状态和时间戳（供下一帧使用）
    state.status = current_status;
    last_seat_ts[a_data.seat_id] = current_ts_ms;
    last_seat_status[a_data.seat_id] = current_status;
    last_seat_status_duration[a_data.seat_id] = state.status_duration;

    // 7. 构造状态变化事件
    if (last_seat_status.find(a_data.seat_id) == last_seat_status.end() || 
        last_seat_status[a_data.seat_id] != current_status) {
        B2C_SeatEvent event;
        event.seat_id = a_data.seat_id;
        event.state = stateToStr(current_status);
        event.timestamp = a_data.timestamp;
        event.duration_sec = state.status_duration;
        out_event = event;
    }

    // 8. 构造快照
    out_snapshot.seat_id = a_data.seat_id;
    out_snapshot.state = stateToStr(current_status);
    out_snapshot.person_count = person_count;
    out_snapshot.timestamp = a_data.timestamp;
}

// 计时器：获取已流逝的秒数
int SeatStateJudger::SeatTimer::getElapsedSeconds() {
    if (!is_running) return 0;
    auto now = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::seconds>(now - start_time);
    return static_cast<int>(elapsed.count());
}

// 毫秒时间戳转 ISO8601 格式
string SeatStateJudger::msToISO8601(int64_t ts_ms) {
    auto sec = ts_ms / 1000;
    auto ms = ts_ms % 1000;
    auto time_point = chrono::system_clock::from_time_t(sec);
    auto in_time_t = chrono::system_clock::to_time_t(time_point);
    struct tm local_tm{};
    localtime_r(&in_time_t, &local_tm);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &local_tm);
    stringstream ss;
    ss << buf << "." << setw(3) << setfill('0') << ms;
    return ss.str();
}

// 读取 last_frame.json 数据（保留原功能，适配单帧监听）
bool SeatStateJudger::readLastFrameData(
    vector<A2B_Data>& out_a2b_data_list,
    vector<json>& out_seat_j_list
) {
    const string last_frame_path = "./runtime/last_frame.jsonl";
    if (!fs::exists(last_frame_path)) {
        cout << "[BModule] Warning: last_frame.jsonl not found" << endl;
        return false;
    }

    //读取last_frame.jsonl
    ifstream file(last_frame_path);
    string line;
    bool has_valid_data = false;

    while (getline(file, line)) {
        if (line.empty()) continue; // 跳过空行
        try {
            json j = json::parse(line); // 每行解析一个JSON对象
            has_valid_data = true;

            // 3. 解析A同学的JSON字段（按他的输出格式调整，以下是基于常规JSONL的示例）
            string image_path = j["image_path"].get<string>();
            replace(image_path.begin(), image_path.end(), '\\', '/'); // 兼容Windows路径（A同学用MSVC编译，路径是\\）
            string full_image_path = fs::current_path().string() + "/" + image_path;

            cv::Mat frame = cv::imread(full_image_path);
            if (frame.empty()) {
                cout << "[BModule] Warning: 图像文件未找到：" << full_image_path << "，使用默认黑帧" << endl;
                frame = cv::Mat::zeros(1080, 1920, CV_8UC3);
            }

            int frame_index = j["frame_index"].get<int>();
            string timestamp = msToISO8601(j["ts_ms"].get<int64_t>());

            // 解析每个座位的数据（A同学的JSON中“seats”数组存储单个座位信息）
            for (auto& seat_j : j["seats"]) {
                A2B_Data a2b_data;
                a2b_data.frame_id = frame_index;
                a2b_data.timestamp = timestamp;
                a2b_data.frame = frame;
                a2b_data.seat_id = to_string(seat_j["seat_id"].get<int>()); // 对齐A同学的seat_id字段

                // 解析座位ROI（A同学的JSON中“seat_roi”字段，x/y/w/h）
                int roi_x = seat_j["seat_roi"]["x"].get<int>();
                int roi_y = seat_j["seat_roi"]["y"].get<int>();
                int roi_w = seat_j["seat_roi"]["w"].get<int>();
                int roi_h = seat_j["seat_roi"]["h"].get<int>();
                // 兼容 seat_roi 全0的情况：用 seat_poly 计算最小包围矩形作为 ROI
                if (roi_w == 0 || roi_h == 0) {
                    // 后续通过 seat_poly 计算 ROI，先初始化一个默认值
                    a2b_data.seat_roi = cv::Rect(0, 0, 1, 1);
                } else {
                    a2b_data.seat_roi = cv::Rect(roi_x, roi_y, roi_w, roi_h);
                }
            }

            // 解析 seat_poly
            if (seat_j.contains("seat_poly") && seat_j["seat_poly"].is_array()) {
                std::vector<cv::Point2i> poly;
                auto& poly_arr = seat_j["seat_poly"]; // 外层数组
                for (size_t i = 0; i < poly_arr.size(); ++i) {
                    auto& point_arr = poly_arr[i]; // 内层 [x,y] 数组
                    if (point_arr.is_array() && point_arr.size() == 2) {
                        int px = point_arr[0].get<int>(); // 第一个元素是 x
                        int py = point_arr[1].get<int>(); // 第二个元素是 y
                        poly.emplace_back(px, py); // 构造点并添加到多边形
                    }
                }
                a2b_data.seat_poly = poly;

                //如果 seat_roi 无效（全0），用 seat_poly 计算最小包围矩形作为 ROI
                if (roi_w == 0 || roi_h == 0 && !poly.empty()) {
                    cv::Rect bounding_rect = cv::boundingRect(poly);
                    a2b_data.seat_roi = bounding_rect;
                }
            }

            // 解析A同学的人检测框（person_boxes字段，直接复用检测结果）
            for (auto& pb : seat_j["person_boxes"]) {
                DetectedObject obj;
                obj.bbox = cv::Rect(
                    pb["x"].get<int>(), pb["y"].get<int>(),
                    pb["w"].get<int>(), pb["h"].get<int>()
                );
                obj.score = pb["conf"].get<double>() / 10.0f; // 归一化到 0-1 范围
                obj.score = ob["conf"].get<double>() / 10.0f;
                obj.class_name = pb["cls_name"].get<string>();
                obj.class_id = pb["cls_id"].get<int>();
                a2b_data.person_boxes.push_back(obj);
            }

            // 解析A同学的物体检测框（object_boxes字段）
            for (auto& ob : seat_j["object_boxes"]) {
                DetectedObject obj;
                obj.bbox = cv::Rect(
                    ob["x"].get<int>(), ob["y"].get<int>(),
                    ob["w"].get<int>(), ob["h"].get<int>()
                );
                obj.score = pb["conf"].get<double>() / 10.0f; // 归一化到 0-1 范围
                obj.score = ob["conf"].get<double>() / 10.0f;
                obj.class_name = ob["cls_name"].get<string>();
                obj.class_id = ob["cls_id"].get<int>();
                a2b_data.object_boxes.push_back(obj);
            }

                out_a2b_data_list.push_back(a2b_data);
                out_seat_j_list.push_back(seat_j);
            }
        } catch (const json::exception& e) {
            cout << "[BModule] Error: 解析A同学的JSONL行失败：" << e.what() << endl;
            continue; // 跳过错误行，继续解析其他行
        }
    }

    return has_valid_data;
}


// 运行主函数
void SeatStateJudger::run(const string& jsonl_path) {
    // 每次运行前重置帧索引存储，避免多次调用时重复
    resetNeedStoreFrameIndexes();
    
    if (!jsonl_path.empty()) {
        vector<vector<A2B_Data>> batch_a2b_data;
        vector<vector<json>> batch_seat_j;
        if (readJsonlFile(jsonl_path, batch_a2b_data, batch_seat_j)) {
            cout << "[Info] 开始处理JSONL文件中的有效帧..." << endl;

            // 逐帧处理（遍历变量名改为jsonl_idx，明确是JSONL顺序索引）
            for (size_t jsonl_idx = 0; jsonl_idx < batch_a2b_data.size(); ++jsonl_idx) {
                auto& frame_a2b = batch_a2b_data[jsonl_idx];
                auto& frame_seat_j = batch_seat_j[jsonl_idx];

                // 关键修改：获取A同学的frame_index
                int a_frame_index = frame_a2b[0].frame_id;  // 同一帧所有座位frame_id相同，取第一个
                cout << "[📺 Frame (A同学ID:" << a_frame_index << ")] 开始处理（" << frame_a2b.size() << "个有效座位）..." << endl;

                // 新增：标记当前帧是否需要入库（满足任一条件即需入库）
                bool need_store_this_frame = false;

                for (size_t seat_idx = 0; seat_idx < frame_a2b.size(); ++seat_idx) {
                    B2CD_State state;
                    vector<B2CD_Alert> alerts;
                    B2C_SeatSnapshot snapshot;
                    optional<B2C_SeatEvent> event;

                    processAData(frame_a2b[seat_idx], frame_seat_j[seat_idx], state, alerts, snapshot, event);

                    ///YZC：我添加了这三个调用函数（最好还要对一下头文件的路径，database.h 这个）
                    // 数据库入库操作
                    if (event.has_value()) {
                        db_.insertSeatEvent(  // 使用成员变量db_
                            event->seat_id,
                            event->state,
                            event->timestamp,
                            event->duration_sec
                        );
                    }
                    
                    // 插入快照
                    db_.insertSnapshot(
                        snapshot.timestamp,
                        snapshot.seat_id,  
                        snapshot.state,
                        snapshot.person_count
                    );
                    
                    // 插入告警
                    for (const auto& alert : alerts) {
                        db_.insertAlert(  // 使用成员变量db_
                            alert.alert_id,
                            alert.seat_id,
                            alert.alert_type,
                            alert.alert_desc,
                            alert.timestamp,
                            alert.is_processed
                        );
                    }
                    
                    // 判定条件（可与A同学协商调整）
                    if (event.has_value()  // 有座位状态变化（如空闲→有人、有人→异常）
                        || !alerts.empty() // 有异常占座告警
                        || state.status != B2CD_State::UNSEATED) { // 座位非空闲（有人或异常占座）
                        need_store_this_frame = true;
                    }

                    // 原有输出逻辑（不变）
                    cout << "  座位 " << state.seat_id << ":" << endl;
                    cout << "    状态: " << stateToStr(state.status) << endl;
                    cout << "    持续时间: " << state.status_duration << "秒" << endl;
                    cout << "    置信度: " << fixed << setprecision(2) << state.confidence << endl;
                    cout << "    关联帧ID: " << state.source_frame_id << endl;
                    if (!alerts.empty()) {
                        cout << "    ⚠️  告警: " << alerts[0].alert_desc << endl;
                        cout << "       告警ID: " << alerts[0].alert_id << endl;
                    }
                    if (event.has_value()) {
                        cout << "    🔄 [状态变化] 变为: " << event.value().state << "（持续" << event.value().duration_sec << "秒）" << endl;
                    }
                    cout << endl;
                }

                // 关键修改：若当前帧需要入库，记录A的frame_index
                if (need_store_this_frame) {
                    need_store_frame_indexes_.insert(a_frame_index);
                    cout << "[Info] 标记帧（A同学ID:" << a_frame_index << "）为需要入库" << endl;
                }

                cout << "[📺 Frame (A同学ID:" << a_frame_index << ")] 处理完成" << endl;
                cout << "-------------------------------------" << endl;
                this_thread::sleep_for(chrono::milliseconds(100));
            }

            // 新增：输出需要入库的帧索引列表（供A查看）
            //vector<int> need_store_list = getNeedStoreFrameIndexes();
            //cout << "[Info] 所有帧处理完成！需要入库的帧索引：" << endl;
            //for (int idx : need_store_list) {
                //cout << " - " << idx << endl;
            //}
        }
    } else {
        // last_frame.json监听逻辑
        cout << "[Info] 未指定JSONL路径，开始监听 last_frame.json..." << endl;
        while (true) {
            vector<A2B_Data> a2b_data_list;
            vector<json> seat_j_list;
            if (readLastFrameData(a2b_data_list, seat_j_list)) {
                cout << "[📺 Last Frame] 开始处理（" << a2b_data_list.size() << "个座位）..." << endl;
                for (size_t i = 0; i < a2b_data_list.size(); ++i) {
                    B2CD_State state;
                    vector<B2CD_Alert> alerts;
                    B2C_SeatSnapshot snapshot;
                    optional<B2C_SeatEvent> event;
                    processAData(a2b_data_list[i], seat_j_list[i], state, alerts, snapshot, event);

                    /////YZC：实时监听模式也入库
                    if (event.has_value()) {
                        db_.insertSeatEvent(
                            event->seat_id,
                            event->state,
                            event->timestamp,
                            event->duration_sec
                        );
                    }

                    db_.insertSnapshot(
                        snapshot.timestamp,
                        snapshot.seat_id,
                        snapshot.state,
                        snapshot.person_count
                    );

                    for (const auto& alert : alerts) {
                        db_.insertAlert(
                            alert.alert_id,
                            alert.seat_id,
                            alert.alert_type,
                            alert.alert_desc,
                            alert.timestamp,
                            alert.is_processed
                        );
                    }
                    
                    cout << "  座位 " << state.seat_id << ":" << endl;
                    cout << "    状态: " << stateToStr(state.status) << endl;
                    cout << "    持续时间: " << state.status_duration << "秒" << endl;
                    cout << "    置信度: " << fixed << setprecision(2) << state.confidence << endl;
                    if (!alerts.empty()) {
                        cout << "    ⚠️  告警: " << alerts[0].alert_desc << endl;
                    }
                    cout << endl;
                }
                cout << "[📺 Last Frame] 处理完成" << endl;
                cout << "-------------------------------------" << endl;
            }
            this_thread::sleep_for(chrono::seconds(1));
        }
    }
}

// 读取JSONL文件
bool SeatStateJudger::readJsonlFile(
    const std::string& jsonl_path,
    std::vector<std::vector<A2B_Data>>& batch_a2b_data,
    std::vector<std::vector<json>>& batch_seat_j
) {
    // 若未指定路径，默认读取A的输出路径
    std::string actual_path = jsonl_path.empty() ? "./runtime/seat_states.jsonl" : jsonl_path;
    if (!fs::exists(actual_path)) {
        cout << "[BModule] Error: A的JSONL文件未找到（路径：" << actual_path << "）" << endl;
        cout << "[BModule] 提示：请运行A的 a_demo.exe --out " << actual_path << " 生成文件" << endl;
        return false;
    }

    ifstream file(actual_path);
    string line;
    int frame_count = 0;

    while (getline(file, line)) {
        if (line.empty()) continue;
        try {
            json j = json::parse(line);
            std::vector<A2B_Data> frame_a2b;
            std::vector<json> frame_seat_j;

            // 解析单帧数据（逻辑和readLastFrameData一致，复用A同学的字段）
            string image_path = j["image_path"].get<string>();
            replace(image_path.begin(), image_path.end(), '\\', '/');
            string full_image_path = fs::current_path().string() + "/" + image_path;

            cv::Mat frame = cv::imread(full_image_path);
            if (frame.empty()) {
                cout << "[BModule] Warning: 图像文件未找到：" << full_image_path << "，使用默认黑帧" << endl;
                frame = cv::Mat::zeros(1080, 1920, CV_8UC3);
            }

            int frame_index = j["frame_index"].get<int>();
            string timestamp = msToISO8601(j["ts_ms"].get<int64_t>());

            for (auto& seat_j : j["seats"]) {
                A2B_Data a2b_data;
                a2b_data.frame_id = frame_index;
                a2b_data.timestamp = timestamp;
                a2b_data.frame = frame;
                a2b_data.seat_id = to_string(seat_j["seat_id"].get<int>());

                // 解析座位ROI
                int roi_x = seat_j["seat_roi"]["x"].get<int>();
                int roi_y = seat_j["seat_roi"]["y"].get<int>();
                int roi_w = seat_j["seat_roi"]["w"].get<int>();
                int roi_h = seat_j["seat_roi"]["h"].get<int>();
                // 兼容 seat_roi 全0的情况：用 seat_poly 计算最小包围矩形作为 ROI
                if (roi_w == 0 || roi_h == 0) {
                    // 后续通过 seat_poly 计算 ROI，先初始化一个默认值
                    a2b_data.seat_roi = cv::Rect(0, 0, 1, 1);
                } else {
                    a2b_data.seat_roi = cv::Rect(roi_x, roi_y, roi_w, roi_h);
                }
            }

            // 解析 seat_poly
            if (seat_j.contains("seat_poly") && seat_j["seat_poly"].is_array()) {
                std::vector<cv::Point2i> poly;
                auto& poly_arr = seat_j["seat_poly"]; // 外层数组
                for (size_t i = 0; i < poly_arr.size(); ++i) {
                    auto& point_arr = poly_arr[i]; // 内层 [x,y] 数组
                    if (point_arr.is_array() && point_arr.size() == 2) {
                        int px = point_arr[0].get<int>(); // 第一个元素是 x
                        int py = point_arr[1].get<int>(); // 第二个元素是 y
                        poly.emplace_back(px, py); // 构造点并添加到多边形
                    }
                }
                a2b_data.seat_poly = poly;

                //如果 seat_roi 无效（全0），用 seat_poly 计算最小包围矩形作为 ROI
                if (roi_w == 0 || roi_h == 0 && !poly.empty()) {
                    cv::Rect bounding_rect = cv::boundingRect(poly);
                    a2b_data.seat_roi = bounding_rect;
                }
            }


                // 解析人检测框
                for (auto& pb : seat_j["person_boxes"]) {
                    DetectedObject obj;
                    obj.bbox = cv::Rect(pb["x"].get<int>(), pb["y"].get<int>(), pb["w"].get<int>(), pb["h"].get<int>());                    obj.score = pb["conf"].get<double>() / 10.0f; // 归一化到 0-1 范围
                    obj.score = ob["conf"].get<double>() / 10.0f;
                    obj.class_name = pb["cls_name"].get<string>();
                    obj.class_id = pb["cls_id"].get<int>();
                    a2b_data.person_boxes.push_back(obj);
                }

                // 解析物体检测框
                for (auto& ob : seat_j["object_boxes"]) {
                    DetectedObject obj;
                    obj.bbox = cv::Rect(ob["x"].get<int>(), ob["y"].get<int>(), ob["w"].get<int>(), ob["h"].get<int>());
                    obj.score = pb["conf"].get<double>() / 10.0f; // 归一化到 0-1 范围
                    obj.score = ob["conf"].get<double>() / 10.0f;
                    obj.class_name = ob["cls_name"].get<string>();
                    obj.class_id = ob["cls_id"].get<int>();
                    a2b_data.object_boxes.push_back(obj);
                }

                frame_a2b.push_back(a2b_data);
                frame_seat_j.push_back(seat_j);
            }

            if (!frame_a2b.empty()) {
                batch_a2b_data.push_back(frame_a2b);
                batch_seat_j.push_back(frame_seat_j);
                frame_count++;
            }
        } catch (const json::exception& e) {
            cout << "[BModule] Error: 解析批量JSONL行失败：" << e.what() << endl;
            continue;
        }
    }

    cout << "[BModule] 成功读取A的" << frame_count << "帧批量数据" << endl;
    return frame_count > 0;
}