// 简化版本的 VisionA：当前仅加载座位并返回 UNKNOWN 状态
#include "vision/VisionA.h"
#include "vision/Publish.h"
#include "vision/SeatRoi.h"
#include "vision/Types.h"
#include "vision/Config.h"
#include "vision/OrtYolo.h"
#include "vision/Mog2.h"
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <chrono>

namespace vision {

    struct VisionA::Impl {
        VisionConfig cfg;                   // configurator &cfg
        std::vector<SeatROI> seats;
        OrtYoloDetector detector{ OrtYoloDetector::SessionOptions{ // struct session_options
            "data/models/yolov8n_640.onnx", // model path
            640,                            // input_w
            640,                            // input_h
            false                           // fake_infer = false 启用真实推理
        } };
        Mog2Manager mog2{ Mog2Config{ cfg.mog2_history, cfg.mog2_var_threshold, cfg.mog2_detect_shadows } };
        // 存储最后一帧的所有检测结果
        std::vector<BBox> last_persons;
        std::vector<BBox> last_objects;
        struct SizeParseResult {
            cv::Mat img;
            float scale;
            int dx, dy;
        };

        // method: resize the input img
        static SizeParseResult sizeParse(const cv::Mat& src, int target_size) {
            int w = src.cols, h = src.rows;
            float scaling_rate = std::min((float)target_size / w, (float)target_size / h);
            int new_w = int(std::round(w * scaling_rate));
            int new_h = int(std::round(h * scaling_rate));
            int dx = (target_size - new_w) / 2;
            int dy = (target_size - new_h) / 2;

            cv::Mat resized;
            cv::resize(src, resized, cv::Size(new_w, new_h));

            cv::Mat canvas = cv::Mat::zeros(target_size, target_size, src.type());
            resized.copyTo(canvas(cv::Rect(dx, dy, new_w, new_h)));    
            // now canvas contains the resized image, and canvas is cv::Mat

            return {canvas, scaling_rate, dx, dy};
        }
    };

    VisionA::VisionA(const VisionConfig& cfg) 
        : impl_(new Impl)
    {
        impl_->cfg = cfg;
        loadSeatsFromJson(cfg.seats_json, impl_->seats);
    }

    VisionA::~VisionA() = default;

    std::vector<SeatFrameState> VisionA::processFrame(const cv::Mat& bgr, 
                                                      int64_t ts_ms, 
                                                      int64_t frame_index) 
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<SeatFrameState> out;
        out.reserve(impl_->seats.size());

        if (bgr.empty()) return out;

        // 1. 前景分割（原尺寸）
        cv::Mat fg_mask = impl_->mog2.apply(bgr);

        // 2. 预处理：letterbox（保持比例，减少形变）
        auto sizeParseRes = Impl::sizeParse(bgr, 640);
        auto parsed_img = sizeParseRes.img;

        // 2. 推理: chg RawDet -> BBox
        auto raw_detected = impl_->detector.infer(parsed_img);
        std::vector<BBox> dets;
        dets.reserve(raw_detected.size());
        for (auto& r : raw_detected) {
            BBox b;
            // scale to original
            float sx = static_cast<float>(bgr.cols) / 640.f;
            float sy = static_cast<float>(bgr.rows) / 640.f;
            float x = r.cx - r.w * 0.5f;
            float y = r.cy - r.h * 0.5f;
            b.rect = cv::Rect(static_cast<int>(x * sx), 
                              static_cast<int>(y * sy), 
                              static_cast<int>(r.w * sx), 
                              static_cast<int>(r.h * sy));
            b.conf = r.conf;
            b.cls_id = r.cls_id;
            b.cls_name = (r.cls_id == 0 ? "person" : "object");
            dets.push_back(b);
        }

        // 3. 人与物简易分类
        std::vector<BBox> persons, objects;   // persons boxes and objects boxes
        for (auto& b : dets) {
            if (b.cls_name == "person") persons.push_back(b);
            else                        objects.push_back(b);
        }
        
        // 保存本帧所有检测结果供外部访问
        impl_->last_persons = persons;
        impl_->last_objects = objects;

        // 4. 座位归属: 根据多边形包含或 IoU 判定座位内元素
        auto iouSeat = [](const cv::Rect& seat, const cv::Rect& box) {
            int ix = std::max(seat.x, box.x);
            int iy = std::max(seat.y, box.y);
            int iw = std::min(seat.x + seat.width, box.x + box.width) - ix;
            int ih = std::min(seat.y + seat.height, box.y + box.height) - iy;
            if (iw <= 0 || ih <= 0) return 0.f;
            float inter = iw * ih;
            float uni = seat.width * seat.height + box.width * box.height - inter;
            return uni <= 0 ? 0.f : (inter / uni);  // IoU = inter / uni 交并比
        };
        
        // 多边形包含判定：检测框中心点是否在多边形内
        auto isBoxInPoly = [](const std::vector<cv::Point>& poly, const cv::Rect& box) {
            if (poly.size() < 3) return false;
            cv::Point center(box.x + box.width / 2, box.y + box.height / 2);
            double dist = cv::pointPolygonTest(poly, center, false);
            return dist >= 0;  // >= 0 表示在多边形内或边界上
        };

    /*      Output SeatFrameState for each seat 
    *  record all the result into the vector containing all the SeatFrameState 
    *  (denoted as out, std::vector<SeatFrameState> )
    */
        for (auto& each_seat : impl_->seats) {  // for each seat in seats table
            SeatFrameState sfs;
            sfs.seat_id = each_seat.seat_id;
            sfs.ts_ms = ts_ms;
            sfs.frame_index = frame_index;
            sfs.seat_roi = each_seat.rect;
            sfs.seat_poly = each_seat.poly;  // 保存多边形信息
            
            // 判断使用多边形还是矩形
            bool use_poly = each_seat.poly.size() >= 3;

            // collect boxes inside seat
            for (auto& p : persons) {
                bool inside = false;
                if (use_poly) {
                    // 增强：框中心或四角任一在多边形内
                    if (isBoxInPoly(each_seat.poly, p.rect)) inside = true;
                    else {
                        std::array<cv::Point,4> corners = {
                            cv::Point(p.rect.x, p.rect.y),
                            cv::Point(p.rect.x+p.rect.width, p.rect.y),
                            cv::Point(p.rect.x, p.rect.y+p.rect.height),
                            cv::Point(p.rect.x+p.rect.width, p.rect.y+p.rect.height)
                        };
                        for (auto &c : corners) {
                            if (cv::pointPolygonTest(each_seat.poly, c, false) >= 0) { inside = true; break; }
                        }
                    }
                } else {
                    inside = (iouSeat(each_seat.rect, p.rect) > impl_->cfg.iou_seat_intersect);
                }
                if (inside) {
                    sfs.person_boxes_in_roi.push_back(p);
                    sfs.person_conf_max = std::max(sfs.person_conf_max, p.conf);
                }
            }
            for (auto& o : objects) {
                bool inside = false;
                if (use_poly) {
                    if (isBoxInPoly(each_seat.poly, o.rect)) inside = true;
                    else {
                        std::array<cv::Point,4> corners = {
                            cv::Point(o.rect.x, o.rect.y),
                            cv::Point(o.rect.x+o.rect.width, o.rect.y),
                            cv::Point(o.rect.x, o.rect.y+o.rect.height),
                            cv::Point(o.rect.x+o.rect.width, o.rect.y+o.rect.height)
                        };
                        for (auto &c : corners) {
                            if (cv::pointPolygonTest(each_seat.poly, c, false) >= 0) { inside = true; break; }
                        }
                    }
                } else {
                    inside = (iouSeat(each_seat.rect, o.rect) > impl_->cfg.iou_seat_intersect);
                }
                if (inside) {
                    sfs.object_boxes_in_roi.push_back(o);
                    sfs.object_conf_max = std::max(sfs.object_conf_max, o.conf);
                }
            }
            // 前景占比：多边形优先
            if (use_poly) {
                sfs.fg_ratio = Mog2Manager::ratioInPoly(fg_mask, each_seat.poly);
            } else {
                sfs.fg_ratio = impl_->mog2.ratioInRoi(fg_mask, each_seat.rect);
            }
            sfs.person_count = static_cast<int>(sfs.person_boxes_in_roi.size());
            sfs.object_count = static_cast<int>(sfs.object_boxes_in_roi.size());
            sfs.has_person = sfs.person_count > 0 && sfs.person_conf_max >= impl_->cfg.conf_thres_person;  // 有人 = 人数 > 0 and conf > conf_thres_person
            sfs.has_object = sfs.object_count > 0 && sfs.object_conf_max >= impl_->cfg.conf_thres_object;  // 有物 = 物数 > 0 and conf > conf_thres_object

            // occupancy rule: has_person => OCCUPIED, else if has_object => OBJECT_ONLY, else EMPTY
            if (sfs.has_person) {
                sfs.occupancy_state = SeatOccupancyState::PERSON;
            } else if (sfs.has_object) {
                sfs.occupancy_state = SeatOccupancyState::OBJECT_ONLY;
            } else {
                // 使用前景兜底：若无检测但前景占比超过阈值，标记为 OBJECT_ONLY（可能有人低头/遮挡）
                if (sfs.fg_ratio >= impl_->cfg.mog2_fg_ratio_thres) {
                    sfs.occupancy_state = SeatOccupancyState::OBJECT_ONLY;
                } else {
                    sfs.occupancy_state = SeatOccupancyState::FREE;
                }
            }

            out.push_back(std::move(sfs));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        int total_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
        for (auto& each_sfs : out) each_sfs.t_post_ms = total_ms; // 简化: 全流程耗时
        return out;
    }

    void VisionA::getLastDetections(std::vector<BBox>& out_persons, std::vector<BBox>& out_objects) const {
        out_persons = impl_->last_persons;
        out_objects = impl_->last_objects;
    }

    void VisionA::setPublisher(Publisher* p) {
        // 留空：demo 未使用
        (void)p;
    }

} // namespace vision