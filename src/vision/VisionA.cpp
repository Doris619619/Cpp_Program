// 简化版本的 VisionA：当前仅加载座位并返回 UNKNOWN 状态
#include "vision/VisionA.h"
#include "vision/Publish.h"
#include "vision/SeatRoi.h"
#include "vision/Types.h"
#include "vision/Config.h"
#include "vision/OrtYolo.h"
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

        // 1. 预处理：size parse dst: detector input size (640x640)
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

        // 4. 座位归属: 根据IoU与thres确定座位内元素
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

            // collect boxes inside seat
            for (auto& p : persons) {
                if (iouSeat(each_seat.rect, p.rect) > impl_->cfg.iou_seat_intersect) {
                    sfs.person_boxes_in_roi.push_back(p);
                    sfs.person_conf_max = std::max(sfs.person_conf_max, p.conf);
                }
            }
            for (auto& o : objects) {
                if (iouSeat(each_seat.rect, o.rect) > impl_->cfg.iou_seat_intersect) {
                    sfs.object_boxes_in_roi.push_back(o);
                    sfs.object_conf_max = std::max(sfs.object_conf_max, o.conf);
                }
            }
            sfs.person_count = static_cast<int>(sfs.person_boxes_in_roi.size());
            sfs.object_count = static_cast<int>(sfs.object_boxes_in_roi.size());
            sfs.has_person = sfs.person_count > 0 && sfs.person_conf_max >= impl_->cfg.conf_thres_person;  // 有人 = 人数 > 0 and conf > conf_thres_person
            sfs.has_object = sfs.object_count > 0 && sfs.object_conf_max >= impl_->cfg.conf_thres_object;  // 有物 = 物数 > 0 and conf > conf_thres_object

            // occupancy rule: has_person => OCCUPIED, else if has_object => OBJECT_ONLY, else EMPTY
            if (sfs.has_person) sfs.occupancy_state = SeatOccupancyState::PERSON;
            else if (sfs.has_object) sfs.occupancy_state = SeatOccupancyState::OBJECT_ONLY;
            else sfs.occupancy_state = SeatOccupancyState::FREE;

            out.push_back(std::move(sfs));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        int total_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
        for (auto& each_sfs : out) each_sfs.t_post_ms = total_ms; // 简化: 全流程耗时
        return out;
    }

    void VisionA::setPublisher(Publisher* p) {
        // 留空：demo 未使用
        (void)p;
    }

} // namespace vision