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
        VisionConfig cfg;
        std::vector<SeatROI> seats;
        OrtYoloDetector detector{ OrtYoloDetector::SessionOptions{ // struct session_options
            "data/models/yolov8n_640.onnx", // model path
            640,                            // input_w
            640,                            // input_h
            true                            // fake_infer   
        } };
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

        // 1. 预处理：resize 到 detector 输入尺寸 (640x640)
        cv::Mat resized;
        //cv::resize(bgr, resized, cv::Size(impl_->detector.isReady() ? impl_->detector.infer(resized).size(): 0), 0, 0); // placeholder to avoid unused warnings
        cv::resize(bgr, resized, cv::Size(640, 640));
        // difference to letterbox? 

        // 2. 推理 (fake random detections); 将 RawDet 转为 BBox
        auto raw = impl_->detector.infer(resized);
        std::vector<BBox> dets;
        dets.reserve(raw.size());
        for (auto& r : raw) {
            BBox b;
            // scale 回原图坐标
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

        // 人与物简易分类
        std::vector<BBox> persons, objects;
        for (auto& b : dets) {
            if (b.cls_name == "person") persons.push_back(b);
            else objects.push_back(b);
        }
        // 3. 座位归属: 根据IoU与thres确定座位内元素
        auto iouSeat = [](const cv::Rect& seat, const cv::Rect& box) {
            int ix = std::max(seat.x, box.x);
            int iy = std::max(seat.y, box.y);
            int iw = std::min(seat.x + seat.width, box.x + box.width) - ix;
            int ih = std::min(seat.y + seat.height, box.y + box.height) - iy;
            if (iw <=0 || ih <=0) return 0.f;
            float inter = iw * ih;
            float uni = seat.width * seat.height + box.width * box.height - inter;
            return uni<=0 ? 0.f : inter / uni;
        };

    /*      Output SeatFrameState for each seat 
    *  record all the result into the vector containing all the SeatFrameState 
    *  (denoted as out, std::vector<SeatFrameState> )
    */
        for (auto& each_seat : impl_->seats) {
            SeatFrameState sfs;
            sfs.seat_id = each_seat.seat_id;
            sfs.ts_ms = ts_ms;
            sfs.frame_index = frame_index;
            sfs.seat_roi = each_seat.rect;

        // collect boxes inside seat
            for (auto& p : persons) {
                if (iouSeat(each_seat.rect, p.rect) > impl_->cfg.iou_seat_intersect) {
                    sfs.person_boxes_in_roi.push_back(p);
                    sfs.person_conf = std::max(sfs.person_conf, p.conf);
                }
            }
            for (auto& o : objects) {
                if (iouSeat(each_seat.rect, o.rect) > impl_->cfg.iou_seat_intersect) {
                    sfs.object_boxes_in_roi.push_back(o);
                    sfs.object_conf = std::max(sfs.object_conf, o.conf);
                }
            }
            sfs.person_count = static_cast<int>(sfs.person_boxes_in_roi.size());
            sfs.object_count = static_cast<int>(sfs.object_boxes_in_roi.size());
            sfs.has_person = sfs.person_count > 0 && sfs.person_conf >= impl_->cfg.conf_thres_person;  // 有人 = 人数 > 0 and conf > conf_thres_person
            sfs.has_object = sfs.object_count > 0 && sfs.object_conf >= impl_->cfg.conf_thres_object;  // 有物 = 物数 > 0 and conf > conf_thres_object

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