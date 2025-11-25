#pragma once
#include <string>
#include <vector>

namespace vision {

// VisionA running config (load from vision.yml)
struct VisionConfig {
    // ===================== 字段fields ===================== //
    
    std::string seats_json   = "config/poly_simple_seats.json";//test_seats.json";//"config/seats.json";             // seats ROI configs
    std::string model_path   = "data/models/yolov8n_640.onnx";  // available onnx models path
    std::string vision_yaml  = "config/vision.yml";             // config file self path (yaml file contains the overall configuration for VisionA)
    std::string log_dir      = "logs";
    std::string snapshot_dir = "cache/snap";
    std::string states_output = "runtime/seat_states.jsonl";    // 每帧座位状态输出(.jsonl)
    std::string annotated_frames_dir = "runtime/frames";        // 帧级标注输出目录(原始图 + 画框 + 座位ROI + 状态)

    // 模型输入尺寸
    int input_w = 640;
    int input_h = 640;

    /* 阈值
    * 置信度: 检测框内对象为特定类别的可信度
    * NMS:   多框重叠时仅保留置信度最高者，避免单人连续多框
    * IoU:   对于重复方框，计算其交并比(IoU = |A and B|/|A or B|)以决定是否合并或归属(保高分抑低分)
    */
    float conf_thres_person      = 0.35f; // 人框置信度阈值, box_conf > ~ 才算人
    float conf_thres_person_low  = 0.30f; // 人框低置信, 高低阈值之间的边缘人框需借助其它条件判断
    float conf_thres_object      = 0.40f; // 物框置信度阈值, box_conf > ~ 才算物
    float nms_iou                = 0.50f; // overlapping box IoU thres
    float iou_seat_intersect     = 0.20f; // 框与座位ROI归属IoU thres, IoU > ~ 才算在座位内
    float mog2_fg_ratio_thres    = 0.08f; // 前景占比兜底阈值

    // 快照策略
    int snapshot_jpg_quality     = 90;
    int snapshot_min_interval_ms = 3000;  // 同座位最小间隔 (此处3000ms，拍照是否有那么频？)
    bool snapshot_on_change_only = true;  // 是否只在状态变化时保存
    int snapshot_heartbeat_ms    = 5000;  // 若非变化也可周期心跳（为0禁用）

    // 允许的物品类别（与模型类别名称对齐）
    std::vector<std::string> object_allow = {
        "backpack","bag","book","laptop","bottle","umbrella"
    };
    // TODO: check the above with the actual model's class names

    // 标注图像与相关记录存储参数
    int annotated_save_freq = 100;          // 标注图像保存频率(save every N frames)

    // ONNX Runtime 线程配置
    int intra_threads = 0; // 0=auto

    // MOG2 参数（可调）
    int mog2_history         = 500;     // bg model 历史帧数, ↑: 稳定性↑ 适应性↓
    int mog2_var_threshold   = 16;      // 判定属于bg的方差阈值
    bool mog2_detect_shadows = false;   // 是否检测阴影

    // 前景噪声后处理（可选）
    bool fg_morph_enable = false;       // 是否启用形态学处理
    int  fg_morph_erode_iterations = 0;

    // 性能/调试
    bool dump_perf_log = true;
    bool enable_async_snapshot = true;

    // 兼容预留字段：可用于不同 YOLO 解码类型
    std::string yolo_variant = "yolov8n"; // 或 "yolov5", "yolov8"

    // ===================== 方法methods ===================== //

    // 配置加载函数
    static VisionConfig fromYaml(const std::string& yaml_path);
    static VisionConfig fromJson(const std::string& json_path);
};

} // namespace vision