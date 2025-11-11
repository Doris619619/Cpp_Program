#pragma once
// 基础模型与枚举：座位/分区/事件

#include <QtCore/QDateTime>
#include <QtCore/QMetaType>
#include <QtCore/QRectF>
#include <QtCore/QString>
#include <QtCore/QJsonObject>
#include <QtCore/QJsonArray>
#include <vector>

namespace seatui {

// 座位状态
enum class SeatState : int {
    Unseated = 0,   // 空闲
    Seated   = 1,   // 有人
    Anomaly  = 2    // 异常（无人但有物等）
};

// 事件类型（管理员端用于告警/日志）
enum class EventType : int {
    SeatStateChange = 0,
    AnomalyDetected = 1,
    CapacityThreshold = 2,
    ReportFromStudent = 3
};

// 座位信息（在平面图坐标系中用 roi 表示区域）
struct SeatInfo {
    int         seat_id = -1;
    QRectF      roi;                    // 像素或逻辑坐标，后续统一一个基准
    SeatState   state = SeatState::Unseated;
    QDateTime   last_update;
    QString     zone_name;              // 可选：所属分区
    int         occupied_seconds = 0;   // 状态持续时长（秒）
};

// 分区信息（用于热力图/统计）
struct ZoneInfo {
    int      zone_id = -1;
    QString  name;
    QRectF   area;                      // 分区几何范围
    int      seat_total = 0;
    int      seat_occupied = 0;         // 当前有人座位数
    double   occupancy_ratio = 0.0;     // 0~1（seat_occupied/seat_total）
};

// 后端事件（供管理员端告警中心/时间轴使用）
struct SeatEvent {
    EventType   type = EventType::SeatStateChange;
    int         seat_id = -1;           // 若为分区事件可置 -1
    QDateTime   ts;
    QString     extra_json;             // 附加信息（JSON 串：时长/截图路径/原因等）
};

// ------ 便捷序列化（可用于 WS/SQLite JSON 列） ------
inline QString toString(SeatState s) {
    switch (s) {
        case SeatState::Unseated: return "Unseated";
        case SeatState::Seated:   return "Seated";
        case SeatState::Anomaly:  return "Anomaly";
    }
    return "Unseated";
}
inline QJsonObject toJson(const SeatInfo& s) {
    return {
        {"seat_id", s.seat_id},
        {"roi", QJsonArray{ s.roi.x(), s.roi.y(), s.roi.width(), s.roi.height() }},
        {"state", toString(s.state)},
        {"last_update", s.last_update.toUTC().toString(Qt::ISODateWithMs)},
        {"zone_name", s.zone_name},
        {"occupied_seconds", s.occupied_seconds}
    };
}
inline QJsonObject toJson(const ZoneInfo& z) {
    return {
        {"zone_id", z.zone_id},
        {"name", z.name},
        {"area", QJsonArray{ z.area.x(), z.area.y(), z.area.width(), z.area.height() }},
        {"seat_total", z.seat_total},
        {"seat_occupied", z.seat_occupied},
        {"occupancy_ratio", z.occupancy_ratio}
    };
}
inline QJsonObject toJson(const SeatEvent& e) {
    return {
        {"type", static_cast<int>(e.type)},
        {"seat_id", e.seat_id},
        {"ts", e.ts.toUTC().toString(Qt::ISODateWithMs)},
        {"extra", e.extra_json}
    };
}

} // namespace seatui

// 供 Qt 信号/槽在跨线程传递时使用
Q_DECLARE_METATYPE(seatui::SeatInfo)
Q_DECLARE_METATYPE(seatui::ZoneInfo)
Q_DECLARE_METATYPE(seatui::SeatEvent)
