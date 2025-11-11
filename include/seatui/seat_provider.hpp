#pragma once
// UI 的数据提供抽象接口（支持信号/槽异步更新）

#include "models.hpp"
#include <QtCore/QObject>

namespace seatui {

class ISeatProvider : public QObject {
    Q_OBJECT
public:
    explicit ISeatProvider(QObject* parent=nullptr) : QObject(parent) {}
    ~ISeatProvider() override = default;

    // 即时快照（UI 初始渲染时调用）
    virtual std::vector<SeatInfo> seats() const = 0;
    virtual std::vector<ZoneInfo> zones() const = 0;

signals:
    // 增量更新（子线程 emit，Qt 自动排队到 UI 线程）
    void seatUpdated(const seatui::SeatInfo& seat);
    void zoneUpdated(const seatui::ZoneInfo& zone);
    void eventRaised(const seatui::SeatEvent& evt);

    // 可选：热力图缓存就绪、快照图片就绪等
    void heatmapReady(const QImage& img); // 若实现了离线生成热力图
};

} // namespace seatui
