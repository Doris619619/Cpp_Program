#pragma once
// UI 层简单事件总线（跨模块通知：选择座位、显示横幅、提交举报等）

#include <QtCore/QObject>
#include <QtCore/QString>
#include <QtGui/QImage>
#include "models.hpp"
#include "net_types.hpp"

namespace seatui {

class EventBus : public QObject {
    Q_OBJECT
public:
    static EventBus& instance() {
        static EventBus bus;
        return bus;
    }

signals:
    // UI 行为事件
    void seatSelected(int seat_id);
    void showBanner(const QString& text, int ms = 2500);
    void screenshotCaptured(const QImage& img);

    // 学生端举报从 UI 发起，后端/WS 客户端去发送
    void reportSubmitted(const seatui::ReportPayload& payload);

    // 管理端控制（例如静默告警）
    void muteAlertsRequested(int minutes);

private:
    explicit EventBus(QObject* parent=nullptr) : QObject(parent) {}
    Q_DISABLE_COPY_MOVE(EventBus)
};

} // namespace seatui
