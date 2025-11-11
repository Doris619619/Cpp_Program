#include <seatui/admin/admin_window.hpp>

#include <QLabel>
#include <QVBoxLayout>
#include <QWidget>

AdminWindow::AdminWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(u8"SeatUI 管理端（预览版）");
    auto central = new QWidget(this);
    auto v = new QVBoxLayout(central);
    v->addWidget(new QLabel(u8"这里将显示：热力图/统计图/告警中心/时间轴（后续接入 QtCharts + heatmap ）", this));
    setCentralWidget(central);
}
