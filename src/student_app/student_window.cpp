#include <seatui/student/student_window.hpp>

#include <QLabel>
#include <QVBoxLayout>
#include <QWidget>

StudentWindow::StudentWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(u8"SeatUI 学生端（预览版）");
    auto central = new QWidget(this);
    auto v = new QVBoxLayout(central);
    v->addWidget(new QLabel(u8"这里将显示：实时座位图 + 举报入口（后续接入 seat_provider_dummy ）", this));
    setCentralWidget(central);
}
