#include <seatui/launcher/role_selector.hpp>
#include <seatui/student/student_window.hpp>
#include <seatui/admin/admin_window.hpp>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>

RoleSelector::RoleSelector(QWidget* parent) : QWidget(parent) {   // ← 注意：没有返回类型！
    auto v = new QVBoxLayout(this);
    v->setContentsMargins(24, 24, 24, 24);
    v->setSpacing(14);

    auto header = new QLabel(u8"请选择进入的端", this);
    header->setStyleSheet("font-size:16pt; font-weight:600;");
    v->addWidget(header);

    auto line = new QLabel(u8"学生端适合选座/举报；管理员端用于监控/统计", this);
    line->setProperty("hint", true);
    v->addWidget(line);

    auto row = new QHBoxLayout();
    row->setSpacing(16);

    auto btnStudent = new QPushButton(u8"学生端\n— 选座 / 举报 / 我的记录", this);
    btnStudent->setMinimumSize(260, 120);
    btnStudent->setProperty("class", "cardBtn");               // 用 property 给 QSS 匹配
    btnStudent->setCursor(Qt::PointingHandCursor);

    auto btnAdmin   = new QPushButton(u8"管理员端\n— 实时热力图 / 告警中心 / 统计图", this);
    btnAdmin->setMinimumSize(260, 120);
    btnAdmin->setProperty("class", "cardBtn");
    btnAdmin->setCursor(Qt::PointingHandCursor);

    row->addWidget(btnStudent, 1);
    row->addWidget(btnAdmin,   1);
    v->addLayout(row);
    v->addStretch();

    connect(btnStudent, &QPushButton::clicked, this, [this]() {
        auto* w = new StudentWindow();
        w->setAttribute(Qt::WA_DeleteOnClose);
        w->show();
        emit openStudent();
    });
    connect(btnAdmin, &QPushButton::clicked, this, [this]() {
        auto* w = new AdminWindow();
        w->setAttribute(Qt::WA_DeleteOnClose);
        w->show();
        emit openAdmin();
    });
}
