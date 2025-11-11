#include <seatui/launcher/role_selector.hpp>
#include <seatui/student/student_window.hpp>
#include <seatui/admin/admin_window.hpp>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>



RoleSelector::RoleSelector(QWidget* parent) : QWidget(parent) {
    auto v = new QVBoxLayout(this);
    auto title = new QLabel(u8"请选择进入的端：", this);
    auto btnStudent = new QPushButton(u8"学生端", this);
    auto btnAdmin   = new QPushButton(u8"管理员端", this);

    title->setAlignment(Qt::AlignCenter);
    btnStudent->setMinimumHeight(48);
    btnAdmin->setMinimumHeight(48);

    v->addWidget(title);
    v->addWidget(btnStudent);
    v->addWidget(btnAdmin);
    v->addStretch(1);

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
