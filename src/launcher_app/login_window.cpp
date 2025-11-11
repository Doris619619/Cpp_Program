#include <seatui/launcher/login_window.hpp>
#include <seatui/launcher/role_selector.hpp>
#include <QStackedWidget>
#include <QWidget>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>

LoginWindow::LoginWindow(QWidget* parent)
    : QMainWindow(parent), stacked_(new QStackedWidget(this)),
    user_(nullptr), pass_(nullptr), msg_(nullptr) {
    setWindowTitle(u8"SeatUI 登录");
    setCentralWidget(stacked_);
    stacked_->addWidget(buildLoginPage()); // index 0
    stacked_->addWidget(buildRolePage());  // index 1
}

QWidget* LoginWindow::buildLoginPage() {
    auto page = new QWidget(this);
    auto v = new QVBoxLayout(page);
    auto form = new QFormLayout();

    user_ = new QLineEdit(page);
    pass_ = new QLineEdit(page);
    pass_->setEchoMode(QLineEdit::Password);
    form->addRow(u8"用户名：", user_);
    form->addRow(u8"密  码：", pass_);

    auto btn = new QPushButton(u8"登录", page);
    msg_ = new QLabel(page);
    msg_->setStyleSheet("color:#b71c1c;"); // 红色提示

    v->addLayout(form);
    v->addWidget(btn);
    v->addWidget(msg_);
    v->addStretch(1);

    connect(btn, &QPushButton::clicked, this, &LoginWindow::onLoginClicked);
    return page;
}

QWidget* LoginWindow::buildRolePage() {
    // 角色选择单独做成一个小部件（两个大按钮）
    auto role = new RoleSelector(this);

    // 选择后弹对应主窗口（注意：本窗口不关闭，便于回退与测试）
    connect(role, &RoleSelector::openStudent, this, [this]() {
        // 懒加载到 RoleSelector 内部实现里完成（见 role_selector.cpp）
    });
    connect(role, &RoleSelector::openAdmin, this, [this]() {
        // 同上
    });

    return role;
}

void LoginWindow::onLoginClicked() {
    const auto u = user_->text().trimmed();
    const auto p = pass_->text();

    // 极简校验：均不为空就“通过”；后续再接 DB/后端/WS
    if (u.isEmpty() || p.isEmpty()) {
        msg_->setText(u8"用户名或密码不能为空。");
        return;
    }
    msg_->clear();
    stacked_->setCurrentIndex(1); // 进入角色选择页
}
