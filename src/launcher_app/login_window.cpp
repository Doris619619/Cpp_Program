#include <seatui/launcher/login_window.hpp>
#include <seatui/launcher/role_selector.hpp>
#include <seatui/student/student_window.hpp>
#include <seatui/admin/admin_window.hpp>
#include <seatui/widgets/card_dialog.hpp>
#include <QStackedWidget>
#include <QWidget>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QFrame>
#include <QHBoxLayout>
#include <QAction>
#include <QCheckBox>
#include <QGraphicsDropShadowEffect>
#include <QColor>
#include <QMessageBox>

#include <QDialog>
#include <QFrame>

#include <QGraphicsDropShadowEffect>
#include <QMouseEvent>

#include <QTimer>
#include <QGuiApplication>
#include <QScreen>
#include <QMessageBox>   // 顶部若还没包含，补上





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

    // 居中容器（左右留白）
    auto outer = new QVBoxLayout(page);
    outer->setContentsMargins(24, 24, 24, 24);
    outer->addStretch();

    // 卡片
    auto card = new QFrame(page);
    card->setObjectName("card");
    card->setMinimumWidth(420);
    auto effect = new QGraphicsDropShadowEffect(card);
    effect->setBlurRadius(24);
    effect->setOffset(0, 6);
    effect->setColor(QColor(0,0,0,40));
    card->setGraphicsEffect(effect);

    auto v = new QVBoxLayout(card);
    v->setContentsMargins(28, 28, 28, 28);
    v->setSpacing(14);

    auto title = new QLabel(u8"欢迎登录 SeatUI", card);
    title->setStyleSheet("font-size: 18pt; font-weight: 600;");

    auto sub = new QLabel(u8"请输入账号信息", card);
    sub->setProperty("hint", true);

    auto form = new QFormLayout();
    form->setLabelAlignment(Qt::AlignLeft);
    form->setFormAlignment(Qt::AlignVCenter);
    form->setHorizontalSpacing(16);
    form->setVerticalSpacing(10);

    user_ = new QLineEdit(card);
    user_->setPlaceholderText(u8"学号 / 工号");

    pass_ = new QLineEdit(card);
    pass_->setPlaceholderText(u8"密码");
    pass_->setEchoMode(QLineEdit::Password);

    // 密码显示/隐藏
    auto toggle = new QAction(u8"显示", pass_);
    toggle->setCheckable(true);
    pass_->addAction(toggle, QLineEdit::TrailingPosition);
    QObject::connect(toggle, &QAction::toggled, pass_, [this, toggle](bool on){
        pass_->setEchoMode(on ? QLineEdit::Normal : QLineEdit::Password);
        toggle->setText(on ? u8"隐藏" : u8"显示");
    });

    form->addRow(u8"用户名", user_);
    form->addRow(u8"密码",   pass_);

    auto row = new QHBoxLayout();
    auto remember = new QCheckBox(u8"记住我", card);
    row->addWidget(remember);
    row->addStretch();

    auto btn = new QPushButton(u8"登录", card);
    btn->setProperty("type", "primary");
    btn->setMinimumHeight(40);

    msg_ = new QLabel(card);
    msg_->setStyleSheet("color:#b71c1c;");

    v->addWidget(title);
    v->addWidget(sub);
    v->addSpacing(4);
    v->addLayout(form);
    v->addLayout(row);
    v->addSpacing(6);
    v->addWidget(btn);
    v->addWidget(msg_);

    outer->addWidget(card, 0, Qt::AlignHCenter);
    outer->addStretch();

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
    const QString u = user_->text().trimmed();
    const QString p = pass_->text();

    if (u.isEmpty() || p.isEmpty()) {
        msg_->setText(u8"用户名或密码不能为空。");
        return;
    }
    msg_->clear();

    struct Cred { const char* user; const char* pass; const char* role; };
    static const Cred CREDS[] = {
        {"student", "123456", "student"},
        {"admin",   "123456", "admin"}
    };

    for (const auto& c : CREDS) {
        if (u == QString::fromUtf8(c.user) && p == QString::fromUtf8(c.pass)) {
            if (QString::fromUtf8(c.role) == "student") {
                auto* w = new StudentWindow();
                w->setAttribute(Qt::WA_DeleteOnClose);
                w->show();
            } else {
                auto* w = new AdminWindow();
                w->setAttribute(Qt::WA_DeleteOnClose);
                w->show();
            }
            // 关键修改：下一拍再关登录窗，更稳
            QTimer::singleShot(0, this, [this]{ this->close(); });
            return;
        }
    }

    CardDialog(u8"登录失败", u8"用户名或密码错误。", this).exec();
}





