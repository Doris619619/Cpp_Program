// main.cpp
#include <QApplication>
#include <QCoreApplication>
#include <QFont>

#include <seatui/launcher/login_window.hpp>

#include <QGuiApplication>  // ← 必须：提供 primaryScreen() 的声明
#include <QScreen>

// 高分屏与位图设置（需在 QApplication 前设置）
static void initHiDpi() {
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
}

// 全局样式（卡片风 + 主按钮 + 弹框与之统一）
static void applyGlobalStyle(QApplication& app) {
    // 字体（Windows 建议雅黑）
    QFont f("Microsoft YaHei UI");
    f.setPointSize(10);
    app.setFont(f);

    // 统一主题 QSS
    const char* qss = R"(
        QWidget { font-family: "Microsoft YaHei UI"; font-size: 10pt; }
        QMainWindow { background: #f5f7fa; }

        /* 卡片容器 */
        QFrame#card {
            background: #ffffff;
            border-radius: 16px;
            border: 1px solid #e6e8eb;
        }

        /* 标签提示色 */
        QLabel[hint="true"] { color: #8a8f98; }

        /* 行编辑 */
        QLineEdit {
            background: #ffffff;
            border: 1px solid #d7dbe3;
            border-radius: 10px;
            padding: 8px 12px;
            selection-background-color: #0ea5e9;
            selection-color: #ffffff;
        }
        QLineEdit:focus {
            border: 1px solid #0ea5e9;
            box-shadow: 0 0 0 3px rgba(14,165,233,0.15);
        }

        /* 主按钮 */
        QPushButton[type="primary"] {
            background: #0ea5e9;
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 10px 16px;
        }
        QPushButton[type="primary"]:hover { background: #0284c7; }
        QPushButton[type="primary"]:pressed { background: #0369a1; }

        /* 大卡片按钮（角色选择） */
        QPushButton[class="cardBtn"] {
            background: #ffffff;
            border: 1px solid #d7dbe3;
            border-radius: 14px;
            padding: 14px 18px;
            text-align: left;
        }
        QPushButton[class="cardBtn"]:hover {
            border-color: #0ea5e9;
            box-shadow: 0 4px 16px rgba(2,132,199,0.15);
        }

        /* 顶部条 */
        QFrame#header { background: #ffffff; border-bottom: 1px solid #e6e8eb; }

        /* —— 弹框与卡片风一致 —— */
        QMessageBox {
            background: #ffffff;
            border: 1px solid #e6e8eb;
            border-radius: 12px;
        }
        QMessageBox QLabel { color: #1f2937; min-width: 280px; }
        QMessageBox QPushButton {
            min-height: 32px;
            padding: 6px 12px;
            border-radius: 8px;
            border: 1px solid #d7dbe3;
            background: #ffffff;
        }
        QMessageBox QPushButton:hover {
            border-color: #0ea5e9;
            box-shadow: 0 2px 8px rgba(2,132,199,0.15);
        }
        QMessageBox QPushButton[type="primary"] {
            background: #0ea5e9; color: #ffffff; border: none;
        }
        QMessageBox QPushButton[type="primary"]:hover { background: #0284c7; }
        QMessageBox QPushButton[type="primary"]:pressed { background: #0369a1; }
    )";
    app.setStyleSheet(qss);
}



int main(int argc, char *argv[]) {
    initHiDpi();
    QApplication app(argc, argv);

    // 建议改为 true：当最后一个窗口关闭时退出应用
    app.setQuitOnLastWindowClosed(true);

    applyGlobalStyle(app);

    // —— 仅此处改变：一次创建两个“无角色限制”的登录窗口 ——
    auto *login1 = new LoginWindow(nullptr);
    login1->setAttribute(Qt::WA_DeleteOnClose);
    login1->setWindowTitle(QStringLiteral("登录窗口 #1"));
    login1->show();

    auto *login2 = new LoginWindow(nullptr);
    login2->setAttribute(Qt::WA_DeleteOnClose);
    login2->setWindowTitle(QStringLiteral("登录窗口 #2"));
    login2->show();

    // 可选：把两个窗口左右排开，测试更直观
    if (auto *scr = QGuiApplication::primaryScreen()) {
        const QRect r = scr->availableGeometry();
        const int w = 520;  // 按你的实际登录窗大小微调
        const int h = 380;
        login1->setGeometry(r.center().x() - w - 16, r.center().y() - h/2, w, h);
        login2->setGeometry(r.center().x() + 16,     r.center().y() - h/2, w, h);
    }

    return app.exec();
}
