#pragma once
#include <QMainWindow>
#include <QTextEdit>
#include <QtWebSockets/QWebSocket>

class QComboBox;
class QPushButton;
class QLabel;
class QWidget;
class QStackedWidget;

class StudentWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit StudentWindow(QWidget* parent = nullptr);

signals:
    // 侧边栏“← 返回登录”被点击时发出，交由上层（RoleSelector/Main）处理切回登录
    void backToLoginRequested();

private slots:
    // 导航页
    void onGenerate();   // 生成路径
    void onClear();      // 清除
    // 侧边栏“返回登录”
    void onBackToLogin();

    // 页面切换
    void gotoDashboard();
    void gotoNavigation();
    void gotoHeatmap();

private:
    // ===== 通用 =====
    QStackedWidget* pages = nullptr;                 // 右侧多页面容器
    QPushButton *btnDash = nullptr, *btnNav = nullptr, *btnHeat = nullptr;

    // ===== 导航页控件 =====
    QWidget*     navCanvas = nullptr;               // 地图画布占位（后续绘制路径）
    QComboBox*   destBox   = nullptr;               // 目标书架 A/B/C/D
    QPushButton* btnGen    = nullptr;               // 生成路径
    QPushButton* btnClear  = nullptr;               // 清除
    QLabel*      navStatus = nullptr;               // 状态提示

    // ===== 构建各页面 =====
    QWidget* buildDashboardPage();                  // 仪表盘主页
    QWidget* buildNavigationPage();                 // 导航页
    QWidget* buildHeatmapPage();                    // 热力图占位页






    // —— 一键求助：成员 —— //
    QPushButton *btnHelp = nullptr;
    QWidget *buildHelpPage();
    void gotoHelp();

    QTextEdit  *helpText_  = nullptr;
    QLabel     *helpImgPreview_ = nullptr;
    QPushButton *helpPickBtn_ = nullptr;
    QPushButton *helpSubmitBtn_ = nullptr;
    QPushButton *helpResetBtn_ = nullptr;

    QByteArray helpImgBytes_;    // PNG/JPEG 原始字节
    QString    helpImgFilename_; // 原始文件名
    QString    helpImgMime_;     // "image/png" ...

    // —— 一键求助：槽函数 —— //
    void onPickImage();
    void onSubmitHelp();
    void onResetHelp();


private:
    // —— WS 客户端 —— //
    void initWsClient();
    void wsSend(const QByteArray& utf8Json);
    QWebSocket* ws_ = nullptr;
    bool wsReady_ = false;


};
