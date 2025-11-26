#pragma once
#include <QMainWindow>
#include <QtWebSockets/QWebSocketServer>
#include <QtWebSockets/QWebSocket>
#include <QList>

class QTabWidget; class QTableWidget; class QLabel; class QPushButton;

class AdminWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit AdminWindow(QWidget* parent=nullptr);

    // 供 WS/DB 调用：学生求助 JSON 到达
    // 传入 UTF-8 字节串，如：{"type":"student_help", "description":"...", "image":{...}, "created_at":"..."}
    Q_SLOT void onHelpArrived(const QByteArray& utf8Json);

private:
    QWidget* buildOverviewPage();
    QWidget* buildHelpCenterPage();
    QWidget* buildHeatmapPage();
    QWidget* buildStatsPage();
    QWidget* buildTimelinePage();

    void appendHelpRow(const QString& when, const QString& user,
                       const QString& text, const QPixmap& thumb,
                       const QByteArray& rawImgBase64, const QString& mime);

private:
    QTabWidget* tabs_ = nullptr;

    // 求助中心
    QTableWidget* helpTable_ = nullptr;



    // —— WebSocket 服务端 —— //
    void initWsServer();
    QWebSocketServer* wsServer_ = nullptr;
    QList<QWebSocket*> wsClients_ = {};
};
