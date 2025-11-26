#include <QTabWidget>
#include <QTableWidget>
#include <QHeaderView>
#include <QVBoxLayout>
#include <QLabel>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QDateTime>
#include <QBuffer>
#include <QImageReader>
#include <QPushButton>
#include <QDialog>
#include <QScrollArea>
#include <QPixmap>
#include <QHBoxLayout>

#include <seatui/widgets/card_dialog.hpp>   // 复用你已有卡片弹框样式
#include <seatui/admin/admin_window.hpp>

#include <QtWebSockets/QWebSocketServer>
#include <QtWebSockets/QWebSocket>

AdminWindow::AdminWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(u8"SeatUI 管理端");
    resize(1100, 720);

    tabs_ = new QTabWidget(this);
    setCentralWidget(tabs_);

    tabs_->addTab(buildOverviewPage(),  u8"总览");
    tabs_->addTab(buildHelpCenterPage(),u8"求助中心");
    tabs_->addTab(buildHeatmapPage(),   u8"热力图");
    tabs_->addTab(buildStatsPage(),     u8"统计");
    tabs_->addTab(buildTimelinePage(),  u8"时间轴");

    initWsServer();
}

QWidget* AdminWindow::buildOverviewPage() {
    auto w = new QWidget(this);
    auto v = new QVBoxLayout(w);
    auto t = new QLabel(u8"这里展示关键 KPI（占位）：\n• 当前占用率\n• 今日异常数\n• 最近 1h 求助…", w);
    t->setStyleSheet("font-size:15px; color:#334155;");
    v->addWidget(t);
    v->addStretch();
    return w;
}

QWidget* AdminWindow::buildHelpCenterPage() {
    auto w = new QWidget(this);
    auto v = new QVBoxLayout(w);

    helpTable_ = new QTableWidget(w);
    helpTable_->setColumnCount(6);
    helpTable_->setHorizontalHeaderLabels({u8"时间(UTC)", u8"用户", u8"摘要", u8"缩略图", u8"MIME", u8"查看"});
    helpTable_->horizontalHeader()->setStretchLastSection(true);
    helpTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    helpTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    helpTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    helpTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    helpTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    v->addWidget(helpTable_);

    auto tip = new QLabel(u8"说明：学生端“一键求助”提交后，这里会出现一条记录；点击“查看”可看原图与全文。", w);
    tip->setStyleSheet("color:#64748b;");
    v->addWidget(tip);

    return w;
}

QWidget* AdminWindow::buildHeatmapPage() {
    auto w = new QWidget(this);
    auto v = new QVBoxLayout(w);
    v->addWidget(new QLabel(u8"🔥 热力图占位（后续接入 QtCharts/自绘 QImage 叠加）", w));
    v->addStretch();
    return w;
}

QWidget* AdminWindow::buildStatsPage() {
    auto w = new QWidget(this);
    auto v = new QVBoxLayout(w);
    v->addWidget(new QLabel(u8"📊 统计图占位（占用率/分区对比/小时聚合等）", w));
    v->addStretch();
    return w;
}

QWidget* AdminWindow::buildTimelinePage() {
    auto w = new QWidget(this);
    auto v = new QVBoxLayout(w);
    v->addWidget(new QLabel(u8"⏱ 时间轴/事件回放占位", w));
    v->addStretch();
    return w;
}

void AdminWindow::appendHelpRow(const QString& when, const QString& user,
                                const QString& text, const QPixmap& thumb,
                                const QByteArray& rawImgBase64, const QString& mime)
{
    const int r = helpTable_->rowCount();
    helpTable_->insertRow(r);

    auto *itemWhen = new QTableWidgetItem(when);
    auto *itemUser = new QTableWidgetItem(user);
    auto *itemSumm = new QTableWidgetItem(text.left(48) + (text.size()>48?QStringLiteral("…"):QString()));
    auto *itemMime = new QTableWidgetItem(mime);

    // 缩略图
    auto *thumbLbl = new QLabel();
    thumbLbl->setPixmap(thumb.scaled(80, 50, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    thumbLbl->setAlignment(Qt::AlignCenter);

    // 查看按钮
    auto *btn = new QPushButton(u8"查看");
    btn->setProperty("type","primary");

    helpTable_->setItem(r, 0, itemWhen);
    helpTable_->setItem(r, 1, itemUser);
    helpTable_->setItem(r, 2, itemSumm);
    helpTable_->setCellWidget(r, 3, thumbLbl);
    helpTable_->setItem(r, 4, itemMime);
    helpTable_->setCellWidget(r, 5, btn);

    // 弹窗预览：原图 + 全文
    connect(btn, &QPushButton::clicked, this, [=]{
        QDialog dlg(this);
        dlg.setWindowTitle(u8"求助详情");
        auto v = new QVBoxLayout(&dlg);
        auto info = new QLabel(QString(u8"时间：%1\n用户：%2\nMIME：%3\n\n描述：\n%4")
                                   .arg(when, user, mime, text), &dlg);
        info->setWordWrap(true);
        v->addWidget(info);

        if (!rawImgBase64.isEmpty()) {
            QByteArray imgBytes = QByteArray::fromBase64(rawImgBase64);
            QPixmap px; px.loadFromData(imgBytes);
            auto area = new QScrollArea(&dlg);
            auto imgL = new QLabel();
            imgL->setPixmap(px);
            area->setWidget(imgL);
            area->setWidgetResizable(true);
            area->setMinimumSize(640, 380);
            v->addWidget(area, 1);
        }

        auto ok = new QPushButton(u8"知道了", &dlg);
        ok->setProperty("type","primary");
        auto h = new QHBoxLayout(); h->addStretch(); h->addWidget(ok);
        v->addLayout(h);
        connect(ok, &QPushButton::clicked, &dlg, &QDialog::accept);
        dlg.exec();
    });
}

void AdminWindow::onHelpArrived(const QByteArray& utf8Json) {
    // 解析 JSON（兼容无图/无用户名）
    QJsonParseError er; QJsonDocument d = QJsonDocument::fromJson(utf8Json, &er);
    if (er.error != QJsonParseError::NoError || !d.isObject()) {
        CardDialog(u8"解析失败", u8"收到的求助 JSON 无法解析。", this).exec();
        return;
    }
    const QJsonObject o = d.object();
    if (o.value("type").toString() != "student_help") return;

    const QString when = o.value("created_at").toString();
    const QString user = o.value("user").toString("student");
    const QString text = o.value("description").toString();

    // 缩略图
    QPixmap th; QByteArray rawB64; QString mime = "image/png";
    if (o.contains("image") && o.value("image").isObject()) {
        const QJsonObject im = o.value("image").toObject();
        rawB64 = im.value("base64").toString().toLatin1();
        mime   = im.value("mime").toString("image/png");
        QByteArray bytes = QByteArray::fromBase64(rawB64);
        th.loadFromData(bytes);
    }
    if (th.isNull()) th = QPixmap(80,50); // 无图给灰底
    if (th.isNull()) th.fill(QColor(230,235,240));

    appendHelpRow(when, user, text, th, rawB64, mime);
}

void AdminWindow::initWsServer() {
    wsServer_ = new QWebSocketServer(QStringLiteral("SeatUI-Admin-WS"),
                                     QWebSocketServer::NonSecureMode, this);
    const QHostAddress host = QHostAddress::LocalHost;  // 127.0.0.1
    const quint16 port = 12345;
    if (!wsServer_->listen(host, port)) {
        CardDialog(u8"WS 启动失败",
                   u8"管理员端 WebSocket 服务器监听失败（127.0.0.1:12345）。", this).exec();
        return;
    }

    connect(wsServer_, &QWebSocketServer::newConnection, this, [this]{
        auto *sock = wsServer_->nextPendingConnection();
        wsClients_ << sock;

        // 学生端连上后可能先发一条 hello，这里统一接入 onHelpArrived
        connect(sock, &QWebSocket::textMessageReceived, this, [this](const QString& msg){
            onHelpArrived(msg.toUtf8());                 // 直接复用你现有解析与入表
        });
        connect(sock, &QWebSocket::disconnected, this, [this, sock]{
            wsClients_.removeAll(sock);
            sock->deleteLater();
        });

        // 可选：欢迎语
        sock->sendTextMessage(QStringLiteral(R"({"type":"hello","role":"admin"})"));
    });
}

