#pragma once

#include <QObject>
#include <QWidget>
#include <QImage>
#include <QVector>
#include <QPoint>
#include <QRect>
#include <QPaintEvent>
#include <QResizeEvent>

class QPainter;

class NavigationCanvas : public QWidget {
    Q_OBJECT
public:
    explicit NavigationCanvas(QWidget* parent = nullptr);
    void setSuperSample(bool on);

protected:
    void paintEvent(QPaintEvent*) override;
    void resizeEvent(QResizeEvent*) override;

private:
    struct Layout {
        int   W = 0, H = 0;
        int   margin = 0;
        QRect rectInner;              // 主绘制区域
        QVector<QPoint> shelfCenters; // 顶部 A/B/C/D 中心
        QPoint startPt;               // 起点（对齐到网格）
    } lay;

    // —— 网格与座位参数 —— //
    int   m_cell        = 15;   // 主网格像素（可微调：18~26）
    int   m_seatCols    = 4;    // 座位横向格数
    int   m_seatRows    = 2;    // 座位纵向格数
    int   m_aisleXCells = 1;    // 座位间横向普通走道宽度（格）
    int   m_aisleYCells = 1;    // 座位间纵向普通走道宽度（格）
    int   m_mainEvery   = 4;    // 每多少个座位插入一次主走道
    int   m_mainWCells  = 2;    // 主走道宽度（格）
    int   m_topReserve  = 3;    // 顶部为书架标签预留的高度（格）
    int   m_borderCells = 1;    // 四周边框留白（格）
    qreal m_seatGap     = 2.0;  // 座位矩形像素级内缩
    qreal m_seatRadius  = 8.0;  // 座位圆角

    bool  useSSAA = true;
    QImage msaaBuffer;

private:
    // —— 绘制 —— //
    void updateLayout(int W, int H);
    void drawScene(QPainter& p, const QSize& sz);
    void drawBackgroundGrid(QPainter& p); // 只画主网格（含加粗的 4 格分界）
    void drawShelvesLabels(QPainter& p);
    void drawSeats(QPainter& p);          // 4×2 座位，留出走道
    void drawStartMark(QPainter& p);

    // —— 辅助 —— //
    inline int snapUp(int v, int s)  const { return ((v + s - 1) / s) * s; }
    inline int snapDn(int v, int s)  const { return  (v / s) * s; }
};
