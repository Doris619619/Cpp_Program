#include <seatui/student/navigation_canvas.hpp>
#include <QPainter>
#include <QLinearGradient>
#include <QFont>
#include <QImage>
#include <QtMath>

NavigationCanvas::NavigationCanvas(QWidget* parent) : QWidget(parent) {
    setMinimumSize(680, 440);
    setAutoFillBackground(false);
}

void NavigationCanvas::setSuperSample(bool on){
    if (useSSAA != on) { useSSAA = on; update(); }
}

void NavigationCanvas::resizeEvent(QResizeEvent*){
    updateLayout(width(), height());
    msaaBuffer = QImage(width()*2, height()*2, QImage::Format_RGBA8888);
    msaaBuffer.fill(Qt::transparent);
}

void NavigationCanvas::updateLayout(int W, int H){
    lay.W = W; lay.H = H;
    lay.margin = int(0.06 * qMin(W, H));
    lay.rectInner = QRect(lay.margin, lay.margin, W - 2*lay.margin, H - 2*lay.margin);

    // 顶部四个书架 A/B/C/D
    const int n = 4;
    const int colW = lay.rectInner.width() / n;
    lay.shelfCenters.clear();
    for (int k = 0; k < n; ++k) {
        const int x = lay.rectInner.left() + colW * k + colW / 2;
        const int y = lay.rectInner.top() + int(0.12 * lay.rectInner.height());
        lay.shelfCenters.push_back(QPoint(x, y));
    }

    // START：右下角、对齐网格、离边各 2 格
    const int cell = m_cell;
    const int rx = snapDn(lay.rectInner.right(), cell)  - m_borderCells * cell - 2*cell;
    const int ry = snapDn(lay.rectInner.bottom(), cell) - m_borderCells * cell - 2*cell;
    lay.startPt = QPoint(rx, ry);
}

void NavigationCanvas::drawBackgroundGrid(QPainter& p){
    // 背景渐变
    QLinearGradient g(0, 0, 0, lay.H);
    g.setColorAt(0.0, QColor(10,14,24));
    g.setColorAt(1.0, QColor(16,19,27));
    p.fillRect(QRect(0,0,lay.W,lay.H), g);

    p.setRenderHint(QPainter::Antialiasing, true);

    // —— 主区域双层边框：更清晰 —— //
    // 外圈（更明显）
    QPen outer(QColor(60, 80, 100, 180));
    outer.setWidthF(2.0);
    p.setPen(outer);
    p.setBrush(Qt::NoBrush);
    p.drawRoundedRect(lay.rectInner.adjusted(0.5, 0.5, -0.5, -0.5), 12, 12);

    // 内圈（细一点，形成内凹感）
    QPen inner(QColor(36,45,60, 160));
    inner.setWidthF(1.0);
    p.setPen(inner);
    p.drawRoundedRect(lay.rectInner.adjusted(6, 6, -6, -6), 10, 10);

    // —— 主网格：对齐 m_cell，并每 4 格一条稍亮的分区线 —— //
    const int cell = m_cell;
    // 网格范围贴着内圈边框：以 lay.rectInner 为准
    const int Lg = snapUp(lay.rectInner.left(),  cell);
    const int Tg = snapUp(lay.rectInner.top(),   cell);
    const int Rg = snapDn(lay.rectInner.right(), cell);
    const int Bg = snapDn(lay.rectInner.bottom(),cell);

    QPen minorPen(QColor(70, 86,108, 70));  minorPen.setWidth(1);
    QPen majorPen(QColor(100,120,140,120)); majorPen.setWidth(1);

    for (int x = Lg, idx = 0; x <= Rg; x += cell, ++idx) {
        p.setPen(idx % 4 == 0 ? majorPen : minorPen);
        p.drawLine(x, Tg, x, Bg);
    }
    for (int y = Tg, idy = 0; y <= Bg; y += cell, ++idy) {
        p.setPen(idy % 4 == 0 ? majorPen : minorPen);
        p.drawLine(Lg, y, Rg, y);
    }
}


void NavigationCanvas::drawShelvesLabels(QPainter& p){
    const int cell = m_cell;

    // 统一尺寸：宽 4 格，高 2 格；完全贴网格
    const int badgeW = 4 * cell;
    const int badgeH = 2 * cell;

    // 纵向位置：在内圈网格顶部往下 1 格（不挡座位），并贴网格
    const int Tg = snapUp(lay.rectInner.top(), cell);
    const int y  = Tg + cell; // 下移 1 格

    const QString labels[4] = {u8"A",u8"B",u8"C",u8"D"};
    const int n = 4;
    const int colW = lay.rectInner.width() / n;

    for (int i = 0; i < n; ++i) {
        // 以分栏中心为基准，再对齐到网格，并确保不越界
        const int cx = lay.rectInner.left() + colW * i + colW / 2;
        int left = snapDn(cx - badgeW/2, cell);
        // clamp 到边界内（左右各留 2px 缓冲）
        left = qMax(left, lay.rectInner.left() + 2);
        if (left + badgeW > lay.rectInner.right() - 2)
            left = lay.rectInner.right() - 2 - badgeW;

        QRect r(left, y, badgeW, badgeH);

        // 背板
        p.setPen(QPen(QColor(55,67,85), 1));
        p.setBrush(QColor(20,30,48,220));
        p.drawRoundedRect(r, 10, 10);

        // 文本
        p.setPen(QColor(224,229,236));
        QFont f = p.font(); f.setBold(true);
        f.setPointSizeF(qMax(10.0, lay.H*0.022));
        p.setFont(f);
        p.drawText(r, Qt::AlignCenter, labels[i]);
    }
}


void NavigationCanvas::drawSeats(QPainter& p){
    const int cell = m_cell;

    // —— 可铺设的网格区域（对齐网格），顶部为标签预留 m_topReserve 格 —— //
    const int Lg = snapUp(lay.rectInner.left(),  cell);
    const int Tg = snapUp(lay.rectInner.top(),   cell);
    const int Rg = snapDn(lay.rectInner.right(), cell);
    const int Bg = snapDn(lay.rectInner.bottom(),cell);

    const int L_area = Lg;                           // 网格对齐的左界
    const int T_area = Tg + m_topReserve * cell;     // 顶部留白后
    const int R_area = Rg;
    const int B_area = Bg;

    // —— 座位尺寸与步进 —— //
    const int seatW = m_seatCols * cell;             // 4 格
    const int seatH = m_seatRows * cell;             // 2 格
    const int stepX = seatW + m_aisleXCells * cell;  // 普通过道
    const int stepY = seatH + m_aisleYCells * cell;

    // 简化的座位布局算法
    // 1) 计算可用的总宽度和高度
    const int availableWidth = R_area - L_area - 2;  // 减去右边预留的2px
    const int availableHeight = B_area - T_area - 2; // 减去底部预留的2px

    // 2) 计算可以容纳的列数和行数
    const int maxCols = (availableWidth + m_aisleXCells * cell) / (stepX);
    const int maxRows = (availableHeight + m_aisleYCells * cell) / (stepY);

    // 确保至少1列1行
    const int numCols = qMax(1, maxCols);
    const int numRows = qMax(1, maxRows);

    // 3) 计算总内容宽度和高度
    const int totalContentWidth = numCols * stepX - m_aisleXCells * cell;
    const int totalContentHeight = numRows * stepY - m_aisleYCells * cell;

    // 4) 计算起始位置，确保水平居中且最左边的座位距离左边框2px
    const int startX = L_area + 2;  // 左边距2px
    const int startY = T_area + (availableHeight - totalContentHeight) / 2; // 垂直居中

    // 5) 生成所有座位位置
    QVector<int> xs, ys;
    for (int col = 0; col < numCols; ++col) {
        xs.push_back(startX + col * stepX);
    }
    for (int row = 0; row < numRows; ++row) {
        ys.push_back(startY + row * stepY);
    }

    // 删除最左边和最右边的一列
    if (xs.size() > 2) {
        xs.removeFirst();  // 删除最左边一列
        xs.removeLast();   // 删除最右边一列
    }

    if (xs.isEmpty() || ys.isEmpty()) return;

    // —— 绘制座位 —— //
    QPen seatPen(QColor(130, 160, 180, 110));
    seatPen.setWidthF(1.2);

    for (int y : ys) {
        for (int x : xs) {
            QRectF seatRect(x + m_seatGap, y + m_seatGap,
                            seatW - 2*m_seatGap, seatH - 2*m_seatGap);

            p.setPen(Qt::NoPen);
            p.setBrush(QColor(26,34,48,220));
            p.drawRoundedRect(seatRect, m_seatRadius, m_seatRadius);

            p.setBrush(Qt::NoBrush);
            p.setPen(seatPen);
            p.drawRoundedRect(seatRect, m_seatRadius, m_seatRadius);
        }
    }
}


void NavigationCanvas::drawStartMark(QPainter& p){
    p.setBrush(QColor(56,189,248));
    p.setPen(Qt::NoPen);
    const int r = int(0.012 * qMin(lay.W, lay.H));
    p.drawEllipse(lay.startPt, r, r);

    p.setPen(QColor(200,225,240));
    QFont f = p.font(); f.setBold(true); f.setPointSizeF(qMax(9.0, lay.H*0.018)); p.setFont(f);
    p.drawText(lay.startPt + QPoint(12, -6), "START");
}

void NavigationCanvas::drawScene(QPainter& p, const QSize&){
    drawBackgroundGrid(p);  // 主网格（带每 4 格一条分界线）
    drawShelvesLabels(p);   // A/B/C/D
    drawSeats(p);           // 4×2 座位 + 走道
    drawStartMark(p);       // 右下角对齐到网格
}

void NavigationCanvas::paintEvent(QPaintEvent*){
    if (useSSAA) {
        if (msaaBuffer.size() != QSize(width()*2, height()*2))
            msaaBuffer = QImage(width()*2, height()*2, QImage::Format_RGBA8888);
        msaaBuffer.fill(Qt::transparent);

        QPainter pm(&msaaBuffer);
        pm.setRenderHint(QPainter::Antialiasing, true);
        pm.scale(2.0, 2.0);
        drawScene(pm, QSize(width(), height()));
        pm.end();

        QPainter p(this);
        p.setRenderHint(QPainter::SmoothPixmapTransform, true);
        p.drawImage(rect(), msaaBuffer);
    } else {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, true);
        drawScene(p, size());
    }
}
