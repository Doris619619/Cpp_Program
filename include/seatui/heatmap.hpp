#pragma once
// 热力图工具：把“带权样本点”渲染为半透明 QImage，可叠加到底图上

#include <QtGui/QImage>
#include <QtGui/QPainter>
#include <QtGui/QColor>
#include <QtCore/QPointF>
#include <vector>
#include <cmath>
#include <algorithm>

namespace seatui {

struct HeatSample {
    QPointF pos;   // 像素坐标（与底图同一坐标系）
    double  w;     // 权重（强度），建议 0~1
};

// 颜色映射：t∈[0,1] -> RGBA（半透明）
// 蓝(0)->绿(0.5)->黄(0.75)->红(1)
inline QColor heatColor(double t, int alpha = 160) {
    t = std::clamp(t, 0.0, 1.0);
    if (t < 0.5) {
        double k = t / 0.5;            // 0..1
        int r = int(0   * (1-k) +  0 * k);
        int g = int(0   * (1-k) +180 * k);
        int b = int(200 * (1-k) +  0 * k);
        return QColor(r,g,b, alpha);
    } else if (t < 0.75) {
        double k = (t - 0.5) / 0.25;
        int r = int(0   * (1-k) +220 * k);
        int g = 180;
        int b = 0;
        return QColor(r,g,b, alpha);
    } else {
        double k = (t - 0.75) / 0.25;
        int r = 220;
        int g = int(180 * (1-k) + 20 * k);
        int b = int(0   * (1-k) + 60 * k);
        return QColor(r,g,b, alpha);
    }
}

// 生成灰度强度图（单通道）：高斯扩散
// size: 输出图尺寸；radiusPx: 高斯半径像素；normalize: 是否把最大值归一化到 1
inline QImage generateHeatGray(
    const QSize& size,
    const std::vector<HeatSample>& samples,
    double radiusPx = 40.0,
    bool normalize = true
    ) {
    QImage gray(size, QImage::Format_Alpha8);
    gray.fill(0);

    if (samples.empty() || size.isEmpty() || radiusPx <= 1.0) return gray;

    // 预计算一维高斯核（近似圆形）
    const int r = int(std::ceil(radiusPx));
    const double sigma = radiusPx / 2.0;
    std::vector<double> kernel(2*r+1);
    for (int i=-r; i<=r; ++i) {
        double v = std::exp(-(i*i)/(2*sigma*sigma));
        kernel[i+r] = v;
    }

    // 累加强度
    std::vector<double> buf(size.width()*size.height(), 0.0);
    auto addSample = [&](const HeatSample& s){
        const int cx = int(std::round(s.pos.x()));
        const int cy = int(std::round(s.pos.y()));
        const double w = s.w;
        if (w<=0) return;

        for (int y = cy - r; y <= cy + r; ++y) {
            if (y < 0 || y >= size.height()) continue;
            int ky = std::abs(y - cy);
            double wy = kernel[ky];
            for (int x = cx - r; x <= cx + r; ++x) {
                if (x < 0 || x >= size.width()) continue;
                int kx = std::abs(x - cx);
                double wx = kernel[kx];
                double val = w * wx * wy;
                buf[y*size.width() + x] += val;
            }
        }
    };
    for (const auto& s : samples) addSample(s);

    // 归一化到 0..255
    double maxv = 0.0;
    if (normalize) {
        for (double v : buf) maxv = std::max(maxv, v);
        if (maxv <= 1e-12) maxv = 1.0;
    } else {
        maxv = 1.0;
    }
    for (int y=0; y<size.height(); ++y) {
        uchar* line = gray.scanLine(y);
        for (int x=0; x<size.width(); ++x) {
            double v = buf[y*size.width()+x] / maxv;
            int iv = int(std::clamp(v, 0.0, 1.0) * 255.0);
            line[x] = static_cast<uchar>(iv);
        }
    }
    return gray;
}

// 把灰度强度图染色为 RGBA 热力图（半透明）
inline QImage colorizeHeat(const QImage& gray, int alpha = 160) {
    QImage out(gray.size(), QImage::Format_ARGB32_Premultiplied);
    for (int y=0; y<gray.height(); ++y) {
        const uchar* g = gray.constScanLine(y);
        QRgb* dst = reinterpret_cast<QRgb*>(out.scanLine(y));
        for (int x=0; x<gray.width(); ++x) {
            double t = g[x] / 255.0;
            QColor c = heatColor(t, int(alpha * t)); // 强度越高越不透明
            dst[x] = c.rgba();
        }
    }
    return out;
}

// 一步到位：样本点 -> 彩色热力图
inline QImage makeHeatmap(
    const QSize& size,
    const std::vector<HeatSample>& samples,
    double radiusPx = 40.0
    ) {
    return colorizeHeat(generateHeatGray(size, samples, radiusPx, /*normalize*/true));
}

} // namespace seatui
