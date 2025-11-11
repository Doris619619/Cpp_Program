#pragma once
// 统计图封装（Qt 6 Charts 快速生成 Line/Bar/Pie）

#include <QtCharts/QChart>
#include <QtCharts/QChartView>
#include <QtCharts/QBarSeries>
#include <QtCharts/QBarSet>
#include <QtCharts/QBarCategoryAxis>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QtCharts/QPieSeries>
#include <QtCharts/QPieSlice>

#include <QtGui/QPainter>
#include <QtCore/QVector>
#include <QtCore/QPointF>
#include <utility>
#include <vector>

namespace seatui { namespace charts {

// 折线图：data = [(x,y)...]
inline QChartView* makeLineChart(
    const QString& title,
    const QVector<QPointF>& data,
    const QString& xLabel = {},
    const QString& yLabel = {}
    ){
    auto* series = new QLineSeries();
    series->append(data);

    auto* chart = new QChart();
    chart->addSeries(series);
    chart->createDefaultAxes();
    chart->setTitle(title);
    chart->legend()->hide();

    const auto axesH = chart->axes(Qt::Horizontal);
    const auto axesV = chart->axes(Qt::Vertical);
    if (!axesH.isEmpty()) {
        if (auto* ax = qobject_cast<QValueAxis*>(axesH.first()))
            ax->setTitleText(xLabel);
    }
    if (!axesV.isEmpty()) {
        if (auto* ay = qobject_cast<QValueAxis*>(axesV.first()))
            ay->setTitleText(yLabel);
    }

    auto* view = new QChartView(chart);
    view->setRenderHint(QPainter::Antialiasing);
    return view;
}

// 柱状图：cats 为 X 轴分类；bars[i] = {seriesName, [v1, v2, ... 对应 cats]}
inline QChartView* makeBarChart(
    const QString& title,
    const QStringList& cats,
    const std::vector<std::pair<QString, QVector<qreal>>>& bars,
    const QString& yLabel = {}
    ){
    auto* ser = new QBarSeries();
    for (const auto& kv : bars) {
        auto* set = new QBarSet(kv.first);
        for (qreal v : kv.second) *set << v;
        ser->append(set);
    }

    auto* chart = new QChart();
    chart->addSeries(ser);
    chart->setTitle(title);

    auto* axisX = new QBarCategoryAxis();
    axisX->append(cats);
    chart->addAxis(axisX, Qt::AlignBottom);
    ser->attachAxis(axisX);

    auto* axisY = new QValueAxis();
    axisY->setTitleText(yLabel);
    chart->addAxis(axisY, Qt::AlignLeft);
    ser->attachAxis(axisY);

    auto* view = new QChartView(chart);
    view->setRenderHint(QPainter::Antialiasing);
    return view;
}

// 饼图
inline QChartView* makePieChart(
    const QString& title,
    const std::vector<std::pair<QString, qreal>>& items
    ){
    auto* series = new QPieSeries();
    for (const auto& it : items) series->append(it.first, it.second);
    series->setLabelsVisible(true);

    auto* chart = new QChart();
    chart->addSeries(series);
    chart->setTitle(title);

    auto* view = new QChartView(chart);
    view->setRenderHint(QPainter::Antialiasing);
    return view;
}

}} // namespace seatui::charts
