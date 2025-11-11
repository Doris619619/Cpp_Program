#pragma once
// 网络/WS 相关的载荷类型（学生端举报等）

#include <QtCore/QString>
#include <QtCore/QDateTime>
#include <QtCore/QJsonObject>
#include <QtCore/QJsonDocument>
#include <QtCore/QMetaType>

namespace seatui {

struct ReportPayload {
    int        seat_or_zone_id = -1;    // 座位或分区（至少填一个）
    QString    category;                // 噪音/占座/卫生/违纪...
    QString    description;             // 详细描述
    QString    image_path;              // 可选：本地截图路径/或已上传URL
    QDateTime  ts = QDateTime::currentDateTimeUtc();

    bool isValid(QString* err=nullptr) const {
        if (seat_or_zone_id < 0) { if (err) *err = "seat_or_zone_id 无效"; return false; }
        if (category.trimmed().isEmpty()) { if (err) *err = "category 不能为空"; return false; }
        if (description.trimmed().isEmpty()) { if (err) *err = "description 不能为空"; return false; }
        return true;
    }

    QJsonObject toJson() const {
        return {
            {"seat_or_zone_id", seat_or_zone_id},
            {"category", category},
            {"description", description},
            {"image_path", image_path},
            {"ts", ts.toUTC().toString(Qt::ISODateWithMs)}
        };
    }

    static ReportPayload fromJson(const QJsonObject& o) {
        ReportPayload p;
        p.seat_or_zone_id = o.value("seat_or_zone_id").toInt(-1);
        p.category        = o.value("category").toString();
        p.description     = o.value("description").toString();
        p.image_path      = o.value("image_path").toString();
        p.ts = QDateTime::fromString(o.value("ts").toString(), Qt::ISODateWithMs);
        if (!p.ts.isValid()) p.ts = QDateTime::currentDateTimeUtc();
        return p;
    }

    QByteArray toJsonBytes() const {
        return QJsonDocument(toJson()).toJson(QJsonDocument::Compact);
    }
};

} // namespace seatui

Q_DECLARE_METATYPE(seatui::ReportPayload)
