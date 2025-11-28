#ifndef DATABASE_SCHEMAS_H
#define DATABASE_SCHEMAS_H

#include <string>

namespace DatabaseSchemas {
    //座位
    const std::string CREATE_SEATS_TABLE = R"(
        CREATE TABLE IF NOT EXISTS seats (
            seat_id TEXT PRIMARY KEY,
            roi_x INTEGER NOT NULL,
            roi_y INTEGER NOT NULL,
            roi_width INTEGER NOT NULL,
            roi_height INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    )";
    //座位事件
    const std::string CREATE_SEAT_EVENTS_TABLE = R"(
        CREATE TABLE IF NOT EXISTS seat_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            seat_id TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('Seated', 'Unseated', 'Anomaly')),
            timestamp DATETIME NOT NULL,
            duration_sec INTEGER DEFAULT 0,
            FOREIGN KEY (seat_id) REFERENCES seats(seat_id)
        );
    )";
    //座位快照
    const std::string CREATE_SEAT_SNAPSHOTS_TABLE = R"(
        CREATE TABLE IF NOT EXISTS seat_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            seat_id TEXT NOT NULL,
            state TEXT NOT NULL,
            person_count INTEGER DEFAULT 0,
            FOREIGN KEY (seat_id) REFERENCES seats(seat_id)
        );
    )";
    //座位每小时聚合数据
    const std::string CREATE_SEAT_AGG_HOURLY_TABLE = R"(
        CREATE TABLE IF NOT EXISTS seat_agg_hourly (
            agg_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_hour DATETIME NOT NULL,
            seat_id TEXT NOT NULL,
            occupied_minutes INTEGER NOT NULL,
            FOREIGN KEY (seat_id) REFERENCES seats(seat_id),
            UNIQUE(date_hour, seat_id)
        );
    )";
    //告警表
    const std::string CREATE_ALERTS_TABLE = R"(
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id TEXT PRIMARY KEY,
            seat_id TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            alert_desc TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            is_processed INTEGER DEFAULT 0,
            created_time TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (seat_id) REFERENCES seats(seat_id)
        )
    )";
} // namespace DatabaseSchemas

#endif // DATABASE_SCHEMAS_H