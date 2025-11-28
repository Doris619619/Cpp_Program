#include "DatabaseInitializer.h"
#include "DatabaseSchemas.h"
#include "../utils/TimeUtils.h"
#include <iostream>
#include <vector>

DatabaseInitializer::DatabaseInitializer(SeatDatabase& db) : database_(db) {}

bool DatabaseInitializer::initializeSampleData() {
    try {
        // 清空现有数据
        if (!clearExistingData()) {
            return false;
        }

        // 插入样本座位数据
        if (!insertSampleSeats()) {
            return false;
        }

        // 插入样本事件数据
        if (!insertSampleEvents()) {
            return false;
        }

        std::cout << "Sample data initialized successfully." << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Sample data initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool DatabaseInitializer::clearExistingData() {
    try {
        database_.beginTransaction();

        // 按正确顺序清空表（考虑外键约束）
        database_.exec("DELETE FROM seat_agg_hourly");
        database_.exec("DELETE FROM seat_snapshots");
        database_.exec("DELETE FROM seat_events");
        database_.exec("DELETE FROM seats");

        database_.commitTransaction();
        return true;

    } catch (const std::exception& e) {
        database_.rollbackTransaction();
        std::cerr << "Failed to clear existing data: " << e.what() << std::endl;
        return false;
    }
}

bool DatabaseInitializer::insertSampleSeats() {
    try {
        // 安静学习区 (quiet)
        database_.insertSeat("A1", 100, 100, 50, 50);
        database_.insertSeat("A2", 200, 100, 50, 50);
        database_.insertSeat("A3",  300, 100, 50, 50);
        database_.insertSeat("A4",  400, 100, 50, 50);

        // 小组讨论区 (gruop)
        //database_.insertSeat("B1",  100, 200, 60, 60);
        //database_.insertSeat("B2",  200, 200, 60, 60);
        //database_.insertSeat("B3",  300, 200, 60, 60);

        // 电脑区 ("Computer")
        //database_.insertSeat("C1", 100, 300, 70, 60);
        //database_.insertSeat("C2", 200, 300, 70, 60);
        //database_.insertSeat("C3", 300, 300, 70, 60);
        //database_.insertSeat("C4", 400, 300, 70, 60);

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to insert sample seats: " << e.what() << std::endl;
        return false;
    }
}

bool DatabaseInitializer::insertSampleEvents() {
    try {
        std::string current_time = TimeUtils::getCurrentTimestamp();
        std::string one_hour_ago = TimeUtils::addHours(current_time, -1);
        std::string two_hours_ago = TimeUtils::addHours(current_time, -2);

        // 插入一些历史事件
        database_.insertSeatEvent("A1", "Seated", two_hours_ago, 3600);
        database_.insertSeatEvent("A2", "Unseated", two_hours_ago, 0);
        //database_.insertSeatEvent("B1", "Anomaly", one_hour_ago, 1800);

        // 插入当前状态事件
        database_.insertSeatEvent("A1", "Seated", current_time, 0);
        database_.insertSeatEvent("A2", "Seated", current_time, 0);
        database_.insertSeatEvent("A3", "Unseated", current_time, 0);
        //database_.insertSeatEvent("B1", "Anomaly", current_time, 0);
        //database_.insertSeatEvent("C1", "Seated", current_time, 0);

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to insert sample events: " << e.what() << std::endl;
        return false;
    }
}

bool DatabaseInitializer::exec(const std::string& sql) {
    try {
        database_.exec(sql);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "SQL execution failed: " << e.what() << std::endl;
        return false;
    }
}