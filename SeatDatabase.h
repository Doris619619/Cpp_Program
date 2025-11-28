#ifndef SEAT_DATABASE_H
#define SEAT_DATABASE_H

//#include "SQLiteCpp/SQLiteCpp.h"
#include "../../include/SQLiteCpp/SQLiteCpp.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>

#include "DataTypes.h"  // 包含共享的数据类型

class SeatDatabase {
public:
    // 单例模式获取实例
    static SeatDatabase& getInstance(const std::string& db_path = "../database/seating_system.db");
    
    // 禁止拷贝和赋值
    SeatDatabase(const SeatDatabase&) = delete;
    SeatDatabase& operator=(const SeatDatabase&) = delete;
    
    // 数据库初始化
    bool initialize();
    
    // 插入操作
    bool insertSeatEvent(const std::string& seat_id, 
                        const std::string& state, // "Seated", "Unseated", "Anomaly"
                        const std::string& timestamp, 
                        int duration_sec = 0);
    
    bool insertSnapshot(const std::string& timestamp, 
                       const std::string& seat_id, 
                       const std::string& state, 
                       int person_count = 0);
    
    bool insertHourlyAggregation(const std::string& date_hour, 
                                const std::string& seat_id, 
                                int occupied_minutes);
    
    // 基础数据插入
    bool insertSeat(const std::string& seat_id, 
                   int roi_x, int roi_y, 
                   int roi_width, int roi_height);
    
    // 查询操作
    int getOccupiedMinutes(const std::string& seat_id, 
                          const std::string& start_time, 
                          const std::string& end_time);
    
    //double getZoneOccupancyRate(const std::string& zone, 
    //                           const std::string& date_hour);
    
    double getOverallOccupancyRate(const std::string& date_hour);
    
    // UI 数据接口
    std::vector<SeatStatus> getCurrentSeatStatus();
    BasicStats getCurrentBasicStats();
    std::vector<HourlyData> getTodayHourlyData();
    std::map<std::string, double> getHourlyZoneOccupancy(const std::string& date_hour);
    std::vector<double> getDailyHourlyOccupancy(const std::string& date);
    
    // 批量操作
    bool beginTransaction();
    bool commitTransaction();
    bool rollbackTransaction();
    
    // 工具方法
    std::vector<std::string> getAllSeatIds();
    std::string getCurrentTimestamp();

     // 添加exec方法
    bool exec(const std::string& sql);
    
    // 告警入库接口
    bool insertAlert(
        const std::string& alert_id,
        const std::string& seat_id,
        const std::string& alert_type,
        const std::string& alert_desc,
        const std::string& timestamp,
        bool is_processed = false
    );

     // 获取未处理告警
    std::vector<AlertData> getUnprocessedAlerts();

    // 标记告警为已处理
    bool markAlertAsProcessed(const std::string& alert_id);
private:
    SeatDatabase(const std::string& db_path);
    ~SeatDatabase() = default;
    
    std::string db_path_;
    std::unique_ptr<SQLite::Database> database_;
    std::mutex db_mutex_;
    
    bool createTables();
    bool createIndexes();
};

#endif // SEAT_DATABASE_H