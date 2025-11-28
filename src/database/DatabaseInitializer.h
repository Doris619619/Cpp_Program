#ifndef DATABASE_INITIALIZER_H
#define DATABASE_INITIALIZER_H

#include "SeatDatabase.h"
#include "../utils/TimeUtils.h"
#include <string>

class DatabaseInitializer {
public:
    DatabaseInitializer(SeatDatabase& db);
    
    // 初始化样本数据
    bool initializeSampleData();
    
    // 清空现有数据
    bool clearExistingData();
    
    // 插入样本座位
    bool insertSampleSeats();
    
    // 插入样本事件
    bool insertSampleEvents();
    
private:
    SeatDatabase& database_;
    
    // 执行SQL语句
    bool exec(const std::string& sql);
};

#endif // DATABASE_INITIALIZER_H