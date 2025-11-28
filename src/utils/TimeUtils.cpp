#include "TimeUtils.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>

std::string TimeUtils::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::string TimeUtils::formatTimestamp(const std::string& timestamp, const std::string& format) {
    std::tm tm = {};
    std::istringstream ss(timestamp);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    
    if (ss.fail()) {
        throw std::runtime_error("Failed to parse timestamp: " + timestamp);
    }
    
    std::stringstream result;
    result << std::put_time(&tm, format.c_str());
    return result.str();
}

std::string TimeUtils::addHours(const std::string& timestamp, int hours) {
    std::tm tm = {};
    std::istringstream ss(timestamp);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    
    if (ss.fail()) {
        throw std::runtime_error("Failed to parse timestamp: " + timestamp);
    }
    
    // 转换为time_t进行操作
    std::time_t time = std::mktime(&tm);
    time += hours * 3600; // 添加小时
    
    // 转换回tm结构
    std::tm* new_tm = std::localtime(&time);
    
    std::stringstream result;
    result << std::put_time(new_tm, "%Y-%m-%d %H:%M:%S");
    return result.str();
}

std::string TimeUtils::getHourStart(const std::string& timestamp) {
    std::tm tm = {};
    std::istringstream ss(timestamp);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    
    if (ss.fail()) {
        throw std::runtime_error("Failed to parse timestamp: " + timestamp);
    }
    
    tm.tm_min = 0;
    tm.tm_sec = 0;
    
    std::stringstream result;
    result << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return result.str();
}

int TimeUtils::getMinutesBetween(const std::string& start, const std::string& end) {
    std::tm tm_start = {}, tm_end = {};
    
    std::istringstream ss_start(start);
    ss_start >> std::get_time(&tm_start, "%Y-%m-%d %H:%M:%S");
    
    std::istringstream ss_end(end);
    ss_end >> std::get_time(&tm_end, "%Y-%m-%d %H:%M:%S");
    
    if (ss_start.fail() || ss_end.fail()) {
        throw std::runtime_error("Failed to parse timestamps");
    }
    
    std::time_t time_start = std::mktime(&tm_start);
    std::time_t time_end = std::mktime(&tm_end);
    
    double difference = std::difftime(time_end, time_start);
    return static_cast<int>(difference / 60); // 转换为分钟
}

bool TimeUtils::isSameDay(const std::string& timestamp1, const std::string& timestamp2) {
    std::tm tm1 = {}, tm2 = {};
    
    std::istringstream ss1(timestamp1);
    ss1 >> std::get_time(&tm1, "%Y-%m-%d %H:%M:%S");
    
    std::istringstream ss2(timestamp2);
    ss2 >> std::get_time(&tm2, "%Y-%m-%d %H:%M:%S");
    
    if (ss1.fail() || ss2.fail()) {
        throw std::runtime_error("Failed to parse timestamps");
    }
    
    return (tm1.tm_year == tm2.tm_year && 
            tm1.tm_mon == tm2.tm_mon && 
            tm1.tm_mday == tm2.tm_mday);
}

time_t TimeUtils::stringToTime(const std::string& timestamp) {
    std::tm tm = {};
    std::istringstream ss(timestamp);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    
    if (ss.fail()) {
        throw std::runtime_error("Failed to parse timestamp: " + timestamp);
    }
    
    return std::mktime(&tm);
}

std::string TimeUtils::timeToString(time_t time) {
    std::tm* tm = std::localtime(&time);
    
    std::stringstream ss;
    ss << std::put_time(tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}