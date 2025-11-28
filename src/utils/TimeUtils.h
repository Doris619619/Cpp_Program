#ifndef TIME_UTILS_H
#define TIME_UTILS_H

#include <string>
#include <chrono>

class TimeUtils {
public:
    static std::string getCurrentTimestamp();
    static std::string formatTimestamp(const std::string& timestamp, const std::string& format);
    static std::string addHours(const std::string& timestamp, int hours);
    static std::string getHourStart(const std::string& timestamp);
    static int getMinutesBetween(const std::string& start, const std::string& end);
    static bool isSameDay(const std::string& timestamp1, const std::string& timestamp2);
    
private:
    static time_t stringToTime(const std::string& timestamp);
    static std::string timeToString(time_t time);
};

#endif // TIME_UTILS_H