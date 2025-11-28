#include "../src/database/SeatDatabase.h"
#include "../src/database/DataTypes.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <iomanip>
#include <sstream>

class SeatDatabaseTester {
private:
    SeatDatabase& db;
    
    // Helper method to get current timestamp
    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        time_t time_t_val = std::chrono::system_clock::to_time_t(now);
        std::tm tm_val = *std::localtime(&time_t_val);
        
        char buffer[20];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_val);
        return std::string(buffer);
    }
    
    // Helper method to format time
    std::string formatTime(const std::chrono::system_clock::time_point& time_point) {
        time_t time_t_val = std::chrono::system_clock::to_time_t(time_point);
        std::tm tm_val = *std::localtime(&time_t_val);
        
        char buffer[20];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_val);
        return std::string(buffer);
    }
    
public:
    SeatDatabaseTester(const std::string& db_path) : db(SeatDatabase::getInstance(db_path)) {}
    
    // Test database initialization
    bool testInitialization() {
        std::cout << "=== Testing Database Initialization ===" << std::endl;
        bool success = db.initialize();
        std::cout << "Database initialization: " << (success ? "SUCCESS" : "FAILED") << std::endl;
        return success;
    }
    
    // Test seat information insertion
    bool testInsertSeats() {
        std::cout << "\n=== Testing Seat Information Insertion ===" << std::endl;
        bool allSuccess = true;
        
        // Insert multiple seats
        std::vector<std::tuple<std::string, int, int, int, int>> seats = {
            {"A1", 100, 200, 50, 60},
            {"A2", 200, 200, 50, 60},
            {"A3", 300, 200, 50, 60},
            {"A4", 400, 200, 50, 60},
        };
        
        for (const auto& seat : seats) {
            bool success = db.insertSeat(
                std::get<0>(seat), 
                std::get<1>(seat), 
                std::get<2>(seat), 
                std::get<3>(seat), 
                std::get<4>(seat)
            );
            std::cout << "Insert seat " << std::get<0>(seat) << ": " 
                      << (success ? "SUCCESS" : "FAILED") << std::endl;
            allSuccess &= success;
        }
        
        return allSuccess;
    }
    
    // Test seat event insertion
    bool testInsertSeatEvents() {
        std::cout << "\n=== Testing Seat Event Insertion ===" << std::endl;
        bool allSuccess = true;
        
        std::vector<std::string> seat_ids = {"A1", "A2", "A3", "A4"};
        std::vector<std::string> states = {"Seated", "Unseated", "Anomaly"};
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> state_dist(0, states.size() - 1);
        std::uniform_int_distribution<> duration_dist(60, 3600);
        
        // Generate event data for the past 24 hours
        auto now = std::chrono::system_clock::now();
        
        for (int i = 0; i < 50; ++i) {
            auto timestamp = now - std::chrono::hours(24) + 
                           std::chrono::minutes(i * 30);
            
            std::string timestamp_str = formatTime(timestamp);
            
            std::string seat_id = seat_ids[i % seat_ids.size()];
            std::string state = states[state_dist(gen)];
            int duration = duration_dist(gen);
            
            bool success = db.insertSeatEvent(seat_id, state, timestamp_str, duration);
            
            if (i < 10) { // Only print first 10 records
                std::cout << "Insert event: " << seat_id << " - " << state 
                          << " - " << timestamp_str << " - " << duration << "s: "
                          << (success ? "SUCCESS" : "FAILED") << std::endl;
            }
            
            allSuccess &= success;
        }
        
        std::cout << "Total 50 seat events inserted" << std::endl;
        return allSuccess;
    }
    
    // Test snapshot insertion
    bool testInsertSnapshots() {
        std::cout << "\n=== Testing Snapshot Insertion ===" << std::endl;
        bool allSuccess = true;
        
        std::vector<std::string> seat_ids = {"A1", "A2", "A3", "A4"};
        std::vector<std::string> states = {"Seated", "Unseated", "Anomaly"};
        
        std::string timestamp_str = getCurrentTimestamp();
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> state_dist(0, states.size() - 1);
        std::uniform_int_distribution<> count_dist(0, 2);
        
        for (const auto& seat_id : seat_ids) {
            std::string state = states[state_dist(gen)];
            int person_count = count_dist(gen);
            
            bool success = db.insertSnapshot(timestamp_str, seat_id, state, person_count);
            std::cout << "Insert snapshot: " << seat_id << " - " << state 
                      << " - " << person_count << " person(s): "
                      << (success ? "SUCCESS" : "FAILED") << std::endl;
            allSuccess &= success;
        }
        
        return allSuccess;
    }
    
    // Test hourly aggregation data insertion
    bool testInsertHourlyAggregation() {
        std::cout << "\n=== Testing Hourly Aggregation Data Insertion ===" << std::endl;
        bool allSuccess = true;
        
        std::vector<std::string> seat_ids = {"A1", "A2", "A3", "A4"};
        
        auto now = std::chrono::system_clock::now();
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> minutes_dist(0, 60);
        
        for (int hour = 0; hour < 24; ++hour) {
            auto timestamp = now - std::chrono::hours(24) + std::chrono::hours(hour);
            std::string date_hour_str = formatTime(timestamp);
            // Replace minutes and seconds with 00:00 for hourly data
            date_hour_str.replace(14, 5, "00:00");
            
            for (const auto& seat_id : seat_ids) {
                int occupied_minutes = minutes_dist(gen);
                bool success = db.insertHourlyAggregation(date_hour_str, seat_id, occupied_minutes);
                allSuccess &= success;
            }
        }
        
        std::cout << "Inserted 24 hours of aggregation data: " << (allSuccess ? "SUCCESS" : "FAILED") << std::endl;
        return allSuccess;
    }
    
    // Test getting current seat status
    void testGetCurrentSeatStatus() {
        std::cout << "\n=== Testing Current Seat Status Retrieval ===" << std::endl;
        
        auto statusList = db.getCurrentSeatStatus();
        std::cout << "Current seat status (" << statusList.size() << " seats):" << std::endl;
        
        for (const auto& status : statusList) {
            std::cout << "Seat " << status.seat_id << ": " << status.state 
                      << " (Last update: " << status.last_update << ")" << std::endl;
        }
    }
    
    // Test getting basic statistics
    void testGetBasicStats() {
        std::cout << "\n=== Testing Basic Statistics Retrieval ===" << std::endl;
        
        BasicStats stats = db.getCurrentBasicStats();
        
        std::cout << "Total seats: " << stats.total_seats << std::endl;
        std::cout << "Occupied seats: " << stats.occupied_seats << std::endl;
        std::cout << "Anomaly seats: " << stats.anomaly_seats << std::endl;
        std::cout << "Overall occupancy rate: " << (stats.overall_occupancy_rate * 100) << "%" << std::endl;
    }
    
    // Test getting occupied minutes
    void testGetOccupiedMinutes() {
        std::cout << "\n=== Testing Occupied Minutes Retrieval ===" << std::endl;
        
        auto now = std::chrono::system_clock::now();
        auto start_time = now - std::chrono::hours(2);
        auto end_time = now;
        
        std::string start_str = formatTime(start_time);
        std::string end_str = formatTime(end_time);
        
        std::vector<std::string> seat_ids = {"A1", "A2", "A3"};
        
        for (const auto& seat_id : seat_ids) {
            int minutes = db.getOccupiedMinutes(seat_id, start_str, end_str);
            std::cout << "Seat " << seat_id << " occupied from " << start_str << " to " << end_str 
                      << ": " << minutes << " minutes" << std::endl;
        }
    }
    
    // Test getting overall occupancy rate
    void testGetOverallOccupancyRate() {
        std::cout << "\n=== Testing Overall Occupancy Rate Retrieval ===" << std::endl;
        
        auto now = std::chrono::system_clock::now();
        std::string date_hour_str = formatTime(now);
        date_hour_str.replace(14, 5, "00:00");
        
        double rate = db.getOverallOccupancyRate(date_hour_str);
        std::cout << "Current hour (" << date_hour_str << ") overall occupancy rate: " 
                  << (rate * 100) << "%" << std::endl;
    }
    
    // Test getting daily hourly occupancy
    void testGetDailyHourlyOccupancy() {
        std::cout << "\n=== Testing Daily Hourly Occupancy Retrieval ===" << std::endl;
        
        auto now = std::chrono::system_clock::now();
        std::string date_str = formatTime(now);
        date_str = date_str.substr(0, 10); // Get only YYYY-MM-DD
        
        auto hourly_rates = db.getDailyHourlyOccupancy(date_str);
        size_t hours_to_display = std::min(hourly_rates.size(), static_cast<size_t>(24));

        std::cout << "Hourly occupancy rates for " << date_str << ":" << std::endl;
        for (size_t hour = 0; hour < hours_to_display; ++hour) {
        std::cout << "  " << (hour < 10 ? "0" : "") << hour << ":00 - " 
                  << (hourly_rates[hour] * 100) << "%" << std::endl;
    }
    }
    
    // Test transaction functionality
    bool testTransaction() {
        std::cout << "\n=== Testing Transaction Functionality ===" << std::endl;
        
        bool success = true;
        
        // Begin transaction
        if (!db.beginTransaction()) {
            std::cerr << "Failed to begin transaction" << std::endl;
            return false;
        }
        
        std::cout << "Transaction started successfully" << std::endl;
        
        // Insert some data within transaction
        std::string timestamp = getCurrentTimestamp();
        success &= db.insertSeatEvent("TEST_SEAT", "Seated", timestamp, 300);
        success &= db.insertSnapshot(timestamp, "TEST_SEAT", "Seated", 1);
        
        if (success) {
            // Commit transaction
            if (db.commitTransaction()) {
                std::cout << "Transaction committed successfully" << std::endl;
            } else {
                std::cerr << "Failed to commit transaction" << std::endl;
                success = false;
            }
        } else {
            // Rollback transaction
            if (db.rollbackTransaction()) {
                std::cout << "Transaction rolled back successfully" << std::endl;
            } else {
                std::cerr << "Failed to rollback transaction" << std::endl;
            }
        }
        
        return success;
    }
    
    // Test getting all seat IDs
    void testGetAllSeatIds() {
        std::cout << "\n=== Testing All Seat IDs Retrieval ===" << std::endl;
        
        auto seat_ids = db.getAllSeatIds();
        
        std::cout << "All seat IDs (" << seat_ids.size() << " seats):" << std::endl;
        for (const auto& seat_id : seat_ids) {
            std::cout << "  " << seat_id << std::endl;
        }
    }
    
    // Test getting today's hourly data
    void testGetTodayHourlyData() {
        std::cout << "\n=== Testing Today's Hourly Data Retrieval ===" << std::endl;
        
        auto hourly_data = db.getTodayHourlyData();
        
        std::cout << "Today's hourly data:" << std::endl;
        for (const auto& data : hourly_data) {
            std::cout << "  " << data.hour << ": " << (data.occupancy_rate * 100) << "%" << std::endl;
        }
    }
    
    // Run all tests
    void runAllTests() {
        std::cout << "Starting comprehensive SeatDatabase testing..." << std::endl;
        std::cout << "==========================================" << std::endl;
        
        bool allTestsPassed = true;
        
        // Execute individual tests
        allTestsPassed &= testInitialization();
        allTestsPassed &= testInsertSeats();
        allTestsPassed &= testInsertSeatEvents();
        allTestsPassed &= testInsertSnapshots();
        allTestsPassed &= testInsertHourlyAggregation();
        
        // Query functionality tests
        testGetCurrentSeatStatus();
        testGetBasicStats();
        testGetOccupiedMinutes();
        testGetOverallOccupancyRate();
        testGetDailyHourlyOccupancy();
        testGetAllSeatIds();
        testGetTodayHourlyData();
        
        // Transaction test
        allTestsPassed &= testTransaction();
        
        std::cout << "\n==========================================" << std::endl;
        std::cout << "Testing completed: " << (allTestsPassed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    }
};

int main() {
    std::cout << "SeatDatabase Test Program" << std::endl;
    std::cout << "=========================" << std::endl;
    
    // Use test database file
    std::string test_db_path = "test_seat_database.db";
    
    try {
        SeatDatabaseTester tester(test_db_path);
        tester.runAllTests();
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred during testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}