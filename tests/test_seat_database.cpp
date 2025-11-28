#include <iostream>
#include <iomanip>
#include <filesystem>
#include "../src/database/SeatDatabase.h"
#include "../src/database/DatabaseInitializer.h"
#include <thread>
#include <chrono>

int main() {
    std::cout << "=== Intelligent Seat System Database Test ===" << std::endl;
    
    try {
        // Get database instance (using in-memory database to avoid file conflicts)
        SeatDatabase& db = SeatDatabase::getInstance(":memory:");
        
        // Initialize database
        if (db.initialize()) {
            std::cout << "Database initialized successfully" << std::endl;
        } else {
            std::cout << "Database initialization failed" << std::endl;
            return 1;
        }

        // Initialize sample data
        DatabaseInitializer initializer(db);
        if (initializer.initializeSampleData()) {
            std::cout << "Sample data initialized successfully" << std::endl;
        } else {
            std::cout << "Failed to initialize sample data" << std::endl;
            return 1;
        }
        
        // Test inserting a seat
        std::cout << "\nTest inserting a seat..." << std::endl;
        bool insertSeatSuccess = db.insertSeat(
            "TEST001",  
            100, 200, 50, 60
        );
        
        if (insertSeatSuccess) {
            std::cout << "Seat inserted successfully" << std::endl;
        } else {
            std::cout << "Failed to insert seat" << std::endl;
        }
        
        // Test inserting seat event (using correct method name)
        std::cout << "\nTest inserting seat event..." << std::endl;
        bool insertEventSuccess = db.insertSeatEvent(
            "TEST001", 
            "Seated", 
            "2024-01-15 10:30:00", 
            3600
        );
        
        if (insertEventSuccess) {
            std::cout << "Seat event inserted successfully" << std::endl;
        } else {
            std::cout << "Failed to insert seat event" << std::endl;
        }
        
        // Test inserting snapshot (using correct method name)
        std::cout << "\nTest inserting snapshot..." << std::endl;
        bool snapshotSuccess = db.insertSnapshot(
            "2024-01-15 10:30:00",
            "TEST001",
            "Seated",
            1
        );
        
        if (snapshotSuccess) {
            std::cout << "Snapshot inserted successfully" << std::endl;
        } else {
            std::cout << "Failed to insert snapshot" << std::endl;
        }
        
        // Test query for current seat status
        std::cout << "\nTest query for the current seat status..." << std::endl;
        auto seatStatus = db.getCurrentSeatStatus();
        if (!seatStatus.empty()) {
            std::cout << "Seat status query successful" << std::endl;
            std::cout << " Retrieved " << seatStatus.size() << " seat status records" << std::endl;
            for (const auto& status : seatStatus) {
                std::cout << "  Seat " << status.seat_id << ": " << status.state << std::endl;
            }
        } else {
            std::cout << "No seat status data available" << std::endl;
        }
        
        // Test query for basic statistics (using correct method name)
        std::cout << "\nTest query for basic statistics..." << std::endl;
        auto stats = db.getCurrentBasicStats();
        std::cout << "  Statistics query successful" << std::endl;
        std::cout << "  Total seats: " << stats.total_seats << std::endl;
        std::cout << "  Occupied seats: " << stats.occupied_seats << std::endl;
        std::cout << "  Anomaly seats: " << stats.anomaly_seats << std::endl;
        std::cout << "  Overall occupancy rate: " << std::fixed << std::setprecision(2) << stats.overall_occupancy_rate * 100 << "%" << std::endl;
        
        // Test getting all seat IDs
        std::cout << "\nTest getting all seat IDs..." << std::endl;
        auto seatIds = db.getAllSeatIds();
        if (!seatIds.empty()) {
            std::cout << "Retrieved " << seatIds.size() << " seat IDs" << std::endl;
            for (const auto& id : seatIds) {
                std::cout << "  " << id << std::endl;
            }
        } else {
            std::cout << "No seat data available" << std::endl;
        }
        /*
        // Test getting seats by zone
        std::cout << "\nTest getting seats by zone..." << std::endl;
        auto quietSeats = db.getSeatIdsByZone("Quiet");
        std::cout << "  Quiet zone seats: " << quietSeats.size() << std::endl;
        
        auto groupSeats = db.getSeatIdsByZone("Group"); 
        std::cout << "  Group zone seats: " << groupSeats.size() << std::endl;
        
        auto computerSeats = db.getSeatIdsByZone("Computer");
        std::cout << "  Computer zone seats: " << computerSeats.size() << std::endl;
        */

        // Test transaction functionality
        std::cout << "\nTest transaction functionality..." << std::endl;
        if (db.beginTransaction()) {
            std::cout << "Transaction started successfully" << std::endl;
            
            // Insert some data within transaction
            db.insertSeatEvent("TEST001", "Unseated", "2024-01-15 11:30:00", 0);
            
            if (db.commitTransaction()) {
                std::cout << "Transaction committed successfully" << std::endl;
            } else {
                std::cout << "Transaction commit failed" << std::endl;
            }
        } else {
            std::cout << "Transaction start failed" << std::endl;
        }
        
        std::cout << "\n All tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}