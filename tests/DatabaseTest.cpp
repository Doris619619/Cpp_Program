#include "../src/database/SeatDatabase.h"
#include "../src/database/DataTypes.h"
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include <thread>

using namespace std;

class DatabaseTest {
private:
    SeatDatabase& db;
    
public:
    DatabaseTest() : db(SeatDatabase::getInstance("../test_database.db")) {}
    
    void runAllTests() {
        cout << "Starting Database Module Tests..." << endl;
        cout << "==========================================" << endl;
        
        // 1. Database Initialization Test
        testDatabaseInitialization();
        
        // 2. Basic Data Insertion Test
        testBasicDataInsertion();
        
        // 3. Seat Events Test
        testSeatEvents();
        
        // 4. Snapshot Data Test
        testSnapshotData();
        
        // 5. Query Functions Test
        testQueryFunctions();
        
        // 6. Statistics Test
        testStatistics();
        
        // 7. Transaction Test
        testTransaction();
        
        cout << "==========================================" << endl;
        cout << "All Tests Completed!" << endl;
    }
    
private:
    void testDatabaseInitialization() {
        cout << "\nTest 1: Database Initialization" << endl;
        bool success = db.initialize();
        assert(success && "Database initialization failed");
        cout << "Database initialized successfully" << endl;
    }
    
    void testBasicDataInsertion() {
        cout << "\nTest 2: Basic Data Insertion" << endl;
        
        // Insert test seat data - only A1, A2, A3, A4
        vector<tuple<string, int, int, int, int>> testSeats = {
            {"A1", 100, 200, 50, 60},
            {"A2", 200, 200, 50, 60},
            {"A3", 300, 200, 50, 60},
            {"A4", 400, 200, 50, 60}
        };
        
        for (const auto& seat : testSeats) {
            bool success = db.insertSeat(
                get<0>(seat), 
                get<1>(seat), get<2>(seat), 
                get<3>(seat), get<4>(seat)
            );
            assert(success && "Seat data insertion failed");
            cout << "Inserted seat: " << get<0>(seat) << endl;
        }
        
        // Verify seat count
        auto allSeats = db.getAllSeatIds();
        assert(allSeats.size() == 4 && "Seat count mismatch");
        cout << "Seat count verified: " << allSeats.size() << endl;
    }
    
    void testSeatEvents() {
        cout << "\nTest 3: Seat Events Recording" << endl;
        
        // Simulate state change events from Module B
        vector<tuple<string, string, string, int>> testEvents = {
            {"A1", "Seated", "2024-01-15T08:00:00.000", 300},
            {"A2", "Unseated", "2024-01-15T08:00:00.000", 0},
            {"A3", "Anomaly", "2024-01-15T08:05:00.000", 600},
            {"A1", "Unseated", "2024-01-15T08:10:00.000", 600},
            {"A4", "Seated", "2024-01-15T08:15:00.000", 450}
        };
        
        for (const auto& event : testEvents) {
            bool success = db.insertSeatEvent(
                get<0>(event), get<1>(event), get<2>(event), get<3>(event)
            );
            assert(success && "Seat event insertion failed");
            cout << "Event: " << get<0>(event) << " -> " << get<1>(event) 
                 << " Duration:" << get<3>(event) << "s" << endl;
        }
        
        // Verify event recording
        int occupiedMinutes = db.getOccupiedMinutes("A1", "2024-01-15T08:00:00.000", "2024-01-15T09:00:00.000");
        assert(occupiedMinutes > 0 && "Occupied time calculation error");
        cout << "Occupied time calculation: " << occupiedMinutes << " minutes" << endl;
    }
    
    void testSnapshotData() {
        cout << "\nTest 4: Snapshot Data" << endl;
        
        // Simulate snapshot data from Module B
        vector<tuple<string, string, string, int>> testSnapshots = {
            {"2024-01-15T08:30:00.000", "A1", "Unseated", 0},
            {"2024-01-15T08:30:00.000", "A2", "Unseated", 0},
            {"2024-01-15T08:30:00.000", "A3", "Anomaly", 0},
            {"2024-01-15T08:30:00.000", "A4", "Seated", 1}
        };
        
        for (const auto& snapshot : testSnapshots) {
            bool success = db.insertSnapshot(
                get<0>(snapshot), get<1>(snapshot), get<2>(snapshot), get<3>(snapshot)
            );
            assert(success && "Snapshot data insertion failed");
            cout << "Snapshot: " << get<1>(snapshot) << " -> " << get<2>(snapshot) 
                 << " Persons:" << get<3>(snapshot) << endl;
        }
    }
    
    void testQueryFunctions() {
        cout << "\nTest 5: Query Functions" << endl;
        
        // Test current seat status query
        auto currentStatus = db.getCurrentSeatStatus();
        assert(!currentStatus.empty() && "Current seat status query failed");
        
        cout << "Current Seat Status:" << endl;
        for (const auto& status : currentStatus) {
            cout << "  " << status.seat_id << ": " << status.state 
                 << " (Updated: " << status.last_update << ")" << endl;
        }
        cout << "Current seat status query successful" << endl;
        
        // Test occupancy rate query
        double occupancyRate = db.getOverallOccupancyRate("2024-01-15 08:00:00");
        cout << "Overall occupancy rate query: " << (occupancyRate * 100) << "%" << endl;
    }
    
    void testStatistics() {
        cout << "\nTest 6: Statistics Functions" << endl;
        
        // Test basic statistics
        BasicStats stats = db.getCurrentBasicStats();
        
        cout << "Basic Statistics:" << endl;
        cout << "  Total Seats: " << stats.total_seats << endl;
        cout << "  Occupied Seats: " << stats.occupied_seats << endl;
        cout << "  Anomaly Seats: " << stats.anomaly_seats << endl;
        cout << "  Overall Occupancy Rate: " << (stats.overall_occupancy_rate * 100) << "%" << endl;
        
        assert(stats.total_seats == 4 && "Total seat count statistics error");
        cout << "Basic statistics verification successful" << endl;
        
        // Test today's hourly data
        auto hourlyData = db.getTodayHourlyData();
        cout << "Today's hourly data points: " << hourlyData.size() << endl;
        if (!hourlyData.empty()) {
            cout << "  Example: " << hourlyData[0].hour << " -> " 
                 << (hourlyData[0].occupancy_rate * 100) << "%" << endl;
        }
        cout << "Hourly data query successful" << endl;
    }
    
    void testTransaction() {
        cout << "\nTest 7: Transaction Functions" << endl;
        
        // Test transaction operations
        bool beginSuccess = db.beginTransaction();
        assert(beginSuccess && "Begin transaction failed");
        cout << "Transaction started successfully" << endl;
        
        // Insert data within transaction
        bool insertSuccess = db.insertSeatEvent("A4", "Seated", "2024-01-15T09:00:00.000", 300);
        assert(insertSuccess && "Data insertion within transaction failed");
        cout << "Data inserted within transaction successfully" << endl;
        
        bool commitSuccess = db.commitTransaction();
        assert(commitSuccess && "Commit transaction failed");
        cout << "Transaction committed successfully" << endl;
        
        // Test rollback
        beginSuccess = db.beginTransaction();
        db.insertSeatEvent("A4", "Unseated", "2024-01-15T09:05:00.000", 0);
        bool rollbackSuccess = db.rollbackTransaction();
        assert(rollbackSuccess && "Rollback transaction failed");
        cout << "Transaction rollback successful" << endl;
        
        // Verify rollback effect - final state should be Seated, not Unseated
        auto status = db.getCurrentSeatStatus();
        string lastState;
        for (const auto& s : status) {
            if (s.seat_id == "A4") {
                lastState = s.state;
                break;
            }
        }
        assert(lastState == "Seated" && "Transaction rollback verification failed");
        cout << "Transaction rollback verification successful" << endl;
    }
};

// Integration Test: Simulate complete interaction between Module B and Database
void runIntegrationTest() {
    cout << "\nIntegration Test: Simulating Complete Module B Workflow" << endl;
    cout << "==========================================" << endl;
    
    SeatDatabase& db = SeatDatabase::getInstance("../integration_test.db");
    db.initialize();
    
    // 1. Initialize seat data (simulate system startup)
    cout << "\n1. Initializing seat data..." << endl;
    vector<string> seatIds = {"A1", "A2", "A3", "A4"};
    for (size_t i = 0; i < seatIds.size(); ++i) {
        db.insertSeat(seatIds[i], 100 + i*50, 200, 50, 60);
    }
    
    // 2. Simulate state detection sequence from Module B
    cout << "\n2. Simulating Module B State Detection Sequence..." << endl;
    
    // Time sequence: 08:00 - 09:00
    vector<tuple<string, string, string, int>> detectionSequence = {
        // Time, SeatID, State, Duration
        {"2024-01-15T08:00:00.000", "A1", "Unseated", 0},
        {"2024-01-15T08:00:00.000", "A2", "Unseated", 0},
        {"2024-01-15T08:00:00.000", "A3", "Unseated", 0},
        {"2024-01-15T08:00:00.000", "A4", "Unseated", 0},
        
        {"2024-01-15T08:10:00.000", "A1", "Seated", 600},     // A1 occupied
        {"2024-01-15T08:15:00.000", "A2", "Seated", 300},     // A2 occupied
        {"2024-01-15T08:20:00.000", "A3", "Anomaly", 1200},   // A3 anomaly occupied
        
        {"2024-01-15T08:25:00.000", "A1", "Unseated", 300},   // A1 left
        {"2024-01-15T08:30:00.000", "A4", "Seated", 1800},    // A4 occupied
        
        {"2024-01-15T08:45:00.000", "A2", "Unseated", 900},   // A2 left
    };
    
    // Process each detection result
    for (const auto& detection : detectionSequence) {
        string timestamp = get<0>(detection);
        string seatId = get<1>(detection);
        string state = get<2>(detection);
        int duration = get<3>(detection);
        
        // Insert event record
        db.insertSeatEvent(seatId, state, timestamp, duration);
        
        // Insert snapshot (simulate periodic saving)
        if (timestamp.find(":00:00") != string::npos || 
            timestamp.find(":30:00") != string::npos) {
            int personCount = (state == "Seated") ? 1 : 0;
            db.insertSnapshot(timestamp, seatId, state, personCount);
        }
        
        cout << "  Processing: " << seatId << " -> " << state 
             << " at " << timestamp << endl;
    }
    
    // 3. Verify final state
    cout << "\n3. Verifying Final State..." << endl;
    auto finalStatus = db.getCurrentSeatStatus();
    auto finalStats = db.getCurrentBasicStats();
    
    cout << "Final Statistics:" << endl;
    cout << "  Total Seats: " << finalStats.total_seats << endl;
    cout << "  Occupied: " << finalStats.occupied_seats << endl;
    cout << "  Anomaly: " << finalStats.anomaly_seats << endl;
    cout << "  Occupancy Rate: " << (finalStats.overall_occupancy_rate * 100) << "%" << endl;
    
    cout << "Seat Status:" << endl;
    for (const auto& status : finalStatus) {
        cout << "  " << status.seat_id << ": " << status.state << endl;
    }
    
    // 4. Test time period queries
    cout << "\n4. Testing Time Period Queries..." << endl;
    int occupiedTime = db.getOccupiedMinutes("A1", "2024-01-15T08:00:00.000", "2024-01-15T09:00:00.000");
    cout << "  A1 occupied during 08:00-09:00: " << occupiedTime << " minutes" << endl;
    
    cout << "\nIntegration Test Completed!" << endl;
}

int main() {
    cout << "Seat System Database Module Test Program" << endl;
    cout << "==========================================" << endl;
    
    try {
        // Run unit tests
        DatabaseTest tester;
        tester.runAllTests();
        
        // Run integration test
        runIntegrationTest();
        
        cout << "\nAll tests passed! Database module functions correctly." << endl;
        cout << "Tip: Check generated database files to verify data persistence" << endl;
        
    } catch (const exception& e) {
        cerr << "Test failed: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}