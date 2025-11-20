#pragma once
#include <string>

namespace vision {

// Enumeration: seat occupancy states.    
enum class SeatOccupancyState {
    FREE = 0,
    PERSON,
    OBJECT_ONLY,
    PERSON_AND_OBJECT,
    UNKNOWN
};

inline std::string toString(SeatOccupancyState s) {
    switch (s) {
        case SeatOccupancyState::FREE:              return "FREE";
        case SeatOccupancyState::PERSON:            return "PERSON";
        case SeatOccupancyState::OBJECT_ONLY:       return "OBJECT_ONLY";
        case SeatOccupancyState::PERSON_AND_OBJECT: return "PERSON_AND_OBJECT";
        default:                                    return "UNKNOWN";
    }
}

} // namespace vision