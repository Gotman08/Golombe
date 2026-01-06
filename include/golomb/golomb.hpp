/**
 * @file golomb.hpp
 * @brief Core data structures for Golomb Ruler Solver
 *
 * Golomb Ruler Solver - High Performance Computing Implementation
 * Copyright (c) 2025 Nicolas Marano
 * Licensed under the MIT License. See LICENSE file for details.
 */

#ifndef GOLOMB_HPP
#define GOLOMB_HPP

#include <vector>
#include <bitset>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <chrono>

// Maximum ruler length for bitset sizing
// G11 = 72, G12 = 85, using 256 for safety margin
constexpr int MAX_LENGTH = 256;
constexpr int MAX_ORDER = 20;

// Known optimal lengths for Golomb rulers (verified - OEIS A003022)
// Index = order, value = optimal length
// Note: Multiple distinct optimal rulers exist for each order
// Examples shown are one possible solution among several
constexpr int OPTIMAL_LENGTHS[] = {
    0,   // G0 (undefined)
    0,   // G1
    1,   // G2: [0, 1]
    3,   // G3: [0, 1, 3]
    6,   // G4: [0, 1, 4, 6] or [0, 1, 3, 6] (both optimal)
    11,  // G5: [0, 1, 4, 9, 11] or [0, 2, 7, 8, 11] (multiple solutions)
    17,  // G6: [0, 1, 4, 10, 12, 17] (one of several)
    25,  // G7: [0, 1, 4, 10, 18, 23, 25] (one of several)
    34,  // G8: [0, 1, 4, 9, 15, 22, 32, 34] (one of several)
    44,  // G9: [0, 1, 5, 12, 25, 27, 35, 41, 44] (one of several)
    55,  // G10: [0, 1, 6, 10, 23, 26, 34, 41, 53, 55] (one of several)
    72,  // G11: [0, 1, 4, 13, 28, 33, 47, 54, 64, 70, 72] (one of several)
    85,  // G12
    106, // G13
    127  // G14
};

// Golomb Ruler structure
// Members ordered for optimal cache alignment (larger first, then smaller)
struct GolombRuler {
    std::vector<int> marks;  // Positions [0, a2, a3, ..., an] (24 bytes on 64-bit)
    int length;              // Last mark value (marks.back()) (4 bytes)
    int order;               // Number of marks (n) (4 bytes)

    GolombRuler() : length(INT_MAX), order(0) {}

    GolombRuler(const std::vector<int>& m)
        : marks(m),
          length(m.empty() ? 0 : m.back()),
          order(static_cast<int>(m.size())) {}

    // Convert to string representation
    std::string toString() const;

    // Print the ruler
    void print() const;

    // Check if this ruler is valid
    bool isValid() const;
};

// Search statistics
struct SearchStats {
    uint64_t nodesExplored;   // Total nodes visited in search tree
    uint64_t nodesPruned;     // Nodes cut by bounding
    double elapsedMs;         // Total execution time in milliseconds
    GolombRuler bestSolution; // Best solution found

    SearchStats() : nodesExplored(0), nodesPruned(0), elapsedMs(0.0) {}
};

// Validation functions (implemented in validation.cpp)
bool isValidGolombRuler(const std::vector<int>& marks);
bool isValidGolombRulerFast(const std::vector<int>& marks, std::bitset<MAX_LENGTH>& usedDiffs);
bool isOptimalLength(int order, int length);

// Timing class (implemented in timing.cpp)
class Timer {
private:
    std::chrono::high_resolution_clock::time_point startTime;
    bool running;

public:
    Timer();
    void start();
    void reset();
    double elapsedMs() const;
};

// CSV output helper
void writeResultCSV(const std::string& filename,
                    int version,
                    const std::vector<SearchStats>& results,
                    const std::vector<int>& orders);

// Print search statistics (defined in timing.cpp)
void printStats(const SearchStats& stats, int order);

// Upper bound estimate: n^2 is a safe upper bound for order n
// Note: For order > 46340, this would overflow int32. We cap at INT_MAX.
[[gnu::always_inline]]
inline int upperBoundEstimate(int order) {
    if (order > 46340) return INT_MAX;  // Prevent overflow: sqrt(INT_MAX) ~ 46340
    return order * order;
}

// Theoretical lower bound (Erdos-Turan): n(n-1)/2
[[gnu::always_inline]]
inline int lowerBoundEstimate(int order) {
    return order * (order - 1) / 2;
}

// Maximum order with known optimal length
constexpr int MAX_KNOWN_ORDER = 14;

// Safe access to OPTIMAL_LENGTHS with bounds checking
inline int getOptimalLength(int order) {
    if (order < 2 || order > MAX_KNOWN_ORDER) {
        return -1;  // Unknown
    }
    return OPTIMAL_LENGTHS[order];
}

// Input validation helper - returns parsed order or -1 on error
inline int parseAndValidateOrder(const char* str, int maxOrder = MAX_ORDER) {
    if (str == nullptr) return -1;

    char* endptr;
    long val = std::strtol(str, &endptr, 10);

    // Check for conversion errors
    if (*endptr != '\0' || endptr == str) {
        return -1;  // Not a valid number
    }

    // Check range
    if (val < 2 || val >= maxOrder) {
        return -1;  // Out of range
    }

    return static_cast<int>(val);
}

#endif // GOLOMB_HPP
