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

/**
 * @brief Maximum ruler length for bitset sizing.
 *
 * Current limit: 256 bits (sufficient for G14=127, G15~160).
 * To support G16+ (estimated length ~190+), would require:
 * 1. Increase this to 512
 * 2. Create BitSet512 in bitset256.hpp (double the words array)
 * 3. Update AVX2 code to use 2x _mm256 operations
 *
 * @note G14=127, G15≈160, G16≈190 (estimated)
 * @see BitSet256 which must match this limit
 */
constexpr int MAX_LENGTH = 256;

/** @brief Maximum supported ruler order */
constexpr int MAX_ORDER = 20;

/**
 * @brief Known optimal lengths for Golomb rulers (OEIS A003022).
 *
 * Index = order, value = optimal length.
 * Multiple distinct optimal rulers exist for each order.
 * @note Verified up to order 14.
 */
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

/**
 * @struct GolombRuler
 * @brief Represents a Golomb ruler with its marks, length, and order.
 *
 * A Golomb ruler is a set of marks at integer positions where all pairwise
 * distances are unique. The ruler always starts at position 0.
 *
 * Members ordered for optimal cache alignment (larger first, then smaller).
 */
struct GolombRuler {
    std::vector<int> marks;  ///< Mark positions [0, a2, a3, ..., an]
    int length;              ///< Ruler length (last mark position)
    int order;               ///< Number of marks

    /** @brief Default constructor, creates empty ruler with INT_MAX length. */
    GolombRuler() : length(INT_MAX), order(0) {}

    /**
     * @brief Constructs a ruler from mark positions.
     * @param m Vector of mark positions (must start with 0)
     */
    GolombRuler(const std::vector<int>& m)
        : marks(m),
          length(m.empty() ? 0 : m.back()),
          order(static_cast<int>(m.size())) {}

    /**
     * @brief Converts ruler to string "[0, a2, ..., an]".
     * @return String representation of the ruler
     */
    std::string toString() const;

    /** @brief Prints ruler info to stdout. */
    void print() const;

    /**
     * @brief Validates this ruler.
     * @return true if all pairwise distances are unique
     */
    bool isValid() const;
};

/**
 * @struct SearchStats
 * @brief Statistics collected during branch-and-bound search.
 */
struct SearchStats {
    uint64_t nodesExplored;   ///< Total nodes visited in search tree
    uint64_t nodesPruned;     ///< Nodes cut by bounding
    double elapsedMs;         ///< Total execution time in milliseconds
    GolombRuler bestSolution; ///< Best solution found

    /** @brief Default constructor, initializes counters to zero. */
    SearchStats() : nodesExplored(0), nodesPruned(0), elapsedMs(0.0) {}
};

/**
 * @brief Validates a Golomb ruler using std::set.
 * @param marks Vector of mark positions
 * @return true if valid Golomb ruler, false otherwise
 * @complexity O(n²) where n = number of marks
 */
bool isValidGolombRuler(const std::vector<int>& marks);

/**
 * @brief Fast Golomb ruler validation using bitset.
 * @param marks Vector of mark positions
 * @param usedDiffs Bitset to track differences (will be modified)
 * @return true if valid Golomb ruler, false otherwise
 * @complexity O(n²) with O(1) lookup
 */
bool isValidGolombRulerFast(const std::vector<int>& marks, std::bitset<MAX_LENGTH>& usedDiffs);

/**
 * @brief Checks if length matches known optimal for given order.
 * @param order Ruler order (2-14)
 * @param length Length to check
 * @return true if matches known optimal length
 */
bool isOptimalLength(int order, int length);

/**
 * @class Timer
 * @brief High-resolution timer for performance measurement.
 */
class Timer {
private:
    std::chrono::high_resolution_clock::time_point startTime;  ///< Start timestamp
    bool running;  ///< Timer running state

public:
    /** @brief Default constructor, timer not running. */
    Timer();

    /** @brief Starts the timer from current time. */
    void start();

    /** @brief Resets timer to current time, stops running. */
    void reset();

    /**
     * @brief Returns elapsed time since start.
     * @return Elapsed time in milliseconds, 0 if not running
     */
    double elapsedMs() const;
};

/**
 * @brief Writes search results to CSV file.
 * @param filename Output CSV file path
 * @param version Solver version number (1-4)
 * @param results Vector of search statistics
 * @param orders Vector of ruler orders tested
 */
void writeResultCSV(const std::string& filename,
                    int version,
                    const std::vector<SearchStats>& results,
                    const std::vector<int>& orders);

/**
 * @brief Prints formatted search statistics to stdout.
 * @param stats Search statistics to display
 * @param order Ruler order for header
 */
void printStats(const SearchStats& stats, int order);

/**
 * @brief Upper bound estimate for ruler length: n².
 * @param order Ruler order
 * @return Upper bound estimate, or INT_MAX for large orders
 * @note Caps at INT_MAX for order > 46340 to prevent overflow
 */
[[gnu::always_inline]]
inline int upperBoundEstimate(int order) {
    if (order > 46340) return INT_MAX;  // Prevent overflow: sqrt(INT_MAX) ~ 46340
    return order * order;
}

/**
 * @brief Theoretical lower bound (Erdős-Turán): n(n-1)/2.
 * @param order Ruler order
 * @return Lower bound estimate
 */
[[gnu::always_inline]]
inline int lowerBoundEstimate(int order) {
    return order * (order - 1) / 2;
}

/** @brief Maximum order with known optimal length (G14) */
constexpr int MAX_KNOWN_ORDER = 14;

/**
 * @brief Safe access to OPTIMAL_LENGTHS with bounds checking.
 * @param order Ruler order (2-14)
 * @return Known optimal length, or -1 if unknown
 */
inline int getOptimalLength(int order) {
    if (order < 2 || order > MAX_KNOWN_ORDER) {
        return -1;  // Unknown
    }
    return OPTIMAL_LENGTHS[order];
}

/**
 * @brief Parses and validates ruler order from string.
 * @param str Input string containing order value
 * @param maxOrder Maximum allowed order (default: MAX_ORDER)
 * @return Parsed order (2 to maxOrder), or -1 on error
 */
inline int parseAndValidateOrder(const char* str, int maxOrder = MAX_ORDER) {
    if (str == nullptr) return -1;

    char* endptr;
    long val = std::strtol(str, &endptr, 10);

    // Check for conversion errors
    if (*endptr != '\0' || endptr == str) {
        return -1;  // Not a valid number
    }

    // Check range (val == maxOrder is valid)
    if (val < 2 || val > maxOrder) {
        return -1;  // Out of range
    }

    return static_cast<int>(val);
}

#endif // GOLOMB_HPP
