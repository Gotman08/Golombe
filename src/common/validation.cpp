/**
 * @file validation.cpp
 * @brief Golomb ruler validation functions
 *
 * Golomb Ruler Solver - High Performance Computing Implementation
 * Copyright (c) 2025 Nicolas Marano
 * Licensed under the MIT License. See LICENSE file for details.
 */

#include "golomb/golomb.hpp"
#include <set>
#include <iostream>
#include <sstream>

// ============================================================================
// Validation Functions
// ============================================================================

/** @copydoc isValidGolombRuler() */
bool isValidGolombRuler(const std::vector<int>& marks) {
    if (marks.empty()) return false;
    if (marks[0] != 0) return false;  // First mark must be 0

    std::set<int> differences;

    for (size_t i = 0; i < marks.size(); ++i) {
        for (size_t j = i + 1; j < marks.size(); ++j) {
            int diff = marks[j] - marks[i];
            if (diff <= 0) return false;  // Marks must be strictly increasing
            if (differences.count(diff)) {
                return false;  // Duplicate difference found
            }
            differences.insert(diff);
        }
    }
    return true;
}

/** @copydoc isValidGolombRulerFast() */
bool isValidGolombRulerFast(const std::vector<int>& marks, std::bitset<MAX_LENGTH>& usedDiffs) {
    if (marks.empty()) return false;
    if (marks[0] != 0) return false;

    usedDiffs.reset();

    for (size_t i = 0; i < marks.size(); ++i) {
        for (size_t j = i + 1; j < marks.size(); ++j) {
            int diff = marks[j] - marks[i];
            if (diff <= 0 || diff >= MAX_LENGTH) return false;
            if (usedDiffs.test(diff)) {
                return false;
            }
            usedDiffs.set(diff);
        }
    }
    return true;
}

/** @copydoc isOptimalLength() */
bool isOptimalLength(int order, int length) {
    if (order < 1 || order > 14) return false;
    return length == OPTIMAL_LENGTHS[order];
}

// ============================================================================
// GolombRuler Methods
// ============================================================================

/** @copydoc GolombRuler::toString() */
std::string GolombRuler::toString() const {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < marks.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << marks[i];
    }
    oss << "]";
    return oss.str();
}

/** @copydoc GolombRuler::print() */
void GolombRuler::print() const {
    std::cout << "Order " << order << ", Length " << length << ": " << toString() << '\n';
}

/** @copydoc GolombRuler::isValid() */
bool GolombRuler::isValid() const {
    return isValidGolombRuler(marks);
}
