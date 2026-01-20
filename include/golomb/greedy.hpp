/**
 * @file greedy.hpp
 * @brief Template-based greedy heuristic for initial bound
 *
 * Golomb Ruler Solver - High Performance Computing Implementation
 * Copyright (c) 2025 Nicolas Marano
 * Licensed under the MIT License. See LICENSE file for details.
 *
 * Works with both std::bitset<MAX_LENGTH> and custom BitSet256.
 * The greedy approach: always choose the smallest valid next position.
 */

#ifndef GREEDY_HPP
#define GREEDY_HPP

/**
 * Shared Greedy Solution Generator
 * This provides a good initial upper bound for branch and bound.
 */

#include "golomb.hpp"
#include <vector>
#include <bitset>

/**
 * Compute a greedy Golomb ruler solution.
 *
 * Template parameter DiffContainer must support:
 *   - test(int) const -> bool : check if difference exists
 *   - set(int)               : mark difference as used
 *
 * @param order Number of marks in the ruler
 * @param diffs Reference to difference container (will be modified then restored)
 * @return Vector of mark positions forming a valid Golomb ruler
 */
template<typename DiffContainer>
inline std::vector<int> computeGreedySolution(int order, DiffContainer& diffs) {
    std::vector<int> marks;
    marks.reserve(order);
    marks.push_back(0);

    // Temporary storage for differences added at each step
    std::vector<int> stepDiffs;
    stepDiffs.reserve(order);

    for (int i = 1; i < order; ++i) {
        int pos = marks.back() + 1;
        // Safety limit to prevent infinite loop in case of unexpected conditions
        const int maxPos = MAX_LENGTH * 2;

        while (pos < maxPos) {
            bool valid = true;
            stepDiffs.clear();

            for (int m : marks) {
                int diff = pos - m;
                if (diff >= MAX_LENGTH || diffs.test(diff)) {
                    valid = false;
                    break;
                }
                stepDiffs.push_back(diff);
            }

            if (valid) {
                marks.push_back(pos);
                for (int d : stepDiffs) {
                    diffs.set(d);
                }
                break;
            }
            ++pos;
        }

        // If we reached the safety limit, stop building the ruler
        if (pos >= maxPos) {
            break;
        }
    }

    return marks;
}

/**
 * Compute greedy solution and return the length (last mark position).
 * This version creates its own difference container.
 *
 * @param order Number of marks in the ruler
 * @return Length of the greedy solution (upper bound)
 */
[[gnu::always_inline]]
inline int computeGreedyBound(int order) {
    std::bitset<MAX_LENGTH> diffs;
    std::vector<int> marks = computeGreedySolution(order, diffs);
    return marks.empty() ? 0 : marks.back();
}

/**
 * Compute greedy solution using bitset and return both marks and bound.
 *
 * @param order Number of marks in the ruler
 * @return GolombRuler containing the greedy solution
 */
[[gnu::always_inline]]
inline GolombRuler computeGreedyRuler(int order) {
    std::bitset<MAX_LENGTH> diffs;
    std::vector<int> marks = computeGreedySolution(order, diffs);
    return GolombRuler(marks);
}

#endif // GREEDY_HPP
