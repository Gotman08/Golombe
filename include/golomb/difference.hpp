/**
 * @file difference.hpp
 * @brief Common difference checking functions for Golomb Ruler search
 *
 * Golomb Ruler Solver - High Performance Computing Implementation
 * Copyright (c) 2025 Nicolas Marano
 * Licensed under the MIT License. See LICENSE file for details.
 *
 * This header provides template implementations of difference checking
 * to eliminate code duplication across v1, v2, v3, and v4 implementations.
 *
 * Requirements for StateType:
 *   - int marks[MAX_ORDER]     : Array of mark positions
 *   - int markCount            : Number of marks currently placed
 *   - BitSet256 usedDiffs      : Bitset tracking used differences
 */

#ifndef GOLOMB_DIFFERENCE_HPP
#define GOLOMB_DIFFERENCE_HPP

#include "golomb.hpp"
#include "bitset256.hpp"

#ifdef USE_AVX2
#include <immintrin.h>
#endif

namespace golomb {

/**
 * @brief Check if a new position creates valid differences (scalar version)
 *
 * @tparam StateType Type with marks[], markCount, and usedDiffs members
 * @param state Current search state
 * @param pos New position to test
 * @param tempDiffs Output array for new differences
 * @param diffCount Output count of new differences
 * @return true if position is valid (no collisions), false otherwise
 */
template<typename StateType>
[[gnu::always_inline]]
inline bool checkDifferencesScalar(StateType& state, int pos, int* tempDiffs, int& diffCount) {
    diffCount = 0;
    for (int i = 0; i < state.markCount; ++i) {
        int diff = pos - state.marks[i];
        if (diff >= MAX_LENGTH || state.usedDiffs.test(diff)) {
            return false;
        }
        tempDiffs[diffCount++] = diff;
    }
    return true;
}

#ifdef USE_AVX2
/**
 * @brief Check if a new position creates valid differences (AVX2 version)
 *
 * Uses AVX2 SIMD instructions for vectorized difference calculation
 * and collision detection. Falls back to scalar for remainder elements.
 *
 * @tparam StateType Type with marks[], markCount, and usedDiffs members
 * @param state Current search state
 * @param pos New position to test
 * @param tempDiffs Output array for new differences
 * @param diffCount Output count of new differences
 * @return true if position is valid (no collisions), false otherwise
 */
template<typename StateType>
[[gnu::always_inline]]
inline bool checkDifferencesAVX2(StateType& state, int pos, int* tempDiffs, int& diffCount) {
    __m256i vpos = _mm256_set1_epi32(pos);

    // Phase 1: Calculate all differences using AVX2
    alignas(32) int allDiffs[MAX_ORDER];
    int totalDiffs = 0;
    int i = 0;

    // Process 8 marks at a time with AVX2
    for (; i + 8 <= state.markCount; i += 8) {
        __m256i vmarks = _mm256_loadu_si256((__m256i*)&state.marks[i]);
        __m256i vdiffs = _mm256_sub_epi32(vpos, vmarks);
        _mm256_storeu_si256((__m256i*)&allDiffs[totalDiffs], vdiffs);
        totalDiffs += 8;
    }

    // Handle remaining marks (scalar)
    for (; i < state.markCount; ++i) {
        allDiffs[totalDiffs++] = pos - state.marks[i];
    }

    // Phase 2: Build collision mask and check bounds
    BitSet256 checkMask;
    checkMask.reset();

    for (int j = 0; j < totalDiffs; ++j) {
        int d = allDiffs[j];
        if (d >= MAX_LENGTH) {
            return false;
        }
        checkMask.set(d);
    }

    // Phase 3: Vectorized collision detection
    if (state.usedDiffs.hasCollisionAVX2(checkMask)) {
        return false;
    }

    // Phase 4: Copy differences to output
    diffCount = totalDiffs;
    for (int j = 0; j < totalDiffs; ++j) {
        tempDiffs[j] = allDiffs[j];
    }

    return true;
}

/**
 * @brief Dispatch to AVX2 or scalar based on mark count
 *
 * @tparam StateType Type with marks[], markCount, and usedDiffs members
 * @param state Current search state
 * @param pos New position to test
 * @param tempDiffs Output array for new differences
 * @param diffCount Output count of new differences
 * @param useAVX2 Whether to use AVX2 when beneficial
 * @return true if position is valid (no collisions), false otherwise
 */
template<typename StateType>
[[gnu::always_inline]]
inline bool checkDifferences(StateType& state, int pos, int* tempDiffs, int& diffCount, bool useAVX2 = true) {
    if (useAVX2 && state.markCount >= 4) {
        return checkDifferencesAVX2(state, pos, tempDiffs, diffCount);
    }
    return checkDifferencesScalar(state, pos, tempDiffs, diffCount);
}

#else  // !USE_AVX2

/**
 * @brief Dispatch to scalar (AVX2 not available)
 */
template<typename StateType>
[[gnu::always_inline]]
inline bool checkDifferences(StateType& state, int pos, int* tempDiffs, int& diffCount, bool /*useAVX2*/ = true) {
    return checkDifferencesScalar(state, pos, tempDiffs, diffCount);
}

#endif  // USE_AVX2

}  // namespace golomb

#endif  // GOLOMB_DIFFERENCE_HPP
