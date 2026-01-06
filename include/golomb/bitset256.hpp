/**
 * @file bitset256.hpp
 * @brief Cache-aligned 256-bit bitset with AVX2 support
 *
 * Golomb Ruler Solver - High Performance Computing Implementation
 * Copyright (c) 2025 Nicolas Marano
 * Licensed under the MIT License. See LICENSE file for details.
 *
 * Optimized for Golomb ruler difference tracking:
 * - 32-byte aligned for AVX2 operations
 * - O(1) test/set/clear operations
 * - AVX2-accelerated collision detection
 */

#ifndef BITSET256_HPP
#define BITSET256_HPP

#include <cstdint>
#include <cassert>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

struct alignas(32) BitSet256 {
    uint64_t words[4];  // 256 bits = 4 x 64-bit words

    inline void reset() {
        words[0] = words[1] = words[2] = words[3] = 0;
    }

    inline bool test(int bit) const {
        assert(bit >= 0 && bit < 256 && "BitSet256::test out of bounds");
        if (bit < 0 || bit >= 256) return false;  // Safe bounds check in release
        return (words[bit >> 6] >> (bit & 63)) & 1;
    }

    inline void set(int bit) {
        assert(bit >= 0 && bit < 256 && "BitSet256::set out of bounds");
        if (bit < 0 || bit >= 256) return;  // Safe bounds check in release
        words[bit >> 6] |= (1ULL << (bit & 63));
    }

    inline void clear(int bit) {
        assert(bit >= 0 && bit < 256 && "BitSet256::clear out of bounds");
        if (bit < 0 || bit >= 256) return;  // Safe bounds check in release
        words[bit >> 6] &= ~(1ULL << (bit & 63));
    }

    // Copy from another BitSet256
    inline void copyFrom(const BitSet256& other) {
        words[0] = other.words[0];
        words[1] = other.words[1];
        words[2] = other.words[2];
        words[3] = other.words[3];
    }

#ifdef USE_AVX2
    /**
     * AVX2-accelerated collision detection
     * Checks if ANY of the bits in 'mask' are already set in this bitset
     * Uses _mm256_and_si256 to check 256 bits in a single operation
     * Returns true if collision detected (at least one bit is set in both)
     */
    inline bool hasCollisionAVX2(const BitSet256& mask) const {
        __m256i used = _mm256_load_si256((__m256i*)words);
        __m256i check = _mm256_load_si256((__m256i*)mask.words);
        __m256i collision = _mm256_and_si256(used, check);

        // _mm256_testz returns 1 if all bits are zero, 0 otherwise
        return !_mm256_testz_si256(collision, collision);
    }
#endif
};

#endif // BITSET256_HPP
