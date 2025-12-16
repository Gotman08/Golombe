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

/**
 * @struct BitSet256
 * @brief Cache-aligned 256-bit bitset optimized for Golomb ruler difference tracking.
 *
 * Custom implementation instead of std::bitset<256> for:
 * - 32-byte alignment required by AVX2 instructions (_mm256_load_si256)
 * - Direct memory access for SIMD intrinsics
 * - O(1) bit operations with vectorized collision detection
 *
 * @note Requires 32-byte alignment for AVX2 operations.
 * @see hasCollisionAVX2() for vectorized collision detection
 */
struct alignas(32) BitSet256 {
    uint64_t words[4];  ///< Internal storage: 256 bits as 4 x 64-bit words

    /**
     * @brief Clears all 256 bits to zero.
     * @complexity O(1) - 4 assignments
     */
    inline void reset() {
        words[0] = words[1] = words[2] = words[3] = 0;
    }

    /**
     * @brief Tests if a specific bit is set.
     * @param bit Bit index (0-255)
     * @return true if the bit is set, false otherwise
     * @note Returns false for out-of-bounds indices in release mode
     */
    inline bool test(int bit) const {
        assert(bit >= 0 && bit < 256 && "BitSet256::test out of bounds");
        if (bit < 0 || bit >= 256) return false;  // Safe bounds check in release
        return (words[bit >> 6] >> (bit & 63)) & 1;
    }

    /**
     * @brief Sets a specific bit to 1.
     * @param bit Bit index (0-255)
     * @note No-op for out-of-bounds indices in release mode
     */
    inline void set(int bit) {
        assert(bit >= 0 && bit < 256 && "BitSet256::set out of bounds");
        if (bit < 0 || bit >= 256) return;  // Safe bounds check in release
        words[bit >> 6] |= (1ULL << (bit & 63));
    }

    /**
     * @brief Clears a specific bit to 0.
     * @param bit Bit index (0-255)
     * @note No-op for out-of-bounds indices in release mode
     */
    inline void clear(int bit) {
        assert(bit >= 0 && bit < 256 && "BitSet256::clear out of bounds");
        if (bit < 0 || bit >= 256) return;  // Safe bounds check in release
        words[bit >> 6] &= ~(1ULL << (bit & 63));
    }

    /**
     * @brief Copies all bits from another BitSet256.
     * @param other Source bitset to copy from
     */
    inline void copyFrom(const BitSet256& other) {
        words[0] = other.words[0];
        words[1] = other.words[1];
        words[2] = other.words[2];
        words[3] = other.words[3];
    }

#ifdef USE_AVX2
    /**
     * @brief AVX2-accelerated collision detection.
     *
     * Checks if ANY bit in 'mask' is already set in this bitset.
     * Uses _mm256_and_si256 to test 256 bits in a single CPU instruction.
     *
     * @param mask Bitset containing bits to check for collision
     * @return true if collision detected (at least one bit set in both)
     * @note Only available when compiled with USE_AVX2
     * @complexity O(1) - single SIMD instruction
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
