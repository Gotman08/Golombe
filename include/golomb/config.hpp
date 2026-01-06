/**
 * @file config.hpp
 * @brief Centralized configuration constants for Golomb Ruler Solver
 *
 * Golomb Ruler Solver - High Performance Computing Implementation
 * Copyright (c) 2025 Nicolas Marano
 * Licensed under the MIT License. See LICENSE file for details.
 *
 * All tuning parameters and magic numbers should be defined here
 * to ensure consistency across versions and ease of maintenance.
 */

#ifndef GOLOMB_CONFIG_HPP
#define GOLOMB_CONFIG_HPP

namespace golomb {
namespace config {

// ============================================================================
// Core Algorithm Limits
// ============================================================================

/// Maximum ruler length for bitset sizing
/// G11 = 72, G12 = 85, using 256 for safety margin
constexpr int MAX_LENGTH = 256;

/// Maximum supported order (number of marks)
constexpr int MAX_ORDER = 20;

/// Maximum order with known optimal length (from OEIS A003022)
constexpr int MAX_KNOWN_ORDER = 14;

// ============================================================================
// Bound Check Intervals
// ============================================================================

/// Default interval for checking atomic bound updates (nodes between checks)
/// Higher = better throughput but slower bound propagation
constexpr int BOUND_CHECK_INTERVAL_DEFAULT = 8192;

/// Bound check interval for v3 (master/worker MPI)
/// Higher interval due to MPI communication overhead
constexpr int BOUND_CHECK_INTERVAL_V3 = 50000;

/// Bound check interval for v4 (hypercube MPI)
/// Lower interval for faster peer-to-peer bound propagation
constexpr int BOUND_CHECK_INTERVAL_V4 = 10000;

/// Local bound cache refresh interval (atomic loads between cache updates)
/// Trade-off: lower = fresher bounds, higher = less atomic contention
constexpr int LOCAL_CACHE_REFRESH_INTERVAL = 8192;

/// MPI bound check frequency (how often to probe for MPI messages)
/// In v4: check MPI every N local cache refreshes
constexpr int MPI_CHECK_FREQUENCY = 2;

// ============================================================================
// Parallelization Tuning
// ============================================================================

/// Maximum allowed thread count for input validation
constexpr int MAX_THREADS = 256;

/// Minimum thread count
constexpr int MIN_THREADS = 1;

/// Default OpenMP cutoff depths by order
/// Below cutoff: spawn tasks, at/above cutoff: sequential
constexpr int getOpenMPCutoffDepth(int order) {
    if (order <= 7) return 1;
    if (order <= 9) return 2;
    return 3;
}

/// Default prefix depth for MPI work distribution
/// Higher depth = more subtrees, better load balance, more memory
constexpr int getMPIPrefixDepth(int order) {
    if (order <= 8) return 3;
    if (order <= 10) return 4;
    if (order <= 12) return 5;
    return 6;
}

// ============================================================================
// MPI Communication Tags
// ============================================================================

constexpr int TAG_WORK = 1;     ///< Work distribution
constexpr int TAG_RESULT = 2;   ///< Result reporting
constexpr int TAG_DONE = 3;     ///< Worker termination
constexpr int TAG_BOUND = 4;    ///< Bound propagation

// ============================================================================
// Performance Hints
// ============================================================================

/// Minimum mark count for AVX2 to be beneficial
/// Below this, scalar is often faster due to setup overhead
constexpr int AVX2_MIN_MARKS = 4;

/// Cache line size for alignment
constexpr int CACHE_LINE_SIZE = 64;

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validate thread count within allowed range
constexpr bool isValidThreadCount(int threads) {
    return threads >= MIN_THREADS && threads <= MAX_THREADS;
}

/// Validate order within supported range
constexpr bool isValidOrder(int order) {
    return order >= 2 && order <= MAX_ORDER;
}

}  // namespace config
}  // namespace golomb

#endif  // GOLOMB_CONFIG_HPP
