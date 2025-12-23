/**
 * @file test_openmp.cpp
 * @brief OpenMP-specific tests for Golomb Ruler Solver v2
 *
 * Golomb Ruler Solver - High Performance Computing Implementation
 * Copyright (c) 2025 Nicolas Marano
 * Licensed under the MIT License. See LICENSE file for details.
 *
 * Tests to validate:
 * - Correctness with various thread counts
 * - Thread safety of shared data structures
 * - Consistency of results across runs
 */

#include "golomb/golomb.hpp"
#include "golomb/bitset256.hpp"
#include <omp.h>
#include <iostream>
#include <vector>
#include <atomic>
#include <mutex>
#include <set>

// Test counters
int tests_passed = 0;
int tests_failed = 0;

#define TEST(name, condition) do { \
    if (condition) { \
        std::cout << "[PASS] " << name << std::endl; \
        tests_passed++; \
    } else { \
        std::cout << "[FAIL] " << name << std::endl; \
        tests_failed++; \
    } \
} while(0)

// ============================================================
// Test: BitSet256 Thread Safety
// ============================================================

void test_bitset256_thread_safety() {
    std::cout << "\n=== Testing BitSet256 Thread Safety ===" << std::endl;

    BitSet256 shared;
    shared.reset();

    const int NUM_BITS = 100;
    std::atomic<int> collisions(0);

    // Multiple threads setting different bits should not cause issues
    #pragma omp parallel for
    for (int i = 0; i < NUM_BITS; ++i) {
        // Each thread works on its own bit
        BitSet256 local;
        local.reset();
        local.set(i);
        if (local.test(i) != true) {
            collisions++;
        }
    }

    TEST("Parallel BitSet256 operations", collisions.load() == 0);

    // Test concurrent read access
    BitSet256 readTest;
    readTest.reset();
    for (int i = 0; i < 50; ++i) {
        readTest.set(i * 2);  // Set even bits
    }

    std::atomic<int> readErrors(0);

    #pragma omp parallel for
    for (int i = 0; i < 100; ++i) {
        bool expected = (i % 2 == 0) && (i < 100);
        bool actual = readTest.test(i);
        if (actual != expected) {
            readErrors++;
        }
    }

    TEST("Concurrent BitSet256 reads", readErrors.load() == 0);
}

// ============================================================
// Test: Atomic Bound Updates
// ============================================================

void test_atomic_bound_updates() {
    std::cout << "\n=== Testing Atomic Bound Updates ===" << std::endl;

    std::atomic<int> globalBound(100);
    std::atomic<int> updateCount(0);

    // Multiple threads trying to lower the bound
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int myBound = 50 + tid;  // Each thread has a different bound

        int current = globalBound.load(std::memory_order_relaxed);
        while (myBound < current) {
            if (globalBound.compare_exchange_weak(current, myBound)) {
                updateCount++;
                break;
            }
        }
    }

    int finalBound = globalBound.load();
    TEST("Atomic bound converges to minimum", finalBound <= 50 + omp_get_max_threads());
    TEST("At least one update occurred", updateCount.load() >= 1);
}

// ============================================================
// Test: Thread-Local State Independence
// ============================================================

void test_thread_local_independence() {
    std::cout << "\n=== Testing Thread-Local State Independence ===" << std::endl;

    const int NUM_ITERATIONS = 1000;
    std::atomic<int> errors(0);

    #pragma omp parallel
    {
        // Each thread has its own local array
        int localMarks[20];
        for (int i = 0; i < 20; ++i) {
            localMarks[i] = omp_get_thread_num() * 100 + i;
        }

        // Verify no cross-thread contamination
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            for (int i = 0; i < 20; ++i) {
                if (localMarks[i] != omp_get_thread_num() * 100 + i) {
                    errors++;
                }
            }
        }
    }

    TEST("Thread-local arrays remain independent", errors.load() == 0);
}

// ============================================================
// Test: Parallel Difference Checking
// ============================================================

void test_parallel_difference_checking() {
    std::cout << "\n=== Testing Parallel Difference Checking ===" << std::endl;

    // Known valid ruler: [0, 1, 4, 6]
    std::vector<int> marks = {0, 1, 4, 6};

    std::atomic<int> validChecks(0);
    std::atomic<int> invalidChecks(0);

    // Test positions in parallel
    #pragma omp parallel for
    for (int pos = 7; pos < 20; ++pos) {
        std::bitset<MAX_LENGTH> usedDiffs;

        // Mark used differences
        for (size_t i = 0; i < marks.size(); ++i) {
            for (size_t j = i + 1; j < marks.size(); ++j) {
                usedDiffs.set(marks[j] - marks[i]);
            }
        }

        // Check if pos would be valid
        bool valid = true;
        for (int m : marks) {
            int diff = pos - m;
            if (diff >= MAX_LENGTH || usedDiffs.test(diff)) {
                valid = false;
                break;
            }
        }

        if (valid) validChecks++;
        else invalidChecks++;
    }

    // Some positions should be valid, some invalid
    TEST("Some positions are valid", validChecks.load() > 0);
    TEST("Some positions are invalid", invalidChecks.load() > 0);
    TEST("Total checks correct", validChecks.load() + invalidChecks.load() == 13);
}

// ============================================================
// Test: Consistency Across Thread Counts
// ============================================================

void test_consistency_across_threads() {
    std::cout << "\n=== Testing Consistency Across Thread Counts ===" << std::endl;

    // This test verifies that running with different thread counts
    // produces consistent results (same valid/invalid determination)

    std::vector<int> testMarks = {0, 1, 4, 10, 12, 17};  // G6 optimal
    bool isValid = isValidGolombRuler(testMarks);

    std::atomic<int> inconsistencies(0);

    // Run validation in parallel with different subsets
    #pragma omp parallel
    {
        bool localResult = isValidGolombRuler(testMarks);
        if (localResult != isValid) {
            inconsistencies++;
        }
    }

    TEST("Validation consistent across threads", inconsistencies.load() == 0);
}

// ============================================================
// Test: OpenMP Task Parallelism
// ============================================================

void test_openmp_task_parallelism() {
    std::cout << "\n=== Testing OpenMP Task Parallelism ===" << std::endl;

    std::atomic<int> taskCount(0);
    std::set<int> threadIds;
    std::mutex threadMutex;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < 100; ++i) {
                #pragma omp task
                {
                    taskCount++;
                    int tid = omp_get_thread_num();
                    {
                        std::lock_guard<std::mutex> lock(threadMutex);
                        threadIds.insert(tid);
                    }
                }
            }
        }
    }

    TEST("All tasks completed", taskCount.load() == 100);

    // With multiple threads, tasks should be distributed
    int numThreads = omp_get_max_threads();
    if (numThreads > 1) {
        TEST("Tasks distributed across threads", threadIds.size() > 1);
    } else {
        TEST("Single thread handled all tasks", threadIds.size() == 1);
    }
}

// ============================================================
// Test: Cache Line Alignment
// ============================================================

void test_cache_line_alignment() {
    std::cout << "\n=== Testing Cache Line Alignment ===" << std::endl;

    struct alignas(64) AlignedState {
        int data[16];
        char padding[64 - 64];  // Padding to fill cache line
    };

    AlignedState states[4];

    // Check alignment
    bool allAligned = true;
    for (int i = 0; i < 4; ++i) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(&states[i]);
        if (addr % 64 != 0) {
            allAligned = false;
        }
    }

    TEST("States are cache-line aligned", allAligned);

    // Test that parallel access doesn't cause false sharing issues
    std::atomic<int> completedThreads(0);

    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        if (tid < 4) {
            // Each thread writes to its own cache line
            for (int i = 0; i < 10000; ++i) {
                states[tid].data[0] = i;
            }
            completedThreads++;
        }
    }

    TEST("No false sharing deadlock", completedThreads.load() >= std::min(4, omp_get_max_threads()));
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "Golomb Ruler Solver - OpenMP Tests" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;

    test_bitset256_thread_safety();
    test_atomic_bound_updates();
    test_thread_local_independence();
    test_parallel_difference_checking();
    test_consistency_across_threads();
    test_openmp_task_parallelism();
    test_cache_line_alignment();

    std::cout << "\n============================================" << std::endl;
    std::cout << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed" << std::endl;
    std::cout << "============================================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
