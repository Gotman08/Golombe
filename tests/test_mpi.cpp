/**
 * @file test_mpi.cpp
 * @brief MPI-specific tests for Golomb Ruler Solver v3/v4
 *
 * Golomb Ruler Solver - High Performance Computing Implementation
 * Copyright (c) 2025 Nicolas Marano
 * Licensed under the MIT License. See LICENSE file for details.
 *
 * Tests to validate:
 * - MPI initialization and finalization
 * - Message passing correctness
 * - Bound propagation across ranks
 * - Work distribution fairness
 *
 * Usage:
 *   mpirun -np 4 ./test_mpi
 */

#include <mpi.h>
#include "golomb/golomb.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

// Test counters (global for simplicity in MPI context)
int tests_passed = 0;
int tests_failed = 0;
int rank, size;

#define TEST(name, condition) do { \
    if (condition) { \
        if (rank == 0) std::cout << "[PASS] " << name << std::endl; \
        tests_passed++; \
    } else { \
        if (rank == 0) std::cout << "[FAIL] " << name << std::endl; \
        tests_failed++; \
    } \
} while(0)

#define RANK0_PRINT(msg) do { \
    if (rank == 0) std::cout << msg << std::endl; \
} while(0)

// ============================================================
// Test: MPI Thread Support
// ============================================================

void test_mpi_thread_support() {
    RANK0_PRINT("\n=== Testing MPI Thread Support ===");

    int provided;
    MPI_Query_thread(&provided);

    const char* level_names[] = {
        "MPI_THREAD_SINGLE",
        "MPI_THREAD_FUNNELED",
        "MPI_THREAD_SERIALIZED",
        "MPI_THREAD_MULTIPLE"
    };

    if (rank == 0) {
        std::cout << "MPI thread level: " << level_names[provided] << std::endl;
    }

    TEST("MPI supports at least THREAD_FUNNELED", provided >= MPI_THREAD_FUNNELED);
}

// ============================================================
// Test: Point-to-Point Communication
// ============================================================

void test_point_to_point() {
    RANK0_PRINT("\n=== Testing Point-to-Point Communication ===");

    const int TAG_TEST = 99;

    if (size >= 2) {
        if (rank == 0) {
            // Send test data to rank 1
            int data = 42;
            MPI_Send(&data, 1, MPI_INT, 1, TAG_TEST, MPI_COMM_WORLD);

            // Receive response
            int response;
            MPI_Recv(&response, 1, MPI_INT, 1, TAG_TEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            TEST("Send/Recv roundtrip", response == data + 1);
        } else if (rank == 1) {
            // Receive and respond
            int data;
            MPI_Recv(&data, 1, MPI_INT, 0, TAG_TEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            data++;
            MPI_Send(&data, 1, MPI_INT, 0, TAG_TEST, MPI_COMM_WORLD);
        }
    } else {
        TEST("Point-to-point (skipped, need 2+ ranks)", true);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================
// Test: Collective Operations
// ============================================================

void test_collective_operations() {
    RANK0_PRINT("\n=== Testing Collective Operations ===");

    // Test MPI_Allreduce (used for bound synchronization)
    int localBound = 100 - rank;  // Each rank has different bound
    int globalMin;

    MPI_Allreduce(&localBound, &globalMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    TEST("Allreduce finds minimum", globalMin == 100 - (size - 1));

    // Test MPI_Reduce (used for statistics gathering)
    uint64_t localNodes = rank * 1000 + 500;
    uint64_t totalNodes;

    MPI_Reduce(&localNodes, &totalNodes, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        uint64_t expected = 0;
        for (int i = 0; i < size; ++i) {
            expected += i * 1000 + 500;
        }
        TEST("Reduce sums correctly", totalNodes == expected);
    }

    // Test MPI_Bcast (used for solution distribution)
    int solution[20];
    if (rank == 0) {
        for (int i = 0; i < 20; ++i) solution[i] = i * 2;
    }

    MPI_Bcast(solution, 20, MPI_INT, 0, MPI_COMM_WORLD);

    bool bcastCorrect = true;
    for (int i = 0; i < 20; ++i) {
        if (solution[i] != i * 2) bcastCorrect = false;
    }
    TEST("Broadcast distributes data correctly", bcastCorrect);
}

// ============================================================
// Test: Non-blocking Communication (Bound Propagation)
// ============================================================

void test_nonblocking_communication() {
    RANK0_PRINT("\n=== Testing Non-blocking Communication ===");

    if (size < 2) {
        TEST("Non-blocking (skipped, need 2+ ranks)", true);
        return;
    }

    const int TAG_BOUND = 4;
    int bound = 100 - rank;

    // Non-blocking send to next rank (ring topology)
    int nextRank = (rank + 1) % size;
    MPI_Request sendReq;
    MPI_Isend(&bound, 1, MPI_INT, nextRank, TAG_BOUND, MPI_COMM_WORLD, &sendReq);

    // Non-blocking receive from previous rank
    int prevRank = (rank - 1 + size) % size;
    int receivedBound;
    MPI_Request recvReq;
    MPI_Irecv(&receivedBound, 1, MPI_INT, prevRank, TAG_BOUND, MPI_COMM_WORLD, &recvReq);

    // Wait for both
    MPI_Wait(&sendReq, MPI_STATUS_IGNORE);
    MPI_Wait(&recvReq, MPI_STATUS_IGNORE);

    int expectedBound = 100 - prevRank;
    TEST("Non-blocking ring communication", receivedBound == expectedBound);

    MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================
// Test: Probe for Incoming Messages
// ============================================================

void test_message_probing() {
    RANK0_PRINT("\n=== Testing Message Probing ===");

    if (size < 2) {
        TEST("Message probing (skipped, need 2+ ranks)", true);
        return;
    }

    const int TAG_PROBE = 100;

    if (rank == 0) {
        // Send message to self via rank 1
        int data = 12345;
        MPI_Send(&data, 1, MPI_INT, 1, TAG_PROBE, MPI_COMM_WORLD);

        // Wait for response
        int response;
        MPI_Recv(&response, 1, MPI_INT, 1, TAG_PROBE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        TEST("Probe-based receive", response == data * 2);
    } else if (rank == 1) {
        // Use Iprobe to check for message
        int flag = 0;
        MPI_Status status;
        int attempts = 0;
        while (!flag && attempts < 1000) {
            MPI_Iprobe(0, TAG_PROBE, MPI_COMM_WORLD, &flag, &status);
            attempts++;
        }

        if (flag) {
            int data;
            MPI_Recv(&data, 1, MPI_INT, 0, TAG_PROBE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            data *= 2;
            MPI_Send(&data, 1, MPI_INT, 0, TAG_PROBE, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================
// Test: Hypercube Topology
// ============================================================

void test_hypercube_topology() {
    RANK0_PRINT("\n=== Testing Hypercube Topology ===");

    // Hypercube dimension calculation
    auto hypercubeDim = [](int s) {
        int d = 0;
        while (s > 1) { s /= 2; d++; }
        return d;
    };

    auto isPowerOfTwo = [](int n) {
        return n > 0 && (n & (n - 1)) == 0;
    };

    // Get hypercube neighbors
    auto getNeighbors = [&](int r, int s) {
        std::vector<int> neighbors;
        int d = hypercubeDim(s);
        for (int i = 0; i < d; ++i) {
            int neighbor = r ^ (1 << i);
            if (neighbor < s) neighbors.push_back(neighbor);
        }
        return neighbors;
    };

    std::vector<int> myNeighbors = getNeighbors(rank, size);

    if (rank == 0) {
        std::cout << "Hypercube dimension: " << hypercubeDim(size) << std::endl;
        std::cout << "Is power of 2: " << (isPowerOfTwo(size) ? "yes" : "no") << std::endl;
    }

    // Each rank should have at most log2(size) neighbors
    int maxNeighbors = hypercubeDim(size);
    TEST("Correct neighbor count", (int)myNeighbors.size() <= maxNeighbors);

    // Test message passing along hypercube edges
    if (size >= 2) {
        const int TAG_HYPERCUBE = 101;
        int sendVal = rank;

        // Send to all neighbors
        std::vector<MPI_Request> sendReqs(myNeighbors.size());
        for (size_t i = 0; i < myNeighbors.size(); ++i) {
            MPI_Isend(&sendVal, 1, MPI_INT, myNeighbors[i], TAG_HYPERCUBE,
                      MPI_COMM_WORLD, &sendReqs[i]);
        }

        // Receive from all neighbors
        std::vector<int> recvVals(myNeighbors.size());
        std::vector<MPI_Request> recvReqs(myNeighbors.size());
        for (size_t i = 0; i < myNeighbors.size(); ++i) {
            MPI_Irecv(&recvVals[i], 1, MPI_INT, myNeighbors[i], TAG_HYPERCUBE,
                      MPI_COMM_WORLD, &recvReqs[i]);
        }

        // Wait for all
        if (!sendReqs.empty()) MPI_Waitall(sendReqs.size(), sendReqs.data(), MPI_STATUSES_IGNORE);
        if (!recvReqs.empty()) MPI_Waitall(recvReqs.size(), recvReqs.data(), MPI_STATUSES_IGNORE);

        // Verify received values match expected neighbors
        bool correctNeighbors = true;
        for (size_t i = 0; i < myNeighbors.size(); ++i) {
            if (recvVals[i] != myNeighbors[i]) correctNeighbors = false;
        }
        TEST("Hypercube neighbor communication", correctNeighbors);
    } else {
        TEST("Hypercube (skipped, need 2+ ranks)", true);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================
// Test: Work Distribution Fairness
// ============================================================

void test_work_distribution() {
    RANK0_PRINT("\n=== Testing Work Distribution ===");

    // Simulate subtree distribution (like in v4)
    int totalSubtrees = 100;
    int subtreesPerRank = totalSubtrees / size;
    int remainder = totalSubtrees % size;

    int myStart, myEnd;
    if (rank < remainder) {
        myStart = rank * (subtreesPerRank + 1);
        myEnd = myStart + subtreesPerRank + 1;
    } else {
        myStart = remainder * (subtreesPerRank + 1) + (rank - remainder) * subtreesPerRank;
        myEnd = myStart + subtreesPerRank;
    }

    int myCount = myEnd - myStart;

    // Gather all counts to rank 0
    std::vector<int> allCounts(size);
    MPI_Gather(&myCount, 1, MPI_INT, allCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Check distribution is fair (difference of at most 1)
        int minCount = *std::min_element(allCounts.begin(), allCounts.end());
        int maxCount = *std::max_element(allCounts.begin(), allCounts.end());
        TEST("Work distribution is balanced", maxCount - minCount <= 1);

        // Check total is correct
        int sum = 0;
        for (int c : allCounts) sum += c;
        TEST("All work is distributed", sum == totalSubtrees);
    }
}

// ============================================================
// Test: Custom MPI Datatype (Subtree-like)
// ============================================================

void test_custom_datatype() {
    RANK0_PRINT("\n=== Testing Custom MPI Datatype ===");

    // Simplified subtree structure
    struct TestSubtree {
        int marks[20];
        int markCount;
        int bound;
    };

    // Create MPI datatype
    MPI_Datatype subtreeType;
    int blocklengths[] = {20, 1, 1};
    MPI_Aint displacements[3];
    displacements[0] = offsetof(TestSubtree, marks);
    displacements[1] = offsetof(TestSubtree, markCount);
    displacements[2] = offsetof(TestSubtree, bound);
    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_INT};

    MPI_Type_create_struct(3, blocklengths, displacements, types, &subtreeType);
    MPI_Type_commit(&subtreeType);

    if (size >= 2) {
        if (rank == 0) {
            TestSubtree sub;
            for (int i = 0; i < 20; ++i) sub.marks[i] = i;
            sub.markCount = 5;
            sub.bound = 42;
            MPI_Send(&sub, 1, subtreeType, 1, 0, MPI_COMM_WORLD);
        } else if (rank == 1) {
            TestSubtree sub;
            MPI_Recv(&sub, 1, subtreeType, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            bool correct = (sub.markCount == 5 && sub.bound == 42);
            for (int i = 0; i < 5 && correct; ++i) {
                if (sub.marks[i] != i) correct = false;
            }

            // Report to rank 0
            int result = correct ? 1 : 0;
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        if (rank == 0) {
            int result;
            MPI_Recv(&result, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            TEST("Custom datatype transmission", result == 1);
        }
    } else {
        TEST("Custom datatype (skipped, need 2+ ranks)", true);
    }

    MPI_Type_free(&subtreeType);
    MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "============================================" << std::endl;
        std::cout << "Golomb Ruler Solver - MPI Tests" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "MPI ranks: " << size << std::endl;
    }

    test_mpi_thread_support();
    test_point_to_point();
    test_collective_operations();
    test_nonblocking_communication();
    test_message_probing();
    test_hypercube_topology();
    test_work_distribution();
    test_custom_datatype();

    // Gather final results
    int global_passed, global_failed;
    MPI_Reduce(&tests_passed, &global_passed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tests_failed, &global_failed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n============================================" << std::endl;
        std::cout << "Results: " << global_passed << " passed, "
                  << global_failed << " failed" << std::endl;
        std::cout << "============================================" << std::endl;
    }

    MPI_Finalize();

    return (rank == 0 && global_failed > 0) ? 1 : 0;
}
