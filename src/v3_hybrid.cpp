/**
 * @file v3_hybrid.cpp
 * @brief Hybrid MPI+OpenMP Golomb Ruler Solver (Master/Worker)
 *
 * Golomb Ruler Solver - High Performance Computing Implementation
 * Copyright (c) 2025 Nicolas Marano
 * Licensed under the MIT License. See LICENSE file for details.
 *
 * Combines MPI for inter-node distribution with OpenMP for intra-node parallelism:
 * - Master/worker MPI distribution with dynamic load balancing
 * - Hypercube topology for fast bound propagation
 * - OpenMP task parallelism within each MPI rank
 * - Bound caching to reduce synchronization overhead
 * - AVX2 SIMD optimizations
 *
 * Usage:
 *   mpirun -np 4 ./golomb_v3 12 --threads 8
 *   # 4 MPI ranks x 8 OpenMP threads = 32 effective workers
 */

#include <mpi.h>
#include <omp.h>
#include "golomb/golomb.hpp"
#include "golomb/greedy.hpp"
#include "golomb/bitset256.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <functional>
#include <iomanip>
#include <fstream>
#include <atomic>
#include <mutex>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

// ============================================================================
// MPI Tags
// ============================================================================

const int TAG_WORK = 1;
const int TAG_RESULT = 2;
const int TAG_DONE = 3;
const int TAG_BOUND = 4;

const int BOUND_CHECK_INTERVAL = 50000;

// ============================================================================
// Subtree (for MPI distribution)
// ============================================================================

struct Subtree {
    int marks[MAX_ORDER];
    int markCount;
    unsigned char usedDiffs[MAX_LENGTH / 8 + 1];
    int bestBound;

    void setDiff(int d) {
        if (d >= 0 && d < MAX_LENGTH) usedDiffs[d / 8] |= (1 << (d % 8));
    }
    bool testDiff(int d) const {
        if (d < 0 || d >= MAX_LENGTH) return false;
        return usedDiffs[d / 8] & (1 << (d % 8));
    }
    void clearDiffs() {
        std::memset(usedDiffs, 0, sizeof(usedDiffs));
    }
};

// ============================================================================
// Result (for MPI communication)
// ============================================================================

struct Result {
    int marks[MAX_ORDER];
    int length;
    int order;
    uint64_t nodesExplored;
    uint64_t nodesPruned;
    double timeMs;
};

// ============================================================================
// Thread State (cache-aligned)
// ============================================================================

struct alignas(64) ThreadState {
    int marks[MAX_ORDER];
    BitSet256 usedDiffs;
    int markCount;
    uint64_t localNodesExplored;
    uint64_t localNodesPruned;
};

// ============================================================================
// Hypercube Utilities
// ============================================================================

int hypercubeDimension(int size) {
    int d = 0, s = size;
    while (s > 1) { s /= 2; d++; }
    return d;
}

std::vector<int> getHypercubeNeighbors(int rank, int size) {
    std::vector<int> neighbors;
    int d = hypercubeDimension(size);
    for (int i = 0; i < d; ++i) {
        int neighbor = rank ^ (1 << i);
        if (neighbor < size) neighbors.push_back(neighbor);
    }
    return neighbors;
}

void broadcastBoundToNeighbors(int bound, int rank, int size) {
    for (int neighbor : getHypercubeNeighbors(rank, size)) {
        MPI_Request request;
        MPI_Isend(&bound, 1, MPI_INT, neighbor, TAG_BOUND, MPI_COMM_WORLD, &request);
        MPI_Request_free(&request);
    }
}

bool checkForBoundUpdate(int& currentBound) {
    int flag;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_BOUND, MPI_COMM_WORLD, &flag, &status);
    if (flag) {
        int newBound;
        MPI_Recv(&newBound, 1, MPI_INT, status.MPI_SOURCE, TAG_BOUND,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (newBound < currentBound) {
            currentBound = newBound;
            return true;
        }
    }
    return false;
}

// ============================================================================
// Hybrid Solver (OpenMP within MPI rank)
// ============================================================================

class HybridSolver {
private:
    int order;
    std::atomic<int> localBestLength;
    GolombRuler bestSolution;
    std::mutex solutionMutex;

    std::atomic<uint64_t> totalNodesExplored;
    std::atomic<uint64_t> totalNodesPruned;

    int* globalBound;
    int mpiRank, mpiSize;
    bool useSIMD;

public:
    HybridSolver(int n, const Subtree& subtree, int* bound, int rank, int size, bool simd = true)
        : order(n), localBestLength(subtree.bestBound),
          totalNodesExplored(0), totalNodesPruned(0),
          globalBound(bound), mpiRank(rank), mpiSize(size), useSIMD(simd) {
        bestSolution.length = subtree.bestBound;
    }

    void solve(const Subtree& subtree) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                ThreadState initialState;
                initializeFromSubtree(initialState, subtree);

                int startDepth = subtree.markCount;
                int lastMark = subtree.marks[subtree.markCount - 1];
                int currentBest = localBestLength.load(std::memory_order_relaxed);
                int maxPos = currentBest - 1;

                for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
                    int remainingMarks = order - startDepth - 1;
                    if (pos + remainingMarks >= currentBest) continue;

                    bool valid = true;
                    std::vector<int> newDiffs;
                    for (int i = 0; i < initialState.markCount && valid; ++i) {
                        int diff = pos - initialState.marks[i];
                        if (diff >= MAX_LENGTH || initialState.usedDiffs.test(diff)) {
                            valid = false;
                        } else {
                            newDiffs.push_back(diff);
                        }
                    }

                    if (valid) {
                        #pragma omp task firstprivate(pos, newDiffs)
                        {
                            ThreadState state;
                            initializeFromSubtree(state, subtree);

                            state.marks[state.markCount++] = pos;
                            for (int d : newDiffs) state.usedDiffs.set(d);

                            branchAndBound(state, startDepth + 1);

                            totalNodesExplored.fetch_add(state.localNodesExplored, std::memory_order_relaxed);
                            totalNodesPruned.fetch_add(state.localNodesPruned, std::memory_order_relaxed);
                        }
                    }
                }
            }
        }
    }

    Result getResult() const {
        Result r;
        r.order = order;
        r.length = bestSolution.length;
        r.nodesExplored = totalNodesExplored.load();
        r.nodesPruned = totalNodesPruned.load();
        for (int i = 0; i < order && i < (int)bestSolution.marks.size(); ++i)
            r.marks[i] = bestSolution.marks[i];
        return r;
    }

private:
    void initializeFromSubtree(ThreadState& state, const Subtree& subtree) {
        state.markCount = subtree.markCount;
        for (int i = 0; i < subtree.markCount; ++i)
            state.marks[i] = subtree.marks[i];

        state.usedDiffs.reset();
        for (int d = 0; d < MAX_LENGTH; ++d)
            if (subtree.testDiff(d)) state.usedDiffs.set(d);

        state.localNodesExplored = 0;
        state.localNodesPruned = 0;
    }

    void branchAndBound(ThreadState& state, int depth) {
        state.localNodesExplored++;

        // Bound caching with periodic MPI check
        static thread_local int cachedBound = INT_MAX;
        static thread_local int checkCounter = 0;

        if (++checkCounter >= 16384) {
            cachedBound = localBestLength.load(std::memory_order_relaxed);
            checkCounter = 0;

            // Periodic MPI bound check (less frequent)
            if (state.localNodesExplored % BOUND_CHECK_INTERVAL == 0) {
                #pragma omp critical(mpi_bound)
                {
                    if (checkForBoundUpdate(*globalBound)) {
                        int newBound = *globalBound;
                        int current = localBestLength.load(std::memory_order_relaxed);
                        while (newBound < current) {
                            if (localBestLength.compare_exchange_weak(current, newBound))
                                break;
                        }
                        cachedBound = newBound;
                    }
                }
            }
        }

        if (depth == order) [[unlikely]] {
            int length = state.marks[state.markCount - 1];
            updateBestSolution(length, state);
            cachedBound = localBestLength.load(std::memory_order_relaxed);
            return;
        }

        int lastMark = state.marks[state.markCount - 1];
        int currentBest = std::min(cachedBound, *globalBound);
        int maxPos = currentBest - 1;

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            int remainingMarks = order - depth - 1;
            if (pos + remainingMarks >= currentBest) {
                state.localNodesPruned++;
                continue;
            }

            int tempDiffs[MAX_ORDER];
            int newDiffCount = 0;
            bool valid;

#ifdef USE_AVX2
            if (useSIMD && state.markCount >= 4) {
                valid = checkDifferencesAVX2(state, pos, tempDiffs, newDiffCount);
            } else {
                valid = checkDifferencesScalar(state, pos, tempDiffs, newDiffCount);
            }
#else
            valid = checkDifferencesScalar(state, pos, tempDiffs, newDiffCount);
#endif

            if (valid) {
                state.marks[state.markCount++] = pos;
                for (int i = 0; i < newDiffCount; ++i)
                    state.usedDiffs.set(tempDiffs[i]);

                branchAndBound(state, depth + 1);

                state.markCount--;
                for (int i = 0; i < newDiffCount; ++i)
                    state.usedDiffs.clear(tempDiffs[i]);

                currentBest = std::min(cachedBound, *globalBound);
                maxPos = currentBest - 1;
            }
        }
    }

    inline bool checkDifferencesScalar(ThreadState& state, int pos, int* tempDiffs, int& diffCount) {
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
    inline bool checkDifferencesAVX2(ThreadState& state, int pos, int* tempDiffs, int& diffCount) {
        __m256i vpos = _mm256_set1_epi32(pos);

        alignas(32) int allDiffs[MAX_ORDER];
        int totalDiffs = 0;
        int i = 0;

        for (; i + 8 <= state.markCount; i += 8) {
            __m256i vmarks = _mm256_loadu_si256((__m256i*)&state.marks[i]);
            __m256i vdiffs = _mm256_sub_epi32(vpos, vmarks);
            _mm256_storeu_si256((__m256i*)&allDiffs[totalDiffs], vdiffs);
            totalDiffs += 8;
        }

        for (; i < state.markCount; ++i) {
            allDiffs[totalDiffs++] = pos - state.marks[i];
        }

        BitSet256 checkMask;
        checkMask.reset();

        for (int j = 0; j < totalDiffs; ++j) {
            int d = allDiffs[j];
            if (d >= MAX_LENGTH) return false;
            checkMask.set(d);
        }

        if (state.usedDiffs.hasCollisionAVX2(checkMask)) return false;

        diffCount = totalDiffs;
        for (int j = 0; j < totalDiffs; ++j) {
            tempDiffs[j] = allDiffs[j];
        }
        return true;
    }
#endif

    void updateBestSolution(int length, const ThreadState& state) {
        int current = localBestLength.load(std::memory_order_relaxed);
        while (length < current) {
            if (localBestLength.compare_exchange_weak(current, length)) {
                std::lock_guard<std::mutex> lock(solutionMutex);
                if (length < bestSolution.length) {
                    bestSolution.length = length;
                    bestSolution.order = order;
                    bestSolution.marks.assign(state.marks, state.marks + state.markCount);

                    if (length < *globalBound) {
                        *globalBound = length;
                        broadcastBoundToNeighbors(length, mpiRank, mpiSize);
                    }
                }
                break;
            }
        }
    }
};

// ============================================================================
// Subtree Generation
// ============================================================================

std::vector<Subtree> generateSubtrees(int order, int prefixDepth, int bound) {
    std::vector<Subtree> subtrees;

    std::function<void(Subtree&, int)> generate = [&](Subtree& current, int depth) {
        if (depth == prefixDepth) {
            subtrees.push_back(current);
            return;
        }

        int lastMark = current.marks[current.markCount - 1];
        int maxPos = (depth == 1) ? std::min(bound - 1, bound / 2) : bound - 1;

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            if (pos + (order - depth - 1) >= bound) break;

            bool valid = true;
            std::vector<int> newDiffs;
            for (int i = 0; i < current.markCount; ++i) {
                int diff = pos - current.marks[i];
                if (diff >= MAX_LENGTH || current.testDiff(diff)) {
                    valid = false;
                    break;
                }
                newDiffs.push_back(diff);
            }

            if (valid) {
                Subtree next = current;
                next.marks[next.markCount++] = pos;
                for (int d : newDiffs) next.setDiff(d);
                generate(next, depth + 1);
            }
        }
    };

    Subtree root;
    root.clearDiffs();
    root.marks[0] = 0;
    root.markCount = 1;
    root.bestBound = bound;
    generate(root, 1);

    return subtrees;
}

// ============================================================================
// MPI Datatypes
// ============================================================================

MPI_Datatype createSubtreeType() {
    MPI_Datatype type;
    int bl[] = {MAX_ORDER, 1, MAX_LENGTH/8+1, 1};
    MPI_Aint disp[4] = {
        offsetof(Subtree, marks), offsetof(Subtree, markCount),
        offsetof(Subtree, usedDiffs), offsetof(Subtree, bestBound)
    };
    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_UNSIGNED_CHAR, MPI_INT};
    MPI_Type_create_struct(4, bl, disp, types, &type);
    MPI_Type_commit(&type);
    return type;
}

MPI_Datatype createResultType() {
    MPI_Datatype type;
    int bl[] = {MAX_ORDER, 1, 1, 1, 1, 1};
    MPI_Aint disp[6] = {
        offsetof(Result, marks), offsetof(Result, length),
        offsetof(Result, order), offsetof(Result, nodesExplored),
        offsetof(Result, nodesPruned), offsetof(Result, timeMs)
    };
    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_INT, MPI_UINT64_T, MPI_UINT64_T, MPI_DOUBLE};
    MPI_Type_create_struct(6, bl, disp, types, &type);
    MPI_Type_commit(&type);
    return type;
}

// ============================================================================
// CSV Output
// ============================================================================

void appendCSV(const std::string& filename, int order, int mpiProcs, int ompThreads,
               double timeMs, uint64_t nodes, uint64_t pruned, const GolombRuler& solution) {
    std::ifstream check(filename);
    bool header = !check.good();
    check.close();

    std::ofstream f(filename, std::ios::app);
    if (!f.is_open()) return;

    if (header) {
        f << "version,order,mpi_procs,omp_threads,total_workers,time_ms,nodes,pruned,solution,length\n";
    }

    f << 3 << "," << order << "," << mpiProcs << "," << ompThreads << ","
      << (mpiProcs * ompThreads) << ","
      << std::fixed << std::setprecision(2) << timeMs << ","
      << nodes << "," << pruned << ","
      << "\"" << solution.toString() << "\"," << solution.length << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "Error: MPI doesn't support required thread level.\n";
        MPI_Finalize();
        return 1;
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            std::cout << "Golomb Ruler Solver v3 - Hybrid MPI+OpenMP\n\n";
            std::cout << "Usage: mpirun -np <procs> " << argv[0] << " <order> [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --threads N   OpenMP threads per rank (default: auto)\n";
            std::cout << "  --depth N     Prefix depth for work distribution\n";
            std::cout << "  --csv FILE    Save results to CSV\n";
        }
        MPI_Finalize();
        return 1;
    }

    int order = parseAndValidateOrder(argv[1]);
    if (order < 4) {
        if (rank == 0) std::cerr << "Error: Order must be >= 4\n";
        MPI_Finalize();
        return 1;
    }

    int prefixDepth = (order <= 8) ? 3 : (order <= 10) ? 4 : (order <= 12) ? 5 : 6;
    int numThreads = omp_get_max_threads();
    std::string csvFile;
    bool useSIMD = true;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threads" && i+1 < argc) numThreads = std::atoi(argv[++i]);
        else if (arg == "--depth" && i+1 < argc) prefixDepth = std::atoi(argv[++i]);
        else if (arg == "--csv" && i+1 < argc) csvFile = argv[++i];
        else if (arg == "--no-simd") useSIMD = false;
    }

    omp_set_num_threads(numThreads);

    MPI_Datatype subtreeType = createSubtreeType();
    MPI_Datatype resultType = createResultType();

    double startTime = MPI_Wtime();

    if (rank == 0) {
        // === MASTER ===
        std::cout << "=== Golomb Ruler Solver v3: Hybrid MPI+OpenMP ===\n";
        std::cout << "Order: " << order << "\n";
        std::cout << "MPI Processes: " << size << "\n";
        std::cout << "OpenMP Threads/rank: " << numThreads << "\n";
        std::cout << "Total workers: " << size * numThreads << "\n";
        std::cout << "Prefix depth: " << prefixDepth << "\n";
#ifdef USE_AVX2
        std::cout << "SIMD/AVX2: " << (useSIMD ? "Enabled" : "Disabled") << "\n";
#endif

        int bound = computeGreedyBound(order);
        std::cout << "Initial bound: " << bound << "\n";

        std::vector<Subtree> subtrees = generateSubtrees(order, prefixDepth, bound);
        std::cout << "Subtrees: " << subtrees.size() << "\n\n";

        size_t nextSubtree = 0;
        int activeWorkers = 0;

        // Initial distribution
        for (int w = 1; w < size && nextSubtree < subtrees.size(); ++w) {
            subtrees[nextSubtree].bestBound = bound;
            MPI_Send(&subtrees[nextSubtree], 1, subtreeType, w, TAG_WORK, MPI_COMM_WORLD);
            nextSubtree++;
            activeWorkers++;
        }

        GolombRuler globalBest;
        globalBest.length = bound;
        uint64_t totalNodes = 0, totalPruned = 0;
        int localBound = bound;

        while (activeWorkers > 0 || nextSubtree < subtrees.size()) {
            // Check for bound updates
            int flag;
            MPI_Status status;
            while (true) {
                MPI_Iprobe(MPI_ANY_SOURCE, TAG_BOUND, MPI_COMM_WORLD, &flag, &status);
                if (!flag) break;
                int newBound;
                MPI_Recv(&newBound, 1, MPI_INT, status.MPI_SOURCE, TAG_BOUND,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (newBound < localBound) localBound = newBound;
            }

            // Check for results
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
                Result result;
                MPI_Recv(&result, 1, resultType, status.MPI_SOURCE, TAG_RESULT,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                activeWorkers--;
                totalNodes += result.nodesExplored;
                totalPruned += result.nodesPruned;

                if (result.length < globalBest.length) {
                    globalBest.length = result.length;
                    globalBest.order = result.order;
                    globalBest.marks.assign(result.marks, result.marks + result.order);
                    localBound = std::min(localBound, result.length);
                }

                if (nextSubtree < subtrees.size()) {
                    subtrees[nextSubtree].bestBound = localBound;
                    MPI_Send(&subtrees[nextSubtree], 1, subtreeType,
                             status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
                    nextSubtree++;
                    activeWorkers++;
                }
            }
        }

        // Terminate workers
        Subtree dummy;
        for (int w = 1; w < size; ++w) {
            MPI_Send(&dummy, 1, subtreeType, w, TAG_DONE, MPI_COMM_WORLD);
        }

        double endTime = MPI_Wtime();
        double totalTime = (endTime - startTime) * 1000.0;

        std::cout << "=== Golomb G" << order << " ===\n";
        std::cout << "Solution: " << globalBest.toString() << "\n";
        std::cout << "Length: " << globalBest.length << "\n";
        std::cout << "Time: " << std::fixed << std::setprecision(2) << totalTime << " ms\n";
        std::cout << "Nodes explored: " << totalNodes << "\n";
        std::cout << "Nodes pruned: " << totalPruned << "\n";

        int optimalLength = getOptimalLength(order);
        if (optimalLength > 0 && globalBest.length == optimalLength) {
            std::cout << "*** OPTIMAL SOLUTION FOUND ***\n";
        }

        if (!csvFile.empty()) {
            appendCSV(csvFile, order, size, numThreads, totalTime, totalNodes, totalPruned, globalBest);
        }

    } else {
        // === WORKER ===
        int localBound = INT_MAX;

        while (true) {
            MPI_Status status;
            Subtree subtree;
            MPI_Recv(&subtree, 1, subtreeType, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_DONE) break;

            localBound = std::min(localBound, subtree.bestBound);

            HybridSolver solver(order, subtree, &localBound, rank, size, useSIMD);
            solver.solve(subtree);

            Result result = solver.getResult();
            result.timeMs = 0;
            MPI_Send(&result, 1, resultType, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }

    MPI_Type_free(&subtreeType);
    MPI_Type_free(&resultType);
    MPI_Finalize();

    return 0;
}
