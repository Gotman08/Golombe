/**
 * Golomb Ruler Solver - Version 4: Pure Hypercube MPI + OpenMP
 *
 * Decentralized architecture where all ranks are equal:
 * - No master/worker bottleneck - each rank computes its own subtrees
 * - Hypercube topology for O(log P) bound propagation
 * - OpenMP task parallelism within each rank
 * - MPI_Allreduce for final solution gathering
 *
 * Key differences from v3 (master/worker):
 * - Better scalability: no master bottleneck
 * - Lower communication latency: bounds propagate in O(log P) steps
 * - Static initial distribution + peer-to-peer bound sharing
 *
 * Best for: Large clusters (16, 32, 64+ nodes)
 * Recommended: Power of 2 process counts for optimal hypercube
 *
 * Usage:
 *   mpirun -np 8 ./golomb_v4 12 --threads 8
 *   # 8 equal peers x 8 OpenMP threads = 64 effective workers
 */

#include <mpi.h>
#include <omp.h>
#include "common/golomb.hpp"
#include "common/greedy.hpp"
#include "common/bitset256.hpp"
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
#include <cmath>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

// ============================================================================
// MPI Tags
// ============================================================================

const int TAG_BOUND = 4;
const int BOUND_CHECK_INTERVAL = 10000;  // Check for MPI bound updates

// ============================================================================
// Subtree Structure
// ============================================================================

struct Subtree {
    int marks[MAX_ORDER];
    int markCount;
    unsigned char usedDiffs[MAX_LENGTH / 8 + 1];
    int index;  // Global index for identification

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

bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
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

// ============================================================================
// Hypercube Bound Manager
// ============================================================================

class HypercubeBoundManager {
private:
    int rank;
    int size;
    std::vector<int> neighbors;
    std::atomic<int> localBound;
    std::mutex mpiMutex;
    int lastBroadcastBound;

public:
    HypercubeBoundManager(int r, int s, int initialBound)
        : rank(r), size(s), localBound(initialBound), lastBroadcastBound(initialBound) {
        neighbors = getHypercubeNeighbors(rank, size);
    }

    int getBound() const {
        return localBound.load(std::memory_order_relaxed);
    }

    bool tryUpdateBound(int newBound) {
        int current = localBound.load(std::memory_order_relaxed);
        while (newBound < current) {
            if (localBound.compare_exchange_weak(current, newBound)) {
                return true;
            }
        }
        return false;
    }

    void broadcastBoundToNeighbors(int bound) {
        std::lock_guard<std::mutex> lock(mpiMutex);
        if (bound < lastBroadcastBound) {
            lastBroadcastBound = bound;
            for (int neighbor : neighbors) {
                MPI_Request request;
                MPI_Isend(&bound, 1, MPI_INT, neighbor, TAG_BOUND, MPI_COMM_WORLD, &request);
                MPI_Request_free(&request);
            }
        }
    }

    void checkAndForwardBounds() {
        std::lock_guard<std::mutex> lock(mpiMutex);

        int flag;
        MPI_Status status;

        while (true) {
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_BOUND, MPI_COMM_WORLD, &flag, &status);
            if (!flag) break;

            int newBound;
            MPI_Recv(&newBound, 1, MPI_INT, status.MPI_SOURCE, TAG_BOUND,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (tryUpdateBound(newBound)) {
                // Forward to other neighbors (except source)
                for (int neighbor : neighbors) {
                    if (neighbor != status.MPI_SOURCE && newBound < lastBroadcastBound) {
                        MPI_Request request;
                        MPI_Isend(&newBound, 1, MPI_INT, neighbor, TAG_BOUND, MPI_COMM_WORLD, &request);
                        MPI_Request_free(&request);
                    }
                }
                lastBroadcastBound = std::min(lastBroadcastBound, newBound);
            }
        }
    }
};

// ============================================================================
// Local Solver (OpenMP within each rank)
// ============================================================================

class LocalSolver {
private:
    int order;
    std::atomic<int>* sharedBound;
    HypercubeBoundManager* boundManager;
    GolombRuler bestSolution;
    std::mutex solutionMutex;

    std::atomic<uint64_t> totalNodesExplored;
    std::atomic<uint64_t> totalNodesPruned;

    int mpiRank;
    bool useSIMD;

public:
    LocalSolver(int n, std::atomic<int>* bound, HypercubeBoundManager* manager, int rank, bool simd = true)
        : order(n), sharedBound(bound), boundManager(manager),
          totalNodesExplored(0), totalNodesPruned(0),
          mpiRank(rank), useSIMD(simd) {
        bestSolution.length = bound->load(std::memory_order_relaxed);
    }

    void solveSubtrees(const std::vector<Subtree>& subtrees) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (const auto& subtree : subtrees) {
                    #pragma omp task firstprivate(subtree)
                    {
                        solveSubtree(subtree);
                    }
                }
            }
        }
    }

    uint64_t getNodesExplored() const { return totalNodesExplored.load(); }
    uint64_t getNodesPruned() const { return totalNodesPruned.load(); }
    const GolombRuler& getBestSolution() const { return bestSolution; }

private:
    void solveSubtree(const Subtree& subtree) {
        ThreadState state;
        initializeFromSubtree(state, subtree);

        int startDepth = subtree.markCount;

        // Generate first level tasks for better parallelism
        int lastMark = state.marks[state.markCount - 1];
        int currentBest = sharedBound->load(std::memory_order_relaxed);
        int maxPos = currentBest - 1;

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            int remainingMarks = order - startDepth - 1;
            if (pos + remainingMarks >= currentBest) break;

            bool valid = true;
            std::vector<int> newDiffs;
            for (int i = 0; i < state.markCount && valid; ++i) {
                int diff = pos - state.marks[i];
                if (diff >= MAX_LENGTH || state.usedDiffs.test(diff)) {
                    valid = false;
                } else {
                    newDiffs.push_back(diff);
                }
            }

            if (valid) {
                ThreadState childState = state;
                childState.marks[childState.markCount++] = pos;
                for (int d : newDiffs) childState.usedDiffs.set(d);

                branchAndBound(childState, startDepth + 1);

                totalNodesExplored.fetch_add(childState.localNodesExplored, std::memory_order_relaxed);
                totalNodesPruned.fetch_add(childState.localNodesPruned, std::memory_order_relaxed);
            }
        }
    }

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
        static thread_local int mpiCheckCounter = 0;

        if (++checkCounter >= 16384) {
            cachedBound = sharedBound->load(std::memory_order_relaxed);
            checkCounter = 0;

            // Periodic MPI bound check (less frequent)
            if (++mpiCheckCounter >= 4) {
                mpiCheckCounter = 0;
                #pragma omp critical(mpi_bound_check)
                {
                    boundManager->checkAndForwardBounds();
                    int mgrBound = boundManager->getBound();
                    if (mgrBound < cachedBound) {
                        sharedBound->store(mgrBound, std::memory_order_relaxed);
                        cachedBound = mgrBound;
                    }
                }
            }
        }

        if (depth == order) [[unlikely]] {
            int length = state.marks[state.markCount - 1];
            updateBestSolution(length, state);
            cachedBound = sharedBound->load(std::memory_order_relaxed);
            return;
        }

        int lastMark = state.marks[state.markCount - 1];
        int currentBest = cachedBound;
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

                // Update cached bound after recursive call
                currentBest = cachedBound;
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
        int current = sharedBound->load(std::memory_order_relaxed);
        while (length < current) {
            if (sharedBound->compare_exchange_weak(current, length)) {
                std::lock_guard<std::mutex> lock(solutionMutex);
                if (length < bestSolution.length) {
                    bestSolution.length = length;
                    bestSolution.order = order;
                    bestSolution.marks.assign(state.marks, state.marks + state.markCount);

                    // Broadcast to hypercube neighbors
                    boundManager->tryUpdateBound(length);
                    boundManager->broadcastBoundToNeighbors(length);
                }
                break;
            }
        }
    }
};

// ============================================================================
// Subtree Generation (all ranks generate the same list, each takes its portion)
// ============================================================================

std::vector<Subtree> generateAllSubtrees(int order, int prefixDepth, int bound) {
    std::vector<Subtree> subtrees;
    int globalIndex = 0;

    std::function<void(Subtree&, int)> generate = [&](Subtree& current, int depth) {
        if (depth == prefixDepth) {
            current.index = globalIndex++;
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
    root.index = -1;
    generate(root, 1);

    return subtrees;
}

// ============================================================================
// Get my portion of subtrees (static distribution)
// ============================================================================

std::vector<Subtree> getMySubtrees(const std::vector<Subtree>& allSubtrees, int rank, int size) {
    std::vector<Subtree> mySubtrees;

    int totalSubtrees = static_cast<int>(allSubtrees.size());
    int subtreesPerRank = totalSubtrees / size;
    int remainder = totalSubtrees % size;

    // Distribute extra subtrees to first 'remainder' ranks
    int myStart, myEnd;
    if (rank < remainder) {
        myStart = rank * (subtreesPerRank + 1);
        myEnd = myStart + subtreesPerRank + 1;
    } else {
        myStart = remainder * (subtreesPerRank + 1) + (rank - remainder) * subtreesPerRank;
        myEnd = myStart + subtreesPerRank;
    }

    for (int i = myStart; i < myEnd && i < totalSubtrees; ++i) {
        mySubtrees.push_back(allSubtrees[i]);
    }

    return mySubtrees;
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

    f << 4 << "," << order << "," << mpiProcs << "," << ompThreads << ","
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
            std::cout << "Golomb Ruler Solver v4 - Pure Hypercube MPI+OpenMP\n\n";
            std::cout << "Usage: mpirun -np <procs> " << argv[0] << " <order> [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --threads N   OpenMP threads per rank (default: auto)\n";
            std::cout << "  --depth N     Prefix depth for work distribution\n";
            std::cout << "  --csv FILE    Save results to CSV\n";
            std::cout << "  --no-simd     Disable AVX2 optimizations\n";
            std::cout << "\nNote: Power of 2 process counts recommended for optimal hypercube\n";
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

    // Auto-select prefix depth based on order and process count
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

    // Synchronize start time
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // ========== ALL RANKS WORK EQUALLY ==========

    // Step 1: Compute initial bound (greedy solution)
    int bound = computeGreedyBound(order);

    // Step 2: Generate all subtrees (deterministic, same on all ranks)
    std::vector<Subtree> allSubtrees = generateAllSubtrees(order, prefixDepth, bound);
    int totalSubtrees = static_cast<int>(allSubtrees.size());

    // Step 3: Each rank takes its portion
    std::vector<Subtree> mySubtrees = getMySubtrees(allSubtrees, rank, size);

    if (rank == 0) {
        std::cout << "=== Golomb Ruler Solver v4: Pure Hypercube ===\n";
        std::cout << "Order: " << order << "\n";
        std::cout << "MPI Processes: " << size;
        if (!isPowerOfTwo(size)) {
            std::cout << " (Warning: not power of 2, hypercube may be suboptimal)";
        }
        std::cout << "\n";
        std::cout << "OpenMP Threads/rank: " << numThreads << "\n";
        std::cout << "Total workers: " << size * numThreads << "\n";
        std::cout << "Prefix depth: " << prefixDepth << "\n";
        std::cout << "Hypercube dimension: " << hypercubeDimension(size) << "\n";
#ifdef USE_AVX2
        std::cout << "SIMD/AVX2: " << (useSIMD ? "Enabled" : "Disabled") << "\n";
#endif
        std::cout << "Initial bound: " << bound << "\n";
        std::cout << "Total subtrees: " << totalSubtrees << "\n";
        std::cout << "Subtrees per rank: ~" << (totalSubtrees / size) << "\n\n";
    }

    // Step 4: Initialize shared bound and hypercube manager
    std::atomic<int> sharedBound(bound);
    HypercubeBoundManager boundManager(rank, size, bound);

    // Step 5: Solve my portion
    LocalSolver solver(order, &sharedBound, &boundManager, rank, useSIMD);
    solver.solveSubtrees(mySubtrees);

    // Step 6: Final synchronization - gather best solution from all ranks
    MPI_Barrier(MPI_COMM_WORLD);

    // Use MPI_Allreduce to find the minimum length and which rank has it
    struct {
        int length;
        int rank;
    } localResult, globalResult;

    localResult.length = solver.getBestSolution().length;
    localResult.rank = rank;

    MPI_Allreduce(&localResult, &globalResult, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

    // Gather statistics from all ranks
    uint64_t localNodes = solver.getNodesExplored();
    uint64_t localPruned = solver.getNodesPruned();
    uint64_t totalNodes, totalPruned;

    MPI_Reduce(&localNodes, &totalNodes, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localPruned, &totalPruned, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    // Broadcast the best solution from the winning rank
    GolombRuler globalBest;
    int solutionMarks[MAX_ORDER];

    if (rank == globalResult.rank) {
        const auto& best = solver.getBestSolution();
        globalBest = best;
        for (int i = 0; i < order && i < (int)best.marks.size(); ++i) {
            solutionMarks[i] = best.marks[i];
        }
    }

    MPI_Bcast(solutionMarks, order, MPI_INT, globalResult.rank, MPI_COMM_WORLD);
    MPI_Bcast(&globalResult.length, 1, MPI_INT, globalResult.rank, MPI_COMM_WORLD);

    if (rank != globalResult.rank) {
        globalBest.order = order;
        globalBest.length = globalResult.length;
        globalBest.marks.assign(solutionMarks, solutionMarks + order);
    }

    double endTime = MPI_Wtime();
    double totalTime = (endTime - startTime) * 1000.0;

    // Print results (rank 0 only)
    if (rank == 0) {
        std::cout << "=== Golomb G" << order << " ===\n";
        std::cout << "Solution: " << globalBest.toString() << "\n";
        std::cout << "Length: " << globalBest.length << "\n";
        std::cout << "Time: " << std::fixed << std::setprecision(2) << totalTime << " ms\n";
        std::cout << "Nodes explored: " << totalNodes << "\n";
        std::cout << "Nodes pruned: " << totalPruned << "\n";
        std::cout << "Best found by rank: " << globalResult.rank << "\n";

        int optimalLength = getOptimalLength(order);
        if (optimalLength > 0 && globalBest.length == optimalLength) {
            std::cout << "*** OPTIMAL SOLUTION FOUND ***\n";
        }

        if (!csvFile.empty()) {
            appendCSV(csvFile, order, size, numThreads, totalTime, totalNodes, totalPruned, globalBest);
            std::cout << "Results saved to " << csvFile << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
