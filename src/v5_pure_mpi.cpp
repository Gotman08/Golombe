/**
 * @file v5_pure_mpi.cpp
 * @brief Pure MPI Golomb Ruler Solver (no OpenMP)
 *
 * Golomb Ruler Solver - High Performance Computing Implementation
 * Copyright (c) 2025 Nicolas Marano
 * Licensed under the MIT License. See LICENSE file for details.
 *
 * Pure MPI implementation based on v4 hypercube architecture:
 * - No OpenMP - each MPI rank is single-threaded
 * - Hypercube topology for O(log P) bound propagation
 * - Static subtree distribution (all ranks generate same list)
 * - Designed for massive parallelism (64, 128, 256+ ranks)
 *
 * Key differences from v4 (hybrid MPI+OpenMP):
 * - No thread management overhead
 * - Simpler memory model (no atomics needed for threads)
 * - Each rank processes subtrees sequentially
 * - Better for fine-grained MPI scaling analysis
 *
 * Best for: Large-scale MPI benchmarks, pure MPI comparison studies
 * Recommended: Many ranks with 1 core each
 *
 * Usage:
 *   mpirun -np 64 ./golomb_v5 12
 *   mpirun -np 128 ./golomb_v5 13
 */

#include <mpi.h>
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
#include <cmath>
#include <sstream>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

// ============================================================================
// MPI Error Checking Macro
// ============================================================================

/**
 * @brief Macro to check MPI return codes and abort on error.
 *
 * Provides detailed error messages including file and line number.
 * Should be used for critical MPI operations (Init, Finalize, Allreduce, etc.)
 *
 * @param call MPI function call to check
 */
#define MPI_CHECK(call) do { \
    int mpi_err = (call); \
    if (mpi_err != MPI_SUCCESS) { \
        char mpi_err_str[MPI_MAX_ERROR_STRING]; \
        int mpi_err_len; \
        MPI_Error_string(mpi_err, mpi_err_str, &mpi_err_len); \
        std::cerr << "MPI Error at " << __FILE__ << ":" << __LINE__ \
                  << ": " << mpi_err_str << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, mpi_err); \
    } \
} while(0)

// ============================================================================
// MPI Tags
// ============================================================================

/** @name MPI Message Tags
 *  Tag used for bound propagation in hypercube topology.
 *  @{
 */
const int TAG_BOUND = 4;  ///< Bound update broadcast via hypercube neighbors
/** @} */

// ============================================================================
// MPI Tracer (for timeline visualization)
// ============================================================================

/**
 * @struct MPIEvent
 * @brief Represents a single MPI event for timeline visualization.
 */
struct MPIEvent {
    int rank;                ///< MPI rank that generated this event
    double start_ms;         ///< Event start time in milliseconds from baseline
    double end_ms;           ///< Event end time in milliseconds from baseline
    std::string event_type;  ///< Event type: "compute", "send", "recv", "idle", "barrier"
    std::string tag;         ///< MPI tag name (e.g., "TAG_BOUND")
    int partner_rank;        ///< Partner rank for send/recv events, -1 otherwise
};

/**
 * @class MPITracer
 * @brief Records MPI events for timeline visualization and performance analysis.
 */
class MPITracer {
public:
    MPITracer(int rank, double baseTime) : rank_(rank), baseTime_(baseTime), enabled_(false) {}

    void enable() { enabled_ = true; }
    bool isEnabled() const { return enabled_; }

    void startCompute() {
        if (!enabled_) return;
        computeStart_ = MPI_Wtime();
    }

    void endCompute() {
        if (!enabled_) return;
        double now = MPI_Wtime();
        if (now > computeStart_ + 0.0001) {
            events_.push_back({rank_, toMs(computeStart_), toMs(now), "compute", "", -1});
        }
    }

    void recordSend(int dest, int tag, double start, double end) {
        if (!enabled_) return;
        events_.push_back({rank_, toMs(start), toMs(end), "send", tagName(tag), dest});
    }

    void recordRecv(int src, int tag, double start, double end) {
        if (!enabled_) return;
        events_.push_back({rank_, toMs(start), toMs(end), "recv", tagName(tag), src});
    }

    void recordBarrier(double start, double end) {
        if (!enabled_) return;
        events_.push_back({rank_, toMs(start), toMs(end), "barrier", "", -1});
    }

    void recordIdle(double start, double end) {
        if (!enabled_) return;
        if (end > start + 0.0001) {
            events_.push_back({rank_, toMs(start), toMs(end), "idle", "", -1});
        }
    }

    void writeCSV(const std::string& filename, MPI_Comm comm) {
        if (!enabled_) return;

        int mySize = events_.size();
        std::vector<int> allSizes;
        int size;
        MPI_Comm_size(comm, &size);

        if (rank_ == 0) {
            allSizes.resize(size);
        }
        MPI_Gather(&mySize, 1, MPI_INT, allSizes.data(), 1, MPI_INT, 0, comm);

        std::vector<char> localBuf;
        for (const auto& e : events_) {
            std::ostringstream oss;
            oss << e.rank << "," << std::fixed << std::setprecision(2) << e.start_ms
                << "," << e.end_ms << "," << e.event_type << "," << e.tag
                << "," << e.partner_rank << "\n";
            std::string line = oss.str();
            localBuf.insert(localBuf.end(), line.begin(), line.end());
        }
        localBuf.push_back('\0');

        int localSize = localBuf.size();
        std::vector<int> recvSizes(size), displs(size);
        MPI_Gather(&localSize, 1, MPI_INT, recvSizes.data(), 1, MPI_INT, 0, comm);

        std::vector<char> globalBuf;
        if (rank_ == 0) {
            int total = 0;
            for (int i = 0; i < size; ++i) {
                displs[i] = total;
                total += recvSizes[i];
            }
            globalBuf.resize(total);
        }

        MPI_Gatherv(localBuf.data(), localSize, MPI_CHAR,
                    globalBuf.data(), recvSizes.data(), displs.data(), MPI_CHAR,
                    0, comm);

        if (rank_ == 0) {
            std::ofstream f(filename);
            f << "rank,start_ms,end_ms,event_type,tag,partner_rank\n";
            f.write(globalBuf.data(), globalBuf.size());
        }
    }

private:
    int rank_;
    double baseTime_;
    double computeStart_ = 0;
    bool enabled_;
    std::vector<MPIEvent> events_;

    double toMs(double t) const { return (t - baseTime_) * 1000.0; }

    static std::string tagName(int tag) {
        switch (tag) {
            case TAG_BOUND: return "TAG_BOUND";
            default: return "";
        }
    }
};

/** Interval between MPI bound checks (in nodes explored) */
const int BOUND_CHECK_INTERVAL = 10000;

// ============================================================================
// Subtree Structure
// ============================================================================

/**
 * @struct Subtree
 * @brief Represents a subtree of the search space for static distribution.
 */
struct Subtree {
    int marks[MAX_ORDER];
    int markCount;
    unsigned char usedDiffs[MAX_LENGTH / 8 + 1];
    int index;

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
// Search State (for sequential exploration)
// ============================================================================

/**
 * @struct SearchState
 * @brief State for sequential branch-and-bound exploration.
 */
struct SearchState {
    int marks[MAX_ORDER];
    BitSet256 usedDiffs;
    int markCount;
    uint64_t nodesExplored;
    uint64_t nodesPruned;
};

// ============================================================================
// Hypercube Utilities
// ============================================================================

/**
 * @brief Computes the dimension of a hypercube for a given number of processes.
 */
int hypercubeDimension(int size) {
    int d = 0, s = size;
    while (s > 1) { s /= 2; d++; }
    return d;
}

/**
 * @brief Tests if a number is a power of two.
 */
bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/**
 * @brief Returns the hypercube neighbors of a given rank.
 */
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
// Hypercube Bound Manager (simplified for single-threaded)
// ============================================================================

/**
 * @class HypercubeBoundManager
 * @brief Manages bound propagation across MPI ranks using hypercube topology.
 *
 * Simplified version without atomics (single-threaded per rank).
 */
class HypercubeBoundManager {
private:
    int rank;
    int size;
    std::vector<int> neighbors;
    int localBound;
    int lastBroadcastBound;

public:
    HypercubeBoundManager(int r, int s, int initialBound)
        : rank(r), size(s), localBound(initialBound), lastBroadcastBound(initialBound) {
        neighbors = getHypercubeNeighbors(rank, size);
    }

    int getBound() const {
        return localBound;
    }

    bool tryUpdateBound(int newBound) {
        if (newBound < localBound) {
            localBound = newBound;
            return true;
        }
        return false;
    }

    void broadcastBoundToNeighbors(int bound) {
        if (bound < lastBroadcastBound) {
            lastBroadcastBound = bound;
            for (int neighbor : neighbors) {
                MPI_Send(&bound, 1, MPI_INT, neighbor, TAG_BOUND, MPI_COMM_WORLD);
            }
        }
    }

    void checkAndForwardBounds() {
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
                        MPI_Send(&newBound, 1, MPI_INT, neighbor, TAG_BOUND, MPI_COMM_WORLD);
                    }
                }
                lastBroadcastBound = std::min(lastBroadcastBound, newBound);
            }
        }
    }
};

// ============================================================================
// Sequential Solver (no OpenMP)
// ============================================================================

/**
 * @class SequentialSolver
 * @brief Branch-and-bound solver for a single MPI rank (no OpenMP).
 *
 * Each MPI rank creates one SequentialSolver that processes its
 * assigned subtrees one at a time, sequentially.
 */
class SequentialSolver {
private:
    int order;
    int* sharedBound;                   // Pointer to local bound (no atomics needed)
    HypercubeBoundManager* boundManager;
    GolombRuler bestSolution;
    uint64_t totalNodesExplored;
    uint64_t totalNodesPruned;
    int mpiRank;
    bool useSIMD;
    int checkCounter;                   // Counter for MPI bound checks

public:
    SequentialSolver(int n, int* bound, HypercubeBoundManager* manager, int rank, bool simd = true)
        : order(n), sharedBound(bound), boundManager(manager),
          totalNodesExplored(0), totalNodesPruned(0),
          mpiRank(rank), useSIMD(simd), checkCounter(0) {
        bestSolution.length = *bound;
    }

    /**
     * @brief Solves all assigned subtrees sequentially.
     */
    void solveSubtrees(const std::vector<Subtree>& subtrees) {
        for (const auto& subtree : subtrees) {
            solveSubtree(subtree);
        }
    }

    uint64_t getNodesExplored() const { return totalNodesExplored; }
    uint64_t getNodesPruned() const { return totalNodesPruned; }
    const GolombRuler& getBestSolution() const { return bestSolution; }

private:
    void solveSubtree(const Subtree& subtree) {
        SearchState state;
        initializeFromSubtree(state, subtree);

        int startDepth = subtree.markCount;

        // Check for MPI bound updates at start of each subtree
        boundManager->checkAndForwardBounds();
        int mgrBound = boundManager->getBound();
        if (mgrBound < *sharedBound) {
            *sharedBound = mgrBound;
        }

        // Explore all valid child positions
        int lastMark = state.marks[state.markCount - 1];
        int currentBest = *sharedBound;
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
                SearchState childState = state;
                childState.marks[childState.markCount++] = pos;
                for (int d : newDiffs) childState.usedDiffs.set(d);

                branchAndBound(childState, startDepth + 1);

                totalNodesExplored += childState.nodesExplored;
                totalNodesPruned += childState.nodesPruned;
            }

            // Update current best after each branch
            currentBest = *sharedBound;
            maxPos = currentBest - 1;
        }
    }

    void initializeFromSubtree(SearchState& state, const Subtree& subtree) {
        state.markCount = subtree.markCount;
        for (int i = 0; i < subtree.markCount; ++i)
            state.marks[i] = subtree.marks[i];

        state.usedDiffs.reset();
        for (int d = 0; d < MAX_LENGTH; ++d)
            if (subtree.testDiff(d)) state.usedDiffs.set(d);

        state.nodesExplored = 0;
        state.nodesPruned = 0;
    }

    void branchAndBound(SearchState& state, int depth) {
        state.nodesExplored++;

        // Periodic MPI bound check
        if (++checkCounter >= BOUND_CHECK_INTERVAL) {
            checkCounter = 0;
            boundManager->checkAndForwardBounds();
            int mgrBound = boundManager->getBound();
            if (mgrBound < *sharedBound) {
                *sharedBound = mgrBound;
            }
        }

        if (depth == order) [[unlikely]] {
            int length = state.marks[state.markCount - 1];
            updateBestSolution(length, state);
            return;
        }

        int lastMark = state.marks[state.markCount - 1];
        int currentBest = *sharedBound;
        int maxPos = currentBest - 1;

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            int remainingMarks = order - depth - 1;
            if (pos + remainingMarks >= currentBest) {
                state.nodesPruned++;
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
                currentBest = *sharedBound;
                maxPos = currentBest - 1;
            }
        }
    }

    inline bool checkDifferencesScalar(SearchState& state, int pos, int* tempDiffs, int& diffCount) {
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
    inline bool checkDifferencesAVX2(SearchState& state, int pos, int* tempDiffs, int& diffCount) {
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

    void updateBestSolution(int length, const SearchState& state) {
        if (length < *sharedBound) {
            *sharedBound = length;
            bestSolution.length = length;
            bestSolution.order = order;
            bestSolution.marks.assign(state.marks, state.marks + state.markCount);

            // Broadcast to hypercube neighbors
            boundManager->tryUpdateBound(length);
            boundManager->broadcastBoundToNeighbors(length);
        }
    }
};

// ============================================================================
// Subtree Generation
// ============================================================================

/**
 * @brief Generates all subtrees up to a specified prefix depth.
 */
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

/**
 * @brief Returns the subtrees assigned to a specific MPI rank.
 */
std::vector<Subtree> getMySubtrees(const std::vector<Subtree>& allSubtrees, int rank, int size) {
    std::vector<Subtree> mySubtrees;

    int totalSubtrees = static_cast<int>(allSubtrees.size());
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

    for (int i = myStart; i < myEnd && i < totalSubtrees; ++i) {
        mySubtrees.push_back(allSubtrees[i]);
    }

    return mySubtrees;
}

// ============================================================================
// CSV Output
// ============================================================================

/**
 * @brief Appends benchmark results to a CSV file.
 */
void appendCSV(const std::string& filename, int order, int mpiProcs,
               double timeMs, uint64_t nodes, uint64_t pruned, const GolombRuler& solution) {
    std::ifstream check(filename);
    bool header = !check.good();
    check.close();

    std::ofstream f(filename, std::ios::app);
    if (!f.is_open()) return;

    if (header) {
        f << "version,order,mpi_procs,omp_threads,total_workers,time_ms,nodes,pruned,solution,length\n";
    }

    // Version 5 = Pure MPI, omp_threads = 1
    f << 5 << "," << order << "," << mpiProcs << ",1,"
      << mpiProcs << ","
      << std::fixed << std::setprecision(2) << timeMs << ","
      << nodes << "," << pruned << ","
      << "\"" << solution.toString() << "\"," << solution.length << "\n";
}

// ============================================================================
// Main
// ============================================================================

/**
 * @brief Main entry point for the pure MPI Golomb solver.
 *
 * Pure MPI implementation based on v4 hypercube architecture:
 * 1. All ranks are equal (no master/worker distinction)
 * 2. Each rank generates the same subtree list deterministically
 * 3. Each rank takes its portion based on rank number
 * 4. Bounds propagate via hypercube topology in O(log P) steps
 * 5. No OpenMP - each rank is single-threaded
 *
 * @param argc Argument count.
 * @param argv Argument values: order [--depth N] [--csv FILE] [--trace FILE] [--no-simd]
 * @return 0 on success, 1 on error.
 */
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            std::cout << "Golomb Ruler Solver v5 - Pure MPI (no OpenMP)\n\n";
            std::cout << "Usage: mpirun -np <procs> " << argv[0] << " <order> [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --depth N     Prefix depth for work distribution\n";
            std::cout << "  --csv FILE    Save results to CSV\n";
            std::cout << "  --trace FILE  Save MPI trace for timeline visualization\n";
            std::cout << "  --no-simd     Disable AVX2 optimizations\n";
            std::cout << "\nNote: Power of 2 process counts recommended for optimal hypercube\n";
            std::cout << "This version uses 1 thread per MPI rank (no OpenMP)\n";
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
    std::string csvFile;
    std::string traceFile;
    bool useSIMD = true;

    // Safe integer parsing helper
    auto safeParseInt = [&rank](const char* str, int defaultVal, const char* name) {
        try {
            int val = std::stoi(str);
            if (val <= 0) {
                if (rank == 0) std::cerr << "Warning: Invalid " << name << " value, using default\n";
                return defaultVal;
            }
            return val;
        } catch (const std::exception&) {
            if (rank == 0) std::cerr << "Warning: Invalid " << name << " value, using default\n";
            return defaultVal;
        }
    };

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--depth" && i+1 < argc) prefixDepth = safeParseInt(argv[++i], prefixDepth, "depth");
        else if (arg == "--csv" && i+1 < argc) csvFile = argv[++i];
        else if (arg == "--trace" && i+1 < argc) traceFile = argv[++i];
        else if (arg == "--no-simd") useSIMD = false;
    }

    // Synchronize start time
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // Initialize tracer
    MPITracer tracer(rank, startTime);
    if (!traceFile.empty()) {
        tracer.enable();
    }

    // ========== ALL RANKS WORK EQUALLY ==========

    // Step 1: Compute initial bound (greedy solution)
    int bound = computeGreedyBound(order);

    // Step 2: Generate all subtrees (deterministic, same on all ranks)
    std::vector<Subtree> allSubtrees = generateAllSubtrees(order, prefixDepth, bound);
    int totalSubtrees = static_cast<int>(allSubtrees.size());

    // Step 3: Each rank takes its portion
    std::vector<Subtree> mySubtrees = getMySubtrees(allSubtrees, rank, size);

    if (rank == 0) {
        std::cout << "=== Golomb Ruler Solver v5: Pure MPI ===\n";
        std::cout << "Order: " << order << "\n";
        std::cout << "MPI Processes: " << size;
        if (!isPowerOfTwo(size)) {
            std::cout << " (Warning: not power of 2, hypercube may be suboptimal)";
        }
        std::cout << "\n";
        std::cout << "OpenMP Threads: 1 (disabled)\n";
        std::cout << "Total workers: " << size << "\n";
        std::cout << "Prefix depth: " << prefixDepth << "\n";
        std::cout << "Hypercube dimension: " << hypercubeDimension(size) << "\n";
#ifdef USE_AVX2
        std::cout << "SIMD/AVX2: " << (useSIMD ? "Enabled" : "Disabled") << "\n";
#endif
        std::cout << "Initial bound: " << bound << "\n";
        std::cout << "Total subtrees: " << totalSubtrees << "\n";
        std::cout << "Subtrees per rank: ~" << (totalSubtrees / size) << "\n\n";
    }

    // Step 4: Initialize local bound and hypercube manager
    int localBound = bound;
    HypercubeBoundManager boundManager(rank, size, bound);

    // Step 5: Solve my portion
    tracer.startCompute();
    SequentialSolver solver(order, &localBound, &boundManager, rank, useSIMD);
    solver.solveSubtrees(mySubtrees);
    tracer.endCompute();

    // Step 6: Drain pending MPI messages and synchronize bounds
    double syncStart = MPI_Wtime();
    for (int iter = 0; iter < 10; ++iter) {
        boundManager.checkAndForwardBounds();

        int mgrBound = boundManager.getBound();
        if (mgrBound < localBound) {
            localBound = mgrBound;
        }

        double barrierStart = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        double barrierEnd = MPI_Wtime();
        tracer.recordBarrier(barrierStart, barrierEnd);
    }
    double syncEnd = MPI_Wtime();
    tracer.recordIdle(syncStart, syncEnd);

    // Step 7: Global bound sync
    int globalFinalBound;
    MPI_Allreduce(&localBound, &globalFinalBound, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    // Step 8: Gather best solution from the rank that has it
    struct {
        int length;
        int rank;
    } localResult, globalResult;

    const auto& localBest = solver.getBestSolution();
    if (localBest.length == globalFinalBound) {
        localResult.length = localBest.length;
        localResult.rank = rank;
    } else {
        localResult.length = INT_MAX;
        localResult.rank = rank;
    }

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
            appendCSV(csvFile, order, size, totalTime, totalNodes, totalPruned, globalBest);
            std::cout << "Results saved to " << csvFile << "\n";
        }
    }

    // Write trace if enabled
    if (!traceFile.empty()) {
        tracer.writeCSV(traceFile, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Trace saved to " << traceFile << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
