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
 *  Tags used to identify different types of MPI messages in master/worker communication.
 *  @{
 */
const int TAG_WORK = 1;    ///< Master sends work (Subtree) to worker
const int TAG_RESULT = 2;  ///< Worker sends result back to master
const int TAG_DONE = 3;    ///< Master signals worker to terminate
const int TAG_BOUND = 4;   ///< Bound update broadcast via hypercube topology
/** @} */

/// Interval (in nodes) between MPI bound update checks
const int BOUND_CHECK_INTERVAL = 50000;

// ============================================================================
// MPI Tracer (for timeline visualization)
// ============================================================================

/**
 * @struct MPIEvent
 * @brief Represents a single MPI event for timeline visualization.
 *
 * Used by MPITracer to record communication and computation events
 * that can be exported to CSV for visualization tools.
 */
struct MPIEvent {
    int rank;                ///< MPI rank that generated the event
    double start_ms;         ///< Event start time in milliseconds
    double end_ms;           ///< Event end time in milliseconds
    std::string event_type;  ///< Type: "compute", "send", "recv", "idle"
    std::string tag;         ///< MPI tag name: "TAG_WORK", "TAG_RESULT", etc.
    int partner_rank;        ///< Partner rank for send/recv events (-1 if N/A)
};

/**
 * @class MPITracer
 * @brief Records MPI events for timeline visualization and performance analysis.
 *
 * Collects timing information for compute, send, receive, and idle phases.
 * Events can be gathered across all ranks and written to CSV for visualization.
 *
 * @note Must be enabled explicitly via enable() before recording events
 */
class MPITracer {
public:
    /**
     * @brief Constructs a new MPI Tracer.
     * @param rank     MPI rank of this process
     * @param baseTime Reference time (typically MPI_Wtime() at start)
     */
    MPITracer(int rank, double baseTime) : rank_(rank), baseTime_(baseTime), enabled_(false) {}

    /// Enables event recording
    void enable() { enabled_ = true; }
    /// Returns whether tracing is enabled
    bool isEnabled() const { return enabled_; }

    /// Marks the start of a compute phase
    void startCompute() {
        if (!enabled_) return;
        computeStart_ = MPI_Wtime();
    }

    /// Marks the end of a compute phase and records the event
    void endCompute() {
        if (!enabled_) return;
        double now = MPI_Wtime();
        events_.push_back({rank_, toMs(computeStart_), toMs(now), "compute", "", -1});
    }

    /// Records a send event
    void recordSend(int dest, int tag, double start, double end) {
        if (!enabled_) return;
        events_.push_back({rank_, toMs(start), toMs(end), "send", tagName(tag), dest});
    }

    /// Records a receive event
    void recordRecv(int src, int tag, double start, double end) {
        if (!enabled_) return;
        events_.push_back({rank_, toMs(start), toMs(end), "recv", tagName(tag), src});
    }

    /// Records an idle period
    void recordIdle(double start, double end) {
        if (!enabled_) return;
        events_.push_back({rank_, toMs(start), toMs(end), "idle", "", -1});
    }

    /**
     * @brief Gathers events from all ranks and writes to CSV.
     * @param filename Output CSV file path
     * @param comm     MPI communicator
     * @note Only rank 0 writes the file
     */
    void writeCSV(const std::string& filename, MPI_Comm comm) {
        if (!enabled_) return;

        // Gather all events to rank 0
        int mySize = events_.size();
        std::vector<int> allSizes;
        int size;
        MPI_Comm_size(comm, &size);

        if (rank_ == 0) {
            allSizes.resize(size);
        }
        MPI_Gather(&mySize, 1, MPI_INT, allSizes.data(), 1, MPI_INT, 0, comm);

        // Serialize local events
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

        // Gather sizes
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

        // Write CSV on rank 0
        if (rank_ == 0) {
            std::ofstream f(filename);
            f << "rank,start_ms,end_ms,event_type,tag,partner_rank\n";
            f.write(globalBuf.data(), globalBuf.size());
        }
    }

private:
    int rank_;                       ///< MPI rank of this process
    double baseTime_;                ///< Reference time for relative timestamps
    double computeStart_ = 0;        ///< Start time of current compute phase
    bool enabled_;                   ///< Whether tracing is active
    std::vector<MPIEvent> events_;   ///< Collected events

    /// Converts absolute time to milliseconds relative to baseTime
    double toMs(double t) const { return (t - baseTime_) * 1000.0; }

    /// Converts MPI tag to string name
    static std::string tagName(int tag) {
        switch (tag) {
            case TAG_WORK: return "TAG_WORK";
            case TAG_RESULT: return "TAG_RESULT";
            case TAG_DONE: return "TAG_DONE";
            case TAG_BOUND: return "TAG_BOUND";
            default: return "";
        }
    }
};

// ============================================================================
// Subtree (for MPI distribution)
// ============================================================================

/**
 * @struct Subtree
 * @brief Represents a partial search tree for MPI work distribution.
 *
 * Contains the prefix marks and used differences needed to resume
 * search from a specific point in the tree. Sent from master to workers.
 *
 * @note Uses compact bit-packed representation for usedDiffs to minimize
 *       MPI message size.
 */
struct Subtree {
    int marks[MAX_ORDER];                    ///< Prefix mark positions
    int markCount;                           ///< Number of marks in prefix
    unsigned char usedDiffs[MAX_LENGTH / 8 + 1]; ///< Bit-packed used differences
    int bestBound;                           ///< Current best bound at distribution time

    /// Sets a difference as used
    void setDiff(int d) {
        if (d >= 0 && d < MAX_LENGTH) usedDiffs[d / 8] |= (1 << (d % 8));
    }
    /// Tests if a difference is used
    bool testDiff(int d) const {
        if (d < 0 || d >= MAX_LENGTH) return false;
        return usedDiffs[d / 8] & (1 << (d % 8));
    }
    /// Clears all differences
    void clearDiffs() {
        std::memset(usedDiffs, 0, sizeof(usedDiffs));
    }
};

// ============================================================================
// Result (for MPI communication)
// ============================================================================

/**
 * @struct Result
 * @brief Contains the result of a worker's subtree exploration.
 *
 * Sent from workers back to master after completing a subtree.
 * Includes the best solution found and performance statistics.
 */
struct Result {
    int marks[MAX_ORDER];    ///< Best solution marks found
    int length;              ///< Length of best solution
    int order;               ///< Order of the ruler
    uint64_t nodesExplored;  ///< Number of nodes explored
    uint64_t nodesPruned;    ///< Number of nodes pruned
    double timeMs;           ///< Time spent (currently unused)
};

// ============================================================================
// Thread State (cache-aligned)
// ============================================================================

/**
 * @struct ThreadState
 * @brief Thread-local search state for OpenMP threads within an MPI rank.
 *
 * Cache-aligned (64 bytes) to prevent false sharing between threads.
 */
struct alignas(64) ThreadState {
    int marks[MAX_ORDER];          ///< Current mark positions
    BitSet256 usedDiffs;           ///< Used differences bitset
    int markCount;                 ///< Number of marks placed
    uint64_t localNodesExplored;   ///< Thread-local explored counter
    uint64_t localNodesPruned;     ///< Thread-local pruned counter
};

// ============================================================================
// Hypercube Utilities
// ============================================================================

/**
 * @brief Computes the dimension of a hypercube topology.
 *
 * @param size Number of MPI processes
 * @return Dimension d such that 2^d >= size
 *
 * @note For non-power-of-2 sizes, returns floor(log2(size))
 */
int hypercubeDimension(int size) {
    int d = 0, s = size;
    while (s > 1) { s /= 2; d++; }
    return d;
}

/**
 * @brief Returns the hypercube neighbors of a given rank.
 *
 * In a d-dimensional hypercube, rank r has neighbors at r XOR 2^i for i in [0,d).
 *
 * @param rank MPI rank to find neighbors for
 * @param size Total number of MPI processes
 * @return Vector of neighbor ranks
 *
 * @complexity O(log size)
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

/**
 * @brief Broadcasts a new bound to hypercube neighbors.
 *
 * Uses standard MPI_Send for bound propagation with O(log P) latency.
 * MPI_Send is buffered for small messages, avoiding both:
 * - The UB of fire-and-forget MPI_Isend+Request_free
 * - The deadlock risk of MPI_Ssend in peer-to-peer patterns
 *
 * @param bound New bound value to broadcast
 * @param rank  Current MPI rank
 * @param size  Total number of MPI processes
 */
void broadcastBoundToNeighbors(int bound, int rank, int size) {
    for (int neighbor : getHypercubeNeighbors(rank, size)) {
        MPI_Send(&bound, 1, MPI_INT, neighbor, TAG_BOUND, MPI_COMM_WORLD);
    }
}

/**
 * @brief Checks for incoming bound updates from other ranks.
 *
 * Non-blocking probe for TAG_BOUND messages. If a better bound is found,
 * updates currentBound and returns true.
 *
 * @param[in,out] currentBound Current best bound, updated if better found
 * @return true if bound was updated, false otherwise
 */
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

/**
 * @class HybridSolver
 * @brief Hybrid MPI+OpenMP solver that runs on each MPI worker rank.
 *
 * Explores a subtree using OpenMP task parallelism while coordinating
 * bounds with other MPI ranks via hypercube topology.
 *
 * Features:
 * - OpenMP tasks for intra-node parallelism
 * - Bound caching to reduce atomic contention
 * - Periodic MPI bound synchronization
 * - AVX2 SIMD for difference checking
 *
 * @see Subtree, Result
 */
class HybridSolver {
private:
    int order;                              ///< Target ruler order
    std::atomic<int> localBestLength;       ///< Best length found by this rank
    GolombRuler bestSolution;               ///< Best solution found
    std::mutex solutionMutex;               ///< Mutex for updating solution

    std::atomic<uint64_t> totalNodesExplored; ///< Total nodes explored
    std::atomic<uint64_t> totalNodesPruned;   ///< Total nodes pruned

    int* globalBound;   ///< Pointer to shared MPI bound (updated by hypercube)
    int mpiRank;        ///< MPI rank of this process
    int mpiSize;        ///< Total MPI processes
    bool useSIMD;       ///< Whether to use AVX2 SIMD

public:
    /**
     * @brief Constructs a HybridSolver for a specific subtree.
     *
     * @param n       Order of the Golomb ruler
     * @param subtree Subtree to explore (provides initial bound)
     * @param bound   Pointer to global MPI bound
     * @param rank    MPI rank of this process
     * @param size    Total MPI processes
     * @param simd    Enable AVX2 SIMD (default: true)
     */
    HybridSolver(int n, const Subtree& subtree, int* bound, int rank, int size, bool simd = true)
        : order(n), localBestLength(subtree.bestBound),
          totalNodesExplored(0), totalNodesPruned(0),
          globalBound(bound), mpiRank(rank), mpiSize(size), useSIMD(simd) {
        bestSolution.length = subtree.bestBound;
    }

    /**
     * @brief Explores the subtree using OpenMP parallel tasks.
     *
     * Generates tasks for each valid next position and explores them
     * in parallel using the thread pool.
     *
     * @param subtree The subtree to explore
     */
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

    /**
     * @brief Returns the result of the subtree exploration.
     *
     * @return Result containing best solution and statistics
     */
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
    /**
     * @brief Initializes thread state from a subtree.
     *
     * Copies marks and differences from the MPI subtree to the local thread state.
     *
     * @param[out] state   Thread state to initialize
     * @param[in]  subtree Subtree to copy from
     */
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

    /**
     * @brief Recursive branch-and-bound search with MPI bound coordination.
     *
     * Explores the search tree, periodically checking for MPI bound updates.
     * Uses bound caching to minimize atomic and MPI overhead.
     *
     * @param[in,out] state Thread-local search state
     * @param[in]     depth Current depth in search tree
     */
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

    /**
     * @brief Checks differences using scalar implementation.
     *
     * @param[in]  state     Current thread state
     * @param[in]  pos       Candidate position
     * @param[out] tempDiffs Array to store new differences
     * @param[out] diffCount Number of differences stored
     * @return true if position is valid, false if collision found
     */
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
    /**
     * @brief Checks differences using AVX2 SIMD vectorization.
     *
     * Processes 8 marks at a time for improved performance.
     *
     * @param[in]  state     Current thread state
     * @param[in]  pos       Candidate position
     * @param[out] tempDiffs Array to store new differences
     * @param[out] diffCount Number of differences stored
     * @return true if position is valid, false if collision found
     */
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

    /**
     * @brief Thread-safe update of the best solution with MPI broadcast.
     *
     * Updates local best and broadcasts to hypercube neighbors if improved.
     * All operations are protected by mutex to ensure consistency.
     *
     * @param[in] length New solution length
     * @param[in] state  Thread state containing the solution
     *
     * @thread_safety Thread-safe via mutex protection
     */
    void updateBestSolution(int length, const ThreadState& state) {
        std::lock_guard<std::mutex> lock(solutionMutex);

        // Check and update atomically under mutex protection
        if (length < localBestLength.load(std::memory_order_acquire)) {
            localBestLength.store(length, std::memory_order_release);
            bestSolution.length = length;
            bestSolution.order = order;
            bestSolution.marks.assign(state.marks, state.marks + state.markCount);

            // Broadcast improved bound to hypercube neighbors
            if (length < *globalBound) {
                *globalBound = length;
                broadcastBoundToNeighbors(length, mpiRank, mpiSize);
            }
        }
    }
};

// ============================================================================
// Subtree Generation
// ============================================================================

/**
 * @brief Generates subtrees for MPI work distribution.
 *
 * Enumerates all valid partial solutions up to prefixDepth and creates
 * Subtree objects that can be distributed to workers.
 *
 * @param order       Order of the Golomb ruler
 * @param prefixDepth Depth to enumerate (number of marks in prefix)
 * @param bound       Initial upper bound for pruning
 *
 * @return Vector of Subtrees for distribution
 *
 * @note Uses symmetry breaking: first mark <= bound/2
 */
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

/**
 * @brief Creates an MPI datatype for Subtree structure.
 *
 * @return Committed MPI_Datatype for sending/receiving Subtrees
 *
 * @note Caller must call MPI_Type_free() when done
 */
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

/**
 * @brief Creates an MPI datatype for Result structure.
 *
 * @return Committed MPI_Datatype for sending/receiving Results
 *
 * @note Caller must call MPI_Type_free() when done
 */
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

/**
 * @brief Appends benchmark results to a CSV file.
 *
 * @param filename   Output CSV file path
 * @param order      Order of the Golomb ruler
 * @param mpiProcs   Number of MPI processes
 * @param ompThreads Number of OpenMP threads per process
 * @param timeMs     Total execution time in milliseconds
 * @param nodes      Total nodes explored
 * @param pruned     Total nodes pruned
 * @param solution   Best solution found
 */
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

/**
 * @brief Main entry point for the hybrid MPI+OpenMP Golomb solver.
 *
 * Implements a master/worker pattern:
 * - Rank 0 (master): Generates subtrees, distributes work, collects results
 * - Ranks 1..N-1 (workers): Receive subtrees, solve using OpenMP, return results
 *
 * Bound propagation uses hypercube topology for O(log P) latency.
 *
 * @param argc Number of command-line arguments
 * @param argv Command-line arguments
 *
 * @return 0 on success, 1 on error
 *
 * Command-line options:
 * - `<order>`: Required. Order of the Golomb ruler
 * - `--threads N`: OpenMP threads per rank
 * - `--depth N`: Prefix depth for work distribution
 * - `--csv FILE`: Output results to CSV
 * - `--trace FILE`: Output MPI trace for visualization
 */
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
            std::cout << "  --trace FILE  Save MPI trace for timeline visualization\n";
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
        if (arg == "--threads" && i+1 < argc) numThreads = safeParseInt(argv[++i], numThreads, "threads");
        else if (arg == "--depth" && i+1 < argc) prefixDepth = safeParseInt(argv[++i], prefixDepth, "depth");
        else if (arg == "--csv" && i+1 < argc) csvFile = argv[++i];
        else if (arg == "--trace" && i+1 < argc) traceFile = argv[++i];
        else if (arg == "--no-simd") useSIMD = false;
    }

    omp_set_num_threads(numThreads);

    MPI_Datatype subtreeType = createSubtreeType();
    MPI_Datatype resultType = createResultType();

    double startTime = MPI_Wtime();

    // Initialize tracer
    MPITracer tracer(rank, startTime);
    if (!traceFile.empty()) {
        tracer.enable();
    }

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
        double idleStart = MPI_Wtime();

        while (true) {
            MPI_Status status;
            Subtree subtree;

            double recvStart = MPI_Wtime();
            MPI_Recv(&subtree, 1, subtreeType, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            double recvEnd = MPI_Wtime();

            // Record idle time before first recv
            if (idleStart > 0 && recvStart > idleStart + 0.0001) {
                tracer.recordIdle(idleStart, recvStart);
            }
            tracer.recordRecv(0, status.MPI_TAG, recvStart, recvEnd);

            if (status.MPI_TAG == TAG_DONE) break;

            localBound = std::min(localBound, subtree.bestBound);

            tracer.startCompute();
            HybridSolver solver(order, subtree, &localBound, rank, size, useSIMD);
            solver.solve(subtree);
            tracer.endCompute();

            Result result = solver.getResult();
            result.timeMs = 0;

            double sendStart = MPI_Wtime();
            MPI_Send(&result, 1, resultType, 0, TAG_RESULT, MPI_COMM_WORLD);
            double sendEnd = MPI_Wtime();
            tracer.recordSend(0, TAG_RESULT, sendStart, sendEnd);

            idleStart = sendEnd;
        }
    }

    // Write trace if enabled
    if (!traceFile.empty()) {
        tracer.writeCSV(traceFile, MPI_COMM_WORLD);
    }

    MPI_Type_free(&subtreeType);
    MPI_Type_free(&resultType);
    MPI_Finalize();

    return 0;
}
