/**
 * @file v4_hypercube.cpp
 * @brief Pure Hypercube MPI+OpenMP Golomb Ruler Solver
 *
 * Golomb Ruler Solver - High Performance Computing Implementation
 * Copyright (c) 2025 Nicolas Marano
 * Licensed under the MIT License. See LICENSE file for details.
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
#include <cmath>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

#include <sstream>

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
 *
 * Used by MPITracer to record compute, communication, and synchronization
 * events for post-mortem analysis and visualization.
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
 *
 * This class collects timing information about compute phases, MPI communications,
 * barriers, and idle periods. Events can be exported to CSV format for visualization
 * with timeline plotting tools.
 *
 * @note Tracing is disabled by default. Call enable() to activate.
 * @see MPIEvent
 */
class MPITracer {
public:
    /**
     * @brief Constructs an MPI tracer for a specific rank.
     * @param rank The MPI rank of this process.
     * @param baseTime Reference time (from MPI_Wtime) for relative timestamps.
     */
    MPITracer(int rank, double baseTime) : rank_(rank), baseTime_(baseTime), enabled_(false) {}

    /**
     * @brief Enables event recording.
     */
    void enable() { enabled_ = true; }

    /**
     * @brief Checks if tracing is enabled.
     * @return true if tracing is active, false otherwise.
     */
    bool isEnabled() const { return enabled_; }

    /**
     * @brief Marks the start of a compute phase.
     * @note Call endCompute() when the compute phase ends.
     */
    void startCompute() {
        if (!enabled_) return;
        computeStart_ = MPI_Wtime();
    }

    /**
     * @brief Marks the end of a compute phase and records the event.
     * @note Events shorter than 0.1ms are filtered out.
     */
    void endCompute() {
        if (!enabled_) return;
        double now = MPI_Wtime();
        if (now > computeStart_ + 0.0001) {  // Avoid zero-duration events
            events_.push_back({rank_, toMs(computeStart_), toMs(now), "compute", "", -1});
        }
    }

    /**
     * @brief Records an MPI send event.
     * @param dest Destination rank.
     * @param tag MPI message tag.
     * @param start Start time (MPI_Wtime).
     * @param end End time (MPI_Wtime).
     */
    void recordSend(int dest, int tag, double start, double end) {
        if (!enabled_) return;
        events_.push_back({rank_, toMs(start), toMs(end), "send", tagName(tag), dest});
    }

    /**
     * @brief Records an MPI receive event.
     * @param src Source rank.
     * @param tag MPI message tag.
     * @param start Start time (MPI_Wtime).
     * @param end End time (MPI_Wtime).
     */
    void recordRecv(int src, int tag, double start, double end) {
        if (!enabled_) return;
        events_.push_back({rank_, toMs(start), toMs(end), "recv", tagName(tag), src});
    }

    /**
     * @brief Records an MPI barrier event.
     * @param start Start time (MPI_Wtime).
     * @param end End time (MPI_Wtime).
     */
    void recordBarrier(double start, double end) {
        if (!enabled_) return;
        events_.push_back({rank_, toMs(start), toMs(end), "barrier", "", -1});
    }

    /**
     * @brief Records an idle period.
     * @param start Start time (MPI_Wtime).
     * @param end End time (MPI_Wtime).
     * @note Events shorter than 0.1ms are filtered out.
     */
    void recordIdle(double start, double end) {
        if (!enabled_) return;
        if (end > start + 0.0001) {  // Avoid zero-duration events
            events_.push_back({rank_, toMs(start), toMs(end), "idle", "", -1});
        }
    }

    /**
     * @brief Gathers all events from all ranks and writes to CSV file.
     *
     * This is a collective operation - all ranks must call this method.
     * Only rank 0 writes the output file.
     *
     * @param filename Output CSV filename.
     * @param comm MPI communicator to use for gathering.
     *
     * @note CSV format: rank,start_ms,end_ms,event_type,tag,partner_rank
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
    int rank_;                        ///< MPI rank of this tracer
    double baseTime_;                 ///< Reference time for relative timestamps
    double computeStart_ = 0;         ///< Start time of current compute phase
    bool enabled_;                    ///< Whether tracing is active
    std::vector<MPIEvent> events_;    ///< Collected events

    /**
     * @brief Converts MPI_Wtime to milliseconds relative to baseTime.
     * @param t Time from MPI_Wtime.
     * @return Time in milliseconds.
     */
    double toMs(double t) const { return (t - baseTime_) * 1000.0; }

    /**
     * @brief Converts MPI tag to human-readable name.
     * @param tag MPI tag value.
     * @return String representation of the tag.
     */
    static std::string tagName(int tag) {
        switch (tag) {
            case TAG_BOUND: return "TAG_BOUND";
            default: return "";
        }
    }
};

/**
 * @brief Interval (in nodes) between MPI bound update checks.
 *
 * Controls how often the branch-and-bound algorithm checks for
 * bound updates received from hypercube neighbors via MPI.
 */
const int BOUND_CHECK_INTERVAL = 10000;

// ============================================================================
// Subtree Structure
// ============================================================================

/**
 * @struct Subtree
 * @brief Represents a subtree of the search space for static distribution.
 *
 * In the hypercube architecture, all ranks generate the same set of subtrees
 * deterministically, then each rank takes its portion based on its rank number.
 * This eliminates the need for master-worker communication for work distribution.
 *
 * @note Uses packed bit array for used differences to minimize memory.
 */
struct Subtree {
    int marks[MAX_ORDER];                      ///< Mark positions in this partial ruler
    int markCount;                             ///< Number of marks placed
    unsigned char usedDiffs[MAX_LENGTH / 8 + 1]; ///< Packed bitset of used differences
    int index;                                 ///< Global index for identification and load balancing

    /**
     * @brief Sets a difference as used in the packed bitset.
     * @param d Difference value to mark as used.
     * @pre d >= 0 && d < MAX_LENGTH
     */
    void setDiff(int d) {
        if (d >= 0 && d < MAX_LENGTH) usedDiffs[d / 8] |= (1 << (d % 8));
    }

    /**
     * @brief Tests if a difference is already used.
     * @param d Difference value to test.
     * @return true if the difference is used, false otherwise.
     */
    bool testDiff(int d) const {
        if (d < 0 || d >= MAX_LENGTH) return false;
        return usedDiffs[d / 8] & (1 << (d % 8));
    }

    /**
     * @brief Clears all used differences.
     */
    void clearDiffs() {
        std::memset(usedDiffs, 0, sizeof(usedDiffs));
    }
};

// ============================================================================
// Thread State (cache-aligned)
// ============================================================================

/**
 * @struct ThreadState
 * @brief Thread-local state for branch-and-bound exploration.
 *
 * Each OpenMP thread maintains its own ThreadState to avoid false sharing
 * and synchronization overhead. The struct is cache-aligned to 64 bytes
 * to ensure each thread's state occupies separate cache lines.
 *
 * @note Uses BitSet256 for O(1) difference collision detection with AVX2.
 */
struct alignas(64) ThreadState {
    int marks[MAX_ORDER];         ///< Current mark positions in the partial ruler
    BitSet256 usedDiffs;          ///< Bitset of differences already used
    int markCount;                ///< Number of marks currently placed
    uint64_t localNodesExplored;  ///< Counter for nodes explored by this thread
    uint64_t localNodesPruned;    ///< Counter for nodes pruned by this thread
};

// ============================================================================
// Hypercube Utilities
// ============================================================================

/**
 * @brief Computes the dimension of a hypercube for a given number of processes.
 *
 * For P processes, the hypercube dimension is ceil(log2(P)).
 * This determines the number of neighbors each rank has.
 *
 * @param size Number of MPI processes.
 * @return Dimension of the hypercube (log2 of size).
 *
 * @note For optimal performance, size should be a power of 2.
 * @complexity O(log P)
 */
int hypercubeDimension(int size) {
    int d = 0, s = size;
    while (s > 1) { s /= 2; d++; }
    return d;
}

/**
 * @brief Tests if a number is a power of two.
 * @param n Number to test.
 * @return true if n is a power of 2, false otherwise.
 * @complexity O(1)
 */
bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/**
 * @brief Returns the hypercube neighbors of a given rank.
 *
 * In a hypercube topology, rank i's neighbors are ranks that differ
 * by exactly one bit in their binary representation. Each rank has
 * at most log2(P) neighbors.
 *
 * @param rank The MPI rank to find neighbors for.
 * @param size Total number of MPI processes.
 * @return Vector of neighbor rank numbers.
 *
 * @note For non-power-of-2 sizes, some ranks may have fewer neighbors.
 * @complexity O(log P)
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
// Hypercube Bound Manager
// ============================================================================

/**
 * @class HypercubeBoundManager
 * @brief Manages bound propagation across MPI ranks using hypercube topology.
 *
 * This class implements O(log P) bound propagation by sending updates only to
 * hypercube neighbors. Each neighbor then forwards to its own neighbors,
 * ensuring all ranks receive the update within log2(P) communication steps.
 *
 * Key features:
 * - Atomic local bound for thread-safe access from OpenMP threads
 * - Mutex-protected MPI operations for thread safety
 * - Fire-and-forget (MPI_Isend + MPI_Request_free) for non-blocking sends
 * - Duplicate suppression via lastBroadcastBound tracking
 *
 * @thread_safety MPI calls are protected by mutex. getBound() is lock-free.
 * @see getHypercubeNeighbors, hypercubeDimension
 */
class HypercubeBoundManager {
private:
    int rank;                        ///< This process's MPI rank
    int size;                        ///< Total number of MPI processes
    std::vector<int> neighbors;      ///< Hypercube neighbors (differ by 1 bit)
    std::atomic<int> localBound;     ///< Current best bound (atomic for thread-safety)
    std::mutex mpiMutex;             ///< Protects MPI send/recv operations
    int lastBroadcastBound;          ///< Last bound we broadcast (avoid duplicates)

public:
    /**
     * @brief Constructs a bound manager for a specific rank.
     * @param r This process's MPI rank.
     * @param s Total number of MPI processes.
     * @param initialBound Initial upper bound (typically from greedy solution).
     */
    HypercubeBoundManager(int r, int s, int initialBound)
        : rank(r), size(s), localBound(initialBound), lastBroadcastBound(initialBound) {
        neighbors = getHypercubeNeighbors(rank, size);
    }

    /**
     * @brief Returns the current best bound.
     * @return Current bound value.
     * @thread_safety Lock-free (atomic load with relaxed ordering).
     */
    int getBound() const {
        return localBound.load(std::memory_order_relaxed);
    }

    /**
     * @brief Attempts to update the local bound if newBound is better.
     * @param newBound Candidate new bound.
     * @return true if the bound was updated, false otherwise.
     * @thread_safety Uses compare-exchange for atomicity.
     */
    bool tryUpdateBound(int newBound) {
        int current = localBound.load(std::memory_order_relaxed);
        while (newBound < current) {
            if (localBound.compare_exchange_weak(current, newBound)) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Broadcasts a new bound to all hypercube neighbors.
     *
     * Uses standard MPI_Send for safe bound propagation.
     * Only sends if bound is better than the last broadcast to avoid
     * flooding the network.
     *
     * @param bound The bound to broadcast.
     * @thread_safety Protected by mpiMutex.
     * @complexity O(log P) sends
     */
    void broadcastBoundToNeighbors(int bound) {
        std::lock_guard<std::mutex> lock(mpiMutex);
        if (bound < lastBroadcastBound) {
            lastBroadcastBound = bound;
            for (int neighbor : neighbors) {
                MPI_Send(&bound, 1, MPI_INT, neighbor, TAG_BOUND, MPI_COMM_WORLD);
            }
        }
    }

    /**
     * @brief Checks for incoming bound updates and forwards them.
     *
     * Drains all pending TAG_BOUND messages using MPI_Iprobe/MPI_Recv.
     * For each received bound that improves our local bound, forwards
     * it to other neighbors (except the source) to continue propagation.
     *
     * @thread_safety Protected by mpiMutex.
     * @note Should be called periodically during computation.
     */
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
                        MPI_Send(&newBound, 1, MPI_INT, neighbor, TAG_BOUND, MPI_COMM_WORLD);
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

/**
 * @class LocalSolver
 * @brief Branch-and-bound solver using OpenMP task parallelism within an MPI rank.
 *
 * This class handles the local computation portion of the hypercube solver.
 * Each MPI rank creates one LocalSolver instance that uses OpenMP tasks to
 * explore its assigned subtrees in parallel.
 *
 * Key features:
 * - OpenMP task-based parallelism for subtree exploration
 * - Thread-local bound caching to minimize atomic operations
 * - Periodic MPI bound checking via HypercubeBoundManager
 * - AVX2-accelerated difference checking (when available)
 * - Automatic bound propagation when better solutions are found
 *
 * @thread_safety Thread-safe for concurrent access from OpenMP threads.
 * @see HypercubeBoundManager, ThreadState
 */
class LocalSolver {
private:
    int order;                              ///< Golomb ruler order to solve
    std::atomic<int>* sharedBound;          ///< Shared upper bound (across threads)
    HypercubeBoundManager* boundManager;    ///< Manager for MPI bound propagation
    GolombRuler bestSolution;               ///< Best solution found by this solver
    std::mutex solutionMutex;               ///< Protects bestSolution updates

    std::atomic<uint64_t> totalNodesExplored;  ///< Aggregate node counter
    std::atomic<uint64_t> totalNodesPruned;    ///< Aggregate pruned counter

    int mpiRank;                            ///< This solver's MPI rank (for logging)
    bool useSIMD;                           ///< Whether to use AVX2 optimizations

public:
    /**
     * @brief Constructs a local solver for a specific MPI rank.
     * @param n Golomb ruler order.
     * @param bound Pointer to shared atomic bound.
     * @param manager Pointer to hypercube bound manager.
     * @param rank MPI rank of this solver.
     * @param simd Whether to use AVX2 SIMD optimizations (default: true).
     */
    LocalSolver(int n, std::atomic<int>* bound, HypercubeBoundManager* manager, int rank, bool simd = true)
        : order(n), sharedBound(bound), boundManager(manager),
          totalNodesExplored(0), totalNodesPruned(0),
          mpiRank(rank), useSIMD(simd) {
        bestSolution.length = bound->load(std::memory_order_relaxed);
    }

    /**
     * @brief Solves all assigned subtrees using OpenMP task parallelism.
     *
     * Creates one OpenMP task per subtree. Tasks are executed in parallel
     * by the OpenMP thread pool. Uses firstprivate to copy subtree data
     * into each task.
     *
     * @param subtrees Vector of subtrees assigned to this rank.
     * @thread_safety Safe to call from a single thread; spawns parallel tasks.
     */
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

    /**
     * @brief Returns total nodes explored across all threads.
     * @return Node count.
     */
    uint64_t getNodesExplored() const { return totalNodesExplored.load(); }

    /**
     * @brief Returns total nodes pruned across all threads.
     * @return Pruned count.
     */
    uint64_t getNodesPruned() const { return totalNodesPruned.load(); }

    /**
     * @brief Returns the best solution found by this solver.
     * @return Reference to best GolombRuler solution.
     */
    const GolombRuler& getBestSolution() const { return bestSolution; }

private:
    /**
     * @brief Solves a single subtree with iterative deepening at first level.
     *
     * Entry point for OpenMP tasks. Initializes thread state from subtree,
     * checks for MPI bound updates, then explores all valid child positions.
     * Aggregates statistics after exploration completes.
     *
     * @param subtree The subtree to explore.
     */
    void solveSubtree(const Subtree& subtree) {
        ThreadState state;
        initializeFromSubtree(state, subtree);

        int startDepth = subtree.markCount;

        // Check for MPI bound updates at start of each subtree
        #pragma omp critical(mpi_bound_check)
        {
            boundManager->checkAndForwardBounds();
            int mgrBound = boundManager->getBound();
            int current = sharedBound->load(std::memory_order_relaxed);
            while (mgrBound < current) {
                if (sharedBound->compare_exchange_weak(current, mgrBound)) break;
            }
        }

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

    /**
     * @brief Initializes thread state from a packed Subtree structure.
     *
     * Copies mark positions and unpacks the difference bitset from
     * Subtree's packed format into BitSet256.
     *
     * @param[out] state Thread state to initialize.
     * @param[in] subtree Source subtree data.
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
     * @brief Recursive branch-and-bound exploration.
     *
     * Core algorithm that recursively explores the search tree. Uses:
     * - Thread-local bound caching for performance
     * - Periodic MPI bound updates (every 8192 nodes)
     * - Lower bound pruning (pos + remaining >= best)
     * - AVX2 or scalar difference checking
     *
     * @param state Current thread state (marks, differences).
     * @param depth Current depth in the search tree.
     *
     * @complexity Exponential in (order - depth), with pruning.
     */
    void branchAndBound(ThreadState& state, int depth) {
        state.localNodesExplored++;

        // Bound caching with periodic MPI check
        static thread_local int cachedBound = INT_MAX;
        static thread_local int checkCounter = 0;
        static thread_local int mpiCheckCounter = 0;

        if (++checkCounter >= 4096) {
            cachedBound = sharedBound->load(std::memory_order_relaxed);
            checkCounter = 0;

            // Periodic MPI bound check
            if (++mpiCheckCounter >= 2) {
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

    /**
     * @brief Checks if a new mark creates valid differences (scalar version).
     *
     * Iterates through all existing marks and computes differences.
     * Returns false immediately if any difference is already used.
     *
     * @param state Current thread state.
     * @param pos Candidate mark position.
     * @param[out] tempDiffs Array to store new differences.
     * @param[out] diffCount Number of differences computed.
     * @return true if all differences are valid, false otherwise.
     *
     * @complexity O(markCount)
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
     * @brief Checks if a new mark creates valid differences (AVX2 version).
     *
     * Uses AVX2 SIMD to compute 8 differences in parallel, then checks
     * for collisions using BitSet256::hasCollisionAVX2.
     *
     * @param state Current thread state.
     * @param pos Candidate mark position.
     * @param[out] tempDiffs Array to store new differences.
     * @param[out] diffCount Number of differences computed.
     * @return true if all differences are valid, false otherwise.
     *
     * @note Only used when markCount >= 4 for efficiency.
     * @complexity O(markCount / 8) for SIMD part + O(1) collision check
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
     * @brief Updates the best solution if a better one is found.
     *
     * Uses atomic compare-exchange for thread-safe bound update,
     * then mutex-protected solution copy. Automatically broadcasts
     * the new bound to hypercube neighbors.
     *
     * @param length New solution length.
     * @param state Thread state containing the solution marks.
     *
     * @thread_safety Uses CAS + mutex for thread safety.
     */
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

/**
 * @brief Generates all subtrees up to a specified prefix depth.
 *
 * This function is called identically by all MPI ranks to generate the
 * same deterministic list of subtrees. This eliminates the need for
 * master-worker communication - each rank simply takes its portion.
 *
 * Key features:
 * - Symmetry breaking: first mark limited to bound/2 at depth 1
 * - Lower bound pruning during generation
 * - Assigns global indices for load balancing analysis
 *
 * @param order Golomb ruler order.
 * @param prefixDepth Depth at which to stop and create subtrees.
 * @param bound Initial upper bound for pruning.
 * @return Vector of all valid subtrees at the specified depth.
 *
 * @note All ranks must call this with identical parameters.
 * @complexity O(number of valid partial rulers at prefixDepth)
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

// ============================================================================
// Get my portion of subtrees (static distribution)
// ============================================================================

/**
 * @brief Returns the subtrees assigned to a specific MPI rank.
 *
 * Implements static block distribution with remainder handling.
 * The first 'remainder' ranks get one extra subtree each to balance load.
 *
 * Distribution formula:
 * - Ranks 0 to (remainder-1): get (N/P + 1) subtrees
 * - Ranks remainder to (P-1): get (N/P) subtrees
 *
 * @param allSubtrees Complete list of subtrees (from generateAllSubtrees).
 * @param rank This process's MPI rank.
 * @param size Total number of MPI processes.
 * @return Vector of subtrees assigned to this rank.
 *
 * @complexity O(subtrees per rank)
 */
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

/**
 * @brief Appends benchmark results to a CSV file.
 *
 * Creates the file with headers if it doesn't exist, then appends
 * a new row with the benchmark results. Uses version=4 to identify
 * this as the hypercube implementation.
 *
 * @param filename Output CSV file path.
 * @param order Golomb ruler order.
 * @param mpiProcs Number of MPI processes used.
 * @param ompThreads Number of OpenMP threads per process.
 * @param timeMs Total execution time in milliseconds.
 * @param nodes Total nodes explored.
 * @param pruned Total nodes pruned.
 * @param solution Best solution found.
 *
 * @note Only rank 0 should call this function.
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

    f << 4 << "," << order << "," << mpiProcs << "," << ompThreads << ","
      << (mpiProcs * ompThreads) << ","
      << std::fixed << std::setprecision(2) << timeMs << ","
      << nodes << "," << pruned << ","
      << "\"" << solution.toString() << "\"," << solution.length << "\n";
}

// ============================================================================
// Main
// ============================================================================

/**
 * @brief Main entry point for the hypercube MPI+OpenMP Golomb solver.
 *
 * Implements a decentralized pure hypercube architecture where:
 * 1. All ranks are equal (no master/worker distinction)
 * 2. Each rank generates the same subtree list deterministically
 * 3. Each rank takes its portion based on rank number
 * 4. Bounds propagate via hypercube topology in O(log P) steps
 * 5. Final solution gathered via MPI_Allreduce + MPI_Bcast
 *
 * This design eliminates the master bottleneck and provides better
 * scalability for large cluster deployments (16, 32, 64+ nodes).
 *
 * @param argc Argument count.
 * @param argv Argument values: order [--threads N] [--depth N] [--csv FILE] [--trace FILE] [--no-simd]
 * @return 0 on success, 1 on error.
 *
 * @note Requires MPI_THREAD_FUNNELED or higher for thread safety.
 * @note Power of 2 process counts are recommended for optimal hypercube.
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
            std::cout << "Golomb Ruler Solver v4 - Pure Hypercube MPI+OpenMP\n\n";
            std::cout << "Usage: mpirun -np <procs> " << argv[0] << " <order> [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --threads N   OpenMP threads per rank (default: auto)\n";
            std::cout << "  --depth N     Prefix depth for work distribution\n";
            std::cout << "  --csv FILE    Save results to CSV\n";
            std::cout << "  --trace FILE  Save MPI trace for timeline visualization\n";
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
    tracer.startCompute();
    LocalSolver solver(order, &sharedBound, &boundManager, rank, useSIMD);
    solver.solveSubtrees(mySubtrees);
    tracer.endCompute();

    // Step 6: Drain pending MPI messages and synchronize bounds
    // CRITICAL: Ranks that finish early must receive late bound updates
    double syncStart = MPI_Wtime();
    for (int iter = 0; iter < 10; ++iter) {
        boundManager.checkAndForwardBounds();

        // Update local bound from manager
        int mgrBound = boundManager.getBound();
        int current = sharedBound.load(std::memory_order_relaxed);
        while (mgrBound < current) {
            if (sharedBound.compare_exchange_weak(current, mgrBound)) break;
        }

        double barrierStart = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        double barrierEnd = MPI_Wtime();
        tracer.recordBarrier(barrierStart, barrierEnd);
    }
    double syncEnd = MPI_Wtime();
    // Record idle time during synchronization phase
    tracer.recordIdle(syncStart, syncEnd);

    // Step 7: Global bound sync - ensures all ranks know the true minimum
    int localFinalBound = sharedBound.load(std::memory_order_relaxed);
    int globalFinalBound;
    MPI_Allreduce(&localFinalBound, &globalFinalBound, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    // Step 8: Gather best solution from the rank that has it
    struct {
        int length;
        int rank;
    } localResult, globalResult;

    // Only report our solution if it matches the global best
    const auto& localBest = solver.getBestSolution();
    if (localBest.length == globalFinalBound) {
        localResult.length = localBest.length;
        localResult.rank = rank;
    } else {
        localResult.length = INT_MAX;  // We don't have the best
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
            appendCSV(csvFile, order, size, numThreads, totalTime, totalNodes, totalPruned, globalBest);
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
