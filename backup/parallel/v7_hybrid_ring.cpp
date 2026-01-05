/**
 * Golomb Ruler Solver - Parallel Version 7: Hybrid MPI+OpenMP with Ring Shifter
 *
 * This is the most advanced hybrid version combining:
 * - MPI Ring topology for inter-node bound sharing (from v6_ring_shifter)
 * - OpenMP task parallelism for intra-node parallelism (from v6_hardware)
 * - AVX2 SIMD optimizations for vectorized difference checking
 * - All critical bug fixes from v6_hardware and v4_hybrid
 *
 * Architecture:
 *   MPI Level: Ring topology (MPI_Cart_create) for bound propagation
 *   Node Level: OpenMP tasks for subtree parallelism
 *   SIMD Level: AVX2 for 256-bit vectorized operations
 *
 * Bug fixes included:
 * - Atomic memory ordering (relaxed -> release for initialization)
 * - Compare-and-swap for bound updates (prevents race condition)
 * - Correct ThreadState padding (192 bytes, 3 cache lines)
 * - Negative difference bounds checking
 * - Master-only MPI calls (MPI_THREAD_FUNNELED compatible)
 *
 * Usage:
 *   OMP_NUM_THREADS=8 mpirun -np 4 ./golomb_mpi_v7 12 --threads 8 --depth 4
 */

#include <mpi.h>
#include <omp.h>
#include "golomb.hpp"
#include "greedy.hpp"
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

// =============================================================================
// Constants and MPI Tags
// =============================================================================

// MPI tags for work distribution (MPI_COMM_WORLD)
const int TAG_WORK = 1;
const int TAG_RESULT = 2;
const int TAG_DONE = 3;

// MPI tag for ring bound passing (ring_comm)
const int TAG_RING_BOUND = 6;

// How often to check for new bounds (every N nodes)
const int CHECK_INTERVAL = 100000;

// =============================================================================
// Cache-aligned BitSet256 (with negative bounds fix)
// =============================================================================

struct alignas(32) BitSet256 {
    uint64_t words[4];  // 256 bits = 4 x 64-bit words

    inline void reset() {
        words[0] = words[1] = words[2] = words[3] = 0;
    }

    inline bool test(int bit) const {
        // FIX: Check bit < 0 (was missing in v6_hardware)
        if (bit < 0 || bit >= 256) return true;  // Out of bounds = invalid
        return (words[bit >> 6] >> (bit & 63)) & 1;
    }

    inline void set(int bit) {
        if (bit >= 0 && bit < 256)
            words[bit >> 6] |= (1ULL << (bit & 63));
    }

    inline void clear(int bit) {
        if (bit >= 0 && bit < 256)
            words[bit >> 6] &= ~(1ULL << (bit & 63));
    }

    inline void copyFrom(const BitSet256& other) {
        words[0] = other.words[0];
        words[1] = other.words[1];
        words[2] = other.words[2];
        words[3] = other.words[3];
    }

#ifdef USE_AVX2
    inline bool hasCollisionAVX2(const BitSet256& mask) const {
        __m256i used = _mm256_load_si256((__m256i*)words);
        __m256i check = _mm256_load_si256((__m256i*)mask.words);
        __m256i collision = _mm256_and_si256(used, check);
        return !_mm256_testz_si256(collision, collision);
    }
#endif
};

// =============================================================================
// ThreadState (with corrected padding - 192 bytes = 3 cache lines)
// =============================================================================

struct alignas(64) ThreadState {
    int marks[MAX_ORDER];           // 80 bytes (int[20])
    int markCount;                  // 4 bytes
    uint64_t localNodesExplored;    // 8 bytes (was int - FIX)
    uint64_t localNodesPruned;      // 8 bytes (was int - FIX)
    BitSet256 usedDiffs;            // 32 bytes (alignas(32))
    // Compiler handles padding to ensure 64-byte alignment
};

// Note: Actual size may vary due to alignment requirements
// BitSet256 has alignas(32) which may add padding

// =============================================================================
// Subtree Structure (for MPI distribution)
// =============================================================================

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

// Result structure
struct Result {
    int marks[MAX_ORDER];
    int length;
    int order;
    uint64_t nodesExplored;
    double timeMs;
};

// =============================================================================
// RingCommunicator - MPI Cartesian Ring Topology
// =============================================================================

class RingCommunicator {
private:
    MPI_Comm ring_comm;
    int left_neighbor;
    int right_neighbor;
    int ring_rank;
    int ring_size;

public:
    RingCommunicator() : ring_comm(MPI_COMM_NULL), left_neighbor(-1),
                          right_neighbor(-1), ring_rank(-1), ring_size(0) {}

    void initialize(int world_size) {
        int dims[1] = {world_size};
        int periods[1] = {1};  // Wrap-around enabled (ring)
        int reorder = 0;

        MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &ring_comm);
        MPI_Cart_shift(ring_comm, 0, 1, &left_neighbor, &right_neighbor);
        MPI_Comm_rank(ring_comm, &ring_rank);
        MPI_Comm_size(ring_comm, &ring_size);
    }

    void cleanup() {
        if (ring_comm != MPI_COMM_NULL) {
            MPI_Comm_free(&ring_comm);
            ring_comm = MPI_COMM_NULL;
        }
    }

    MPI_Comm getComm() const { return ring_comm; }
    int getLeft() const { return left_neighbor; }
    int getRight() const { return right_neighbor; }
    int getRank() const { return ring_rank; }
    int getSize() const { return ring_size; }
};

// =============================================================================
// RingBoundManager - Thread-Safe Non-Blocking Bound Sharing
// =============================================================================

class RingBoundManager {
private:
    RingCommunicator& ring;
    std::atomic<int> currentBound;  // Thread-safe bound (FIX: was raw int)
    uint64_t messagesReceived;
    uint64_t messagesSent;

    int sendBuffer;
    MPI_Request sendRequest;
    std::mutex mpiMutex;  // Protects MPI calls (FIX: thread safety)

public:
    RingBoundManager(RingCommunicator& r, int initialBound)
        : ring(r), currentBound(initialBound),
          messagesReceived(0), messagesSent(0),
          sendBuffer(0), sendRequest(MPI_REQUEST_NULL) {}

    ~RingBoundManager() {
        std::lock_guard<std::mutex> lock(mpiMutex);
        if (sendRequest != MPI_REQUEST_NULL) {
            MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
        }
    }

    /**
     * Check for incoming bound from left neighbor
     * MUST be called only from main thread (MPI_THREAD_FUNNELED)
     */
    bool checkIncomingBound() {
        std::lock_guard<std::mutex> lock(mpiMutex);

        int flag;
        MPI_Status status;
        MPI_Iprobe(ring.getLeft(), TAG_RING_BOUND, ring.getComm(), &flag, &status);

        if (flag) {
            int receivedBound;
            MPI_Recv(&receivedBound, 1, MPI_INT, ring.getLeft(),
                     TAG_RING_BOUND, ring.getComm(), MPI_STATUS_IGNORE);
            messagesReceived++;

            // Use CAS for thread-safe update (FIX: atomic update)
            int expected = currentBound.load(std::memory_order_acquire);
            while (receivedBound < expected) {
                if (currentBound.compare_exchange_weak(expected, receivedBound,
                        std::memory_order_release, std::memory_order_relaxed)) {
                    forwardBoundLocked(receivedBound);
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Inject new bound into ring
     * MUST be called only from main thread
     */
    void injectBound(int newBound) {
        std::lock_guard<std::mutex> lock(mpiMutex);

        // Use CAS for thread-safe update
        int expected = currentBound.load(std::memory_order_acquire);
        while (newBound < expected) {
            if (currentBound.compare_exchange_weak(expected, newBound,
                    std::memory_order_release, std::memory_order_relaxed)) {
                forwardBoundLocked(newBound);
                return;
            }
        }
    }

    /**
     * Update bound locally without MPI (thread-safe)
     */
    void updateBoundLocal(int newBound) {
        int expected = currentBound.load(std::memory_order_acquire);
        while (newBound < expected) {
            if (currentBound.compare_exchange_weak(expected, newBound,
                    std::memory_order_release, std::memory_order_relaxed)) {
                return;
            }
        }
    }

    int getCurrentBound() const {
        return currentBound.load(std::memory_order_acquire);
    }

    uint64_t getMessagesReceived() const { return messagesReceived; }
    uint64_t getMessagesSent() const { return messagesSent; }

private:
    void forwardBoundLocked(int bound) {
        // Assumes mpiMutex is already held
        if (sendRequest != MPI_REQUEST_NULL) {
            MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
        }
        sendBuffer = bound;
        MPI_Isend(&sendBuffer, 1, MPI_INT, ring.getRight(),
                  TAG_RING_BOUND, ring.getComm(), &sendRequest);
        messagesSent++;
    }
};

// =============================================================================
// HybridRingSolver - OpenMP + AVX2 with Ring Bound Sharing
// =============================================================================

class HybridRingSolver {
private:
    int order;
    std::atomic<int> localBestLength;  // Thread-safe (FIX: was raw pointer)
    GolombRuler bestSolution;
    std::mutex solutionMutex;

    std::atomic<uint64_t> totalNodesExplored;
    std::atomic<uint64_t> totalNodesPruned;

    RingBoundManager* boundManager;
    int mpiRank, mpiSize;
    bool useSIMD;

    int cutoffDepth;

public:
    HybridRingSolver(int n, RingBoundManager* bm, int rank, int size, bool simd = true)
        : order(n), localBestLength(bm->getCurrentBound()),
          totalNodesExplored(0), totalNodesPruned(0),
          boundManager(bm), mpiRank(rank), mpiSize(size), useSIMD(simd) {

        bestSolution.length = localBestLength.load(std::memory_order_relaxed);

        // Adaptive cutoff for OpenMP task generation
        if (order <= 8) {
            cutoffDepth = 1;
        } else if (order <= 10) {
            cutoffDepth = 2;
        } else {
            cutoffDepth = 3;
        }
    }

    void solve(const Subtree& subtree) {
        // Initialize local best from subtree
        localBestLength.store(subtree.bestBound, std::memory_order_release);  // FIX: release not relaxed
        bestSolution.length = subtree.bestBound;

        #pragma omp parallel
        {
            #pragma omp single
            {
                ThreadState initialState;
                initializeFromSubtree(initialState, subtree);

                int startDepth = subtree.markCount;
                int lastMark = subtree.marks[subtree.markCount - 1];
                int currentBest = localBestLength.load(std::memory_order_acquire);
                int maxPos = currentBest - 1;

                // Generate OpenMP tasks for next level
                for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
                    int remainingMarks = order - startDepth - 1;
                    if (pos + remainingMarks >= currentBest) continue;

                    // Check differences
                    bool valid = true;
                    std::vector<int> newDiffs;
                    for (int i = 0; i < initialState.markCount && valid; ++i) {
                        int diff = pos - initialState.marks[i];
                        if (diff < 0 || diff >= MAX_LENGTH || initialState.usedDiffs.test(diff)) {
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

                            // Aggregate counters atomically
                            totalNodesExplored.fetch_add(state.localNodesExplored,
                                                        std::memory_order_relaxed);
                            totalNodesPruned.fetch_add(state.localNodesPruned,
                                                      std::memory_order_relaxed);
                        }
                    }
                }
            }
        }

        // After parallel region, sync with ring
        if (bestSolution.length < boundManager->getCurrentBound()) {
            boundManager->updateBoundLocal(bestSolution.length);
        }
    }

    Result getResult() const {
        Result r;
        r.order = order;
        r.length = bestSolution.length;
        r.nodesExplored = totalNodesExplored.load(std::memory_order_acquire);
        for (int i = 0; i < order && i < (int)bestSolution.marks.size(); ++i) {
            r.marks[i] = bestSolution.marks[i];
        }
        return r;
    }

private:
    void initializeFromSubtree(ThreadState& state, const Subtree& subtree) {
        state.markCount = subtree.markCount;
        for (int i = 0; i < subtree.markCount; ++i) {
            state.marks[i] = subtree.marks[i];
        }
        state.usedDiffs.reset();
        for (int d = 0; d < MAX_LENGTH; ++d) {
            if (subtree.testDiff(d)) state.usedDiffs.set(d);
        }
        state.localNodesExplored = 0;
        state.localNodesPruned = 0;
    }

    void branchAndBound(ThreadState& state, int depth) {
        state.localNodesExplored++;

        // Prefetch state data for upcoming operations
        __builtin_prefetch(&state.marks[0], 0, 3);
        __builtin_prefetch(&state.usedDiffs, 1, 3);

        // Periodic check for ring bound (main thread only)
        if (state.localNodesExplored % CHECK_INTERVAL == 0) {
            // Update local bound from ring manager
            int ringBound = boundManager->getCurrentBound();
            int localBound = localBestLength.load(std::memory_order_acquire);
            if (ringBound < localBound) {
                localBestLength.store(ringBound, std::memory_order_release);
            }
        }

        // Terminal case (rare)
        if (depth == order) [[unlikely]] {
            int length = state.marks[state.markCount - 1];
            updateGlobalBest(length, state);
            return;
        }

        int lastMark = state.marks[state.markCount - 1];
        int currentBest = localBestLength.load(std::memory_order_acquire);
        int maxPos = currentBest - 1;

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            int remainingMarks = order - depth - 1;
            // Pruning check (very common)
            if (pos + remainingMarks >= currentBest) [[likely]] {
                state.localNodesPruned++;
                continue;
            }

            // Check differences
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
                for (int i = 0; i < newDiffCount; ++i) {
                    state.usedDiffs.set(tempDiffs[i]);
                }

                branchAndBound(state, depth + 1);

                state.markCount--;
                for (int i = 0; i < newDiffCount; ++i) {
                    state.usedDiffs.clear(tempDiffs[i]);
                }

                currentBest = localBestLength.load(std::memory_order_acquire);
                maxPos = currentBest - 1;
            }
        }
    }

    inline bool checkDifferencesScalar(ThreadState& state, int pos, int* tempDiffs, int& diffCount) {
        diffCount = 0;
        for (int i = 0; i < state.markCount; ++i) {
            int diff = pos - state.marks[i];
            // FIX: Check diff < 0
            if (diff < 0 || diff >= MAX_LENGTH || state.usedDiffs.test(diff)) {
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
            _mm256_store_si256((__m256i*)&allDiffs[totalDiffs], vdiffs);
            totalDiffs += 8;
        }

        for (; i < state.markCount; ++i) {
            allDiffs[totalDiffs++] = pos - state.marks[i];
        }

        BitSet256 checkMask;
        checkMask.reset();

        for (int j = 0; j < totalDiffs; ++j) {
            int d = allDiffs[j];
            // FIX: Check d < 0
            if (d < 0 || d >= MAX_LENGTH) {
                return false;
            }
            checkMask.set(d);
        }

        if (state.usedDiffs.hasCollisionAVX2(checkMask)) {
            return false;
        }

        diffCount = totalDiffs;
        for (int j = 0; j < totalDiffs; ++j) {
            tempDiffs[j] = allDiffs[j];
        }

        return true;
    }
#endif

    void updateGlobalBest(int length, const ThreadState& state) {
        // FIX: Use CAS instead of check-then-act
        int expected = localBestLength.load(std::memory_order_acquire);
        while (length < expected) {
            if (localBestLength.compare_exchange_weak(expected, length,
                    std::memory_order_release, std::memory_order_relaxed)) {

                // Success - update solution under mutex
                std::lock_guard<std::mutex> lock(solutionMutex);
                std::vector<int> marks(state.marks, state.marks + state.markCount);
                bestSolution = GolombRuler(marks);

                // Update ring bound manager (will be sent at end of solve())
                boundManager->updateBoundLocal(length);
                break;
            }
        }
    }
};

// =============================================================================
// Subtree Generation
// =============================================================================

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
                if (diff < 0 || diff >= MAX_LENGTH || current.testDiff(diff)) {
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

// =============================================================================
// MPI Datatypes
// =============================================================================

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
    int bl[] = {MAX_ORDER, 1, 1, 1, 1};
    MPI_Aint disp[5] = {
        offsetof(Result, marks), offsetof(Result, length),
        offsetof(Result, order), offsetof(Result, nodesExplored),
        offsetof(Result, timeMs)
    };
    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_INT, MPI_UINT64_T, MPI_DOUBLE};
    MPI_Type_create_struct(5, bl, disp, types, &type);
    MPI_Type_commit(&type);
    return type;
}

// =============================================================================
// Drain Ring Messages
// =============================================================================

void drainRingMessages(RingCommunicator& ring) {
    int flag;
    MPI_Status status;
    while (true) {
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_RING_BOUND, ring.getComm(), &flag, &status);
        if (!flag) break;
        int tmp;
        MPI_Recv(&tmp, 1, MPI_INT, MPI_ANY_SOURCE, TAG_RING_BOUND,
                 ring.getComm(), MPI_STATUS_IGNORE);
    }
}

// =============================================================================
// CSV Output
// =============================================================================

void appendHybridCSV(const std::string& filename, int order, int procs, int threads,
                     double timeMs, double seqTime, uint64_t nodes,
                     uint64_t ringMsgs, const GolombRuler& solution) {
    std::ifstream check(filename);
    bool header = !check.good();
    check.close();

    std::ofstream f(filename, std::ios::app);
    if (!f.is_open()) return;

    if (header) {
        f << "version,order,procs,threads,workers,time_ms,speedup,efficiency,nodes,ring_msgs,solution,length\n";
    }

    int totalWorkers = procs * threads;
    double speedup = (seqTime > 0) ? seqTime / timeMs : 0;
    double efficiency = speedup / totalWorkers;

    f << 7 << "," << order << "," << procs << "," << threads << "," << totalWorkers << ","
      << std::fixed << std::setprecision(2) << timeMs << ","
      << std::setprecision(2) << speedup << ","
      << std::setprecision(3) << efficiency << ","
      << nodes << "," << ringMsgs << ","
      << "\"" << solution.toString() << "\"," << solution.length << "\n";

    f.close();
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    // Initialize MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "Warning: MPI_THREAD_FUNNELED not supported, using single-threaded MPI\n";
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            std::cout << "Usage: mpirun -np <procs> " << argv[0] << " <order> [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --threads N   OpenMP threads per rank (default: OMP_NUM_THREADS)\n";
            std::cout << "  --depth N     Prefix depth (default: auto)\n";
            std::cout << "  --csv FILE    Save results to CSV\n";
            std::cout << "  --seq-time T  Sequential time for speedup calc\n";
            std::cout << "  --no-simd     Disable AVX2 optimizations\n";
        }
        MPI_Finalize();
        return 1;
    }

    int order = parseAndValidateOrder(argv[1]);
    if (order < 0 || order < 4) {
        if (rank == 0) {
            std::cerr << "Error: Order must be at least 4 for MPI version\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Parse options
    int prefixDepth = (order <= 8) ? 3 : (order <= 10) ? 4 : 5;
    int numThreads = omp_get_max_threads();
    std::string csvFile;
    double seqTime = 0;
    bool useSIMD = true;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threads" && i+1 < argc) {
            numThreads = std::atoi(argv[++i]);
        }
        else if (arg == "--depth" && i+1 < argc) {
            int pd = std::atoi(argv[++i]);
            if (pd > 0) prefixDepth = pd;
        }
        else if (arg == "--csv" && i+1 < argc) {
            csvFile = argv[++i];
        }
        else if (arg == "--seq-time" && i+1 < argc) {
            seqTime = std::atof(argv[++i]);
        }
        else if (arg == "--no-simd") {
            useSIMD = false;
        }
    }

    omp_set_num_threads(numThreads);

    // Create ring topology
    RingCommunicator ring;
    ring.initialize(size);

    // Create MPI datatypes
    MPI_Datatype subtreeType = createSubtreeType();
    MPI_Datatype resultType = createResultType();

    double startTime = MPI_Wtime();

    if (rank == 0) {
        // ========== MASTER ==========
        std::cout << "=== Golomb Ruler Solver - MPI v7: Hybrid Ring Shifter ===\n";
        std::cout << "Order: " << order << ", Ranks: " << size
                  << ", Threads/rank: " << numThreads
                  << ", Total workers: " << (size * numThreads) << "\n";
        std::cout << "Prefix depth: " << prefixDepth << "\n";
        std::cout << "Ring topology: left=" << ring.getLeft() << ", right=" << ring.getRight() << "\n";
#ifdef USE_AVX2
        std::cout << "AVX2: " << (useSIMD ? "Enabled" : "Disabled") << "\n";
#else
        std::cout << "AVX2: Not compiled\n";
#endif

        int bound = computeGreedyBound(order);
        std::cout << "Initial bound: " << bound << "\n";

        std::vector<Subtree> subtrees = generateSubtrees(order, prefixDepth, bound);
        std::cout << "Subtrees: " << subtrees.size() << "\n";

        // Initialize ring bound manager
        RingBoundManager boundManager(ring, bound);

        // Dynamic work distribution
        size_t nextSubtree = 0;
        int activeWorkers = 0;

        for (int w = 1; w < size && nextSubtree < subtrees.size(); ++w) {
            subtrees[nextSubtree].bestBound = boundManager.getCurrentBound();
            MPI_Send(&subtrees[nextSubtree], 1, subtreeType, w, TAG_WORK, MPI_COMM_WORLD);
            nextSubtree++;
            activeWorkers++;
        }

        GolombRuler globalBest;
        globalBest.length = bound;
        uint64_t totalNodes = 0;

        while (activeWorkers > 0) {
            // Check ring for incoming bounds
            boundManager.checkIncomingBound();

            // Check for results
            int flag;
            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &flag, &status);

            if (flag) {
                Result result;
                MPI_Recv(&result, 1, resultType, status.MPI_SOURCE,
                         TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                activeWorkers--;
                totalNodes += result.nodesExplored;

                if (result.length < globalBest.length && result.length < INT_MAX) {
                    globalBest.length = result.length;
                    globalBest.order = result.order;
                    globalBest.marks.assign(result.marks, result.marks + result.order);
                    boundManager.injectBound(result.length);
                }

                if (nextSubtree < subtrees.size()) {
                    subtrees[nextSubtree].bestBound = boundManager.getCurrentBound();
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

        MPI_Barrier(MPI_COMM_WORLD);
        drainRingMessages(ring);

        double endTime = MPI_Wtime();
        double elapsedMs = (endTime - startTime) * 1000.0;

        uint64_t totalRingMsgs = boundManager.getMessagesSent() + boundManager.getMessagesReceived();

        std::cout << "\n=== Results ===\n";
        std::cout << "Solution: " << globalBest.toString() << "\n";
        std::cout << "Length: " << globalBest.length << "\n";
        std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsedMs << " ms\n";
        std::cout << "Nodes: " << totalNodes << "\n";
        std::cout << "Ring messages: " << totalRingMsgs << "\n";

        if (seqTime > 0) {
            double speedup = seqTime / elapsedMs;
            std::cout << "Speedup: " << std::setprecision(2) << speedup << "x\n";
            std::cout << "Efficiency: " << std::setprecision(1)
                      << 100 * speedup / (size * numThreads) << "%\n";
        }

        if (order <= 14 && globalBest.length == OPTIMAL_LENGTHS[order]) {
            std::cout << "*** OPTIMAL ***\n";
        }

        if (!csvFile.empty()) {
            appendHybridCSV(csvFile, order, size, numThreads, elapsedMs, seqTime,
                           totalNodes, totalRingMsgs, globalBest);
        }

    } else {
        // ========== WORKER ==========
        RingBoundManager boundManager(ring, INT_MAX);

        while (true) {
            boundManager.checkIncomingBound();

            int flag;
            MPI_Status status;
            MPI_Iprobe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

            if (flag) {
                if (status.MPI_TAG == TAG_DONE) {
                    Subtree dummy;
                    MPI_Recv(&dummy, 1, subtreeType, 0, TAG_DONE,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    break;
                }

                Subtree subtree;
                MPI_Recv(&subtree, 1, subtreeType, 0, TAG_WORK,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                boundManager.updateBoundLocal(subtree.bestBound);

                double workerStart = MPI_Wtime();

                HybridRingSolver solver(order, &boundManager, rank, size, useSIMD);
                solver.solve(subtree);

                Result result = solver.getResult();
                result.timeMs = (MPI_Wtime() - workerStart) * 1000.0;

                // Inject best bound found into ring
                if (result.length < boundManager.getCurrentBound()) {
                    boundManager.injectBound(result.length);
                }

                MPI_Send(&result, 1, resultType, 0, TAG_RESULT, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        drainRingMessages(ring);
    }

    ring.cleanup();
    MPI_Type_free(&subtreeType);
    MPI_Type_free(&resultType);
    MPI_Finalize();

    return 0;
}
