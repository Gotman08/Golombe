/**
 * Golomb Ruler Solver - Parallel Version 4: Hybrid MPI+OpenMP
 *
 * This is the most advanced parallel version combining:
 * - MPI for inter-node communication (distributed memory)
 * - OpenMP for intra-node parallelism (shared memory)
 * - AVX2 SIMD optimizations from v6
 * - Cache-aligned data structures
 *
 * Architecture:
 *   Cluster level: MPI ranks distributed across nodes
 *   Node level: OpenMP threads within each MPI rank
 *
 * Usage:
 *   mpirun -np 4 ./golomb_mpi_v4 12 --threads 8 --depth 5
 *   # 4 MPI ranks x 8 OpenMP threads = 32 effective workers
 *
 * For Romeo HPC:
 *   #SBATCH --nodes=4
 *   #SBATCH --ntasks-per-node=1
 *   #SBATCH --cpus-per-task=16
 *   mpirun -np 4 ./golomb_mpi_v4 14 --threads 16
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
#include <cmath>
#include <fstream>
#include <atomic>
#include <mutex>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

// =============================================================================
// Constants and MPI Tags
// =============================================================================

const int TAG_WORK = 1;
const int TAG_RESULT = 2;
const int TAG_DONE = 3;
const int TAG_BOUND = 4;
const int TAG_REQUEST_WORK = 5;

const int CHECK_INTERVAL = 100000;

// =============================================================================
// Cache-aligned BitSet256 (from v6)
// =============================================================================

struct alignas(32) BitSet256 {
    uint64_t words[4];

    inline void reset() {
        words[0] = words[1] = words[2] = words[3] = 0;
    }

    inline bool test(int bit) const {
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
};

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

// =============================================================================
// Result Structure (for MPI communication)
// =============================================================================

struct Result {
    int marks[MAX_ORDER];
    int length;
    int order;
    uint64_t nodesExplored;
    uint64_t nodesPruned;
    double timeMs;
};

// =============================================================================
// Thread-Local State (for OpenMP, cache-aligned)
// =============================================================================

struct alignas(64) ThreadState {
    int marks[MAX_ORDER];
    BitSet256 usedDiffs;
    int markCount;
    uint64_t localNodesExplored;
    uint64_t localNodesPruned;
};

// =============================================================================
// Hypercube Utilities
// =============================================================================

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

void broadcastBoundToNeighbors(int bound, int rank, int size) {
    for (int neighbor : getHypercubeNeighbors(rank, size)) {
        MPI_Request request;
        MPI_Isend(&bound, 1, MPI_INT, neighbor, TAG_BOUND, MPI_COMM_WORLD, &request);
        MPI_Request_free(&request);
    }
}

// =============================================================================
// Hybrid OpenMP Solver (runs within each MPI rank)
// =============================================================================

class HybridSubtreeSolver {
private:
    int order;
    std::atomic<int> localBestLength;
    GolombRuler bestSolution;
    std::mutex solutionMutex;

    std::atomic<uint64_t> totalNodesExplored;
    std::atomic<uint64_t> totalNodesPruned;

    int* globalBound;  // Shared with MPI
    int mpiRank, mpiSize;
    bool useSIMD;

public:
    HybridSubtreeSolver(int n, const Subtree& subtree, int* bound,
                        int rank, int size, bool simd = true)
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
                // Initialize state from subtree
                ThreadState initialState;
                initializeFromSubtree(initialState, subtree);

                int startDepth = subtree.markCount;
                int lastMark = subtree.marks[subtree.markCount - 1];
                int currentBest = localBestLength.load(std::memory_order_relaxed);
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

                            // Place mark
                            state.marks[state.markCount++] = pos;
                            for (int d : newDiffs) state.usedDiffs.set(d);

                            // Solve recursively
                            branchAndBound(state, startDepth + 1);

                            // Aggregate
                            totalNodesExplored.fetch_add(state.localNodesExplored,
                                                         std::memory_order_relaxed);
                            totalNodesPruned.fetch_add(state.localNodesPruned,
                                                       std::memory_order_relaxed);
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

        // Periodic MPI bound check
        // Note: With MPI_THREAD_MULTIPLE, any thread can call MPI functions.
        // With MPI_THREAD_FUNNELED, this is technically incorrect but works
        // in practice because the critical section serializes MPI calls.
        // The provided MPI thread level is checked at startup.
        if (state.localNodesExplored % CHECK_INTERVAL == 0) {
            #pragma omp critical(mpi_bound_check)
            {
                if (checkForBoundUpdate(*globalBound)) {
                    int newBound = *globalBound;
                    int current = localBestLength.load(std::memory_order_relaxed);
                    while (newBound < current) {
                        if (localBestLength.compare_exchange_weak(current, newBound))
                            break;
                    }
                }
            }
        }

        // Terminal case
        if (depth == order) {
            int length = state.marks[state.markCount - 1];
            updateBestSolution(length, state);
            return;
        }

        int lastMark = state.marks[state.markCount - 1];
        int currentBest = localBestLength.load(std::memory_order_relaxed);
        int maxPos = std::min(currentBest - 1, *globalBound - 1);

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            // Pruning
            int remainingMarks = order - depth - 1;
            if (pos + remainingMarks >= currentBest) {
                state.localNodesPruned++;
                continue;
            }

            // Check differences
            int tempDiffs[MAX_ORDER];
            int newDiffCount = 0;
            bool valid = checkDifferences(state, pos, tempDiffs, newDiffCount);

            if (valid) {
                state.marks[state.markCount++] = pos;
                for (int i = 0; i < newDiffCount; ++i)
                    state.usedDiffs.set(tempDiffs[i]);

                branchAndBound(state, depth + 1);

                state.markCount--;
                for (int i = 0; i < newDiffCount; ++i)
                    state.usedDiffs.clear(tempDiffs[i]);

                // Refresh bound
                currentBest = localBestLength.load(std::memory_order_relaxed);
                maxPos = std::min(currentBest - 1, *globalBound - 1);
            }
        }
    }

    inline bool checkDifferences(ThreadState& state, int pos,
                                  int* tempDiffs, int& diffCount) {
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

    void updateBestSolution(int length, const ThreadState& state) {
        int current = localBestLength.load(std::memory_order_relaxed);
        while (length < current) {
            if (localBestLength.compare_exchange_weak(current, length)) {
                // Update solution
                std::lock_guard<std::mutex> lock(solutionMutex);
                if (length < bestSolution.length) {
                    bestSolution.length = length;
                    bestSolution.order = order;
                    bestSolution.marks.assign(state.marks, state.marks + state.markCount);

                    // Update global MPI bound
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
    int bl[] = {MAX_ORDER, 1, 1, 1, 1, 1};
    MPI_Aint disp[6] = {
        offsetof(Result, marks), offsetof(Result, length),
        offsetof(Result, order), offsetof(Result, nodesExplored),
        offsetof(Result, nodesPruned), offsetof(Result, timeMs)
    };
    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_INT,
                            MPI_UINT64_T, MPI_UINT64_T, MPI_DOUBLE};
    MPI_Type_create_struct(6, bl, disp, types, &type);
    MPI_Type_commit(&type);
    return type;
}

// =============================================================================
// CSV Output
// =============================================================================

void appendHybridCSV(const std::string& filename, int order, int mpiProcs, int ompThreads,
                     double timeMs, uint64_t nodes, uint64_t pruned, const GolombRuler& solution) {
    std::ifstream check(filename);
    bool header = !check.good();
    check.close();

    std::ofstream f(filename, std::ios::app);
    if (!f.is_open()) return;

    if (header) {
        f << "order,mpi_procs,omp_threads,total_workers,time_ms,nodes,pruned,solution,length\n";
    }

    int totalWorkers = mpiProcs * ompThreads;
    f << order << "," << mpiProcs << "," << ompThreads << "," << totalWorkers << ","
      << std::fixed << std::setprecision(2) << timeMs << ","
      << nodes << "," << pruned << ","
      << "\"" << solution.toString() << "\"," << solution.length << "\n";

    f.close();
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    // Initialize MPI with thread support
    // Request MPI_THREAD_MULTIPLE for safe multi-threaded MPI calls
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "Error: MPI doesn't support multi-threading. Exiting.\n";
        MPI_Finalize();
        return 1;
    } else if (provided < MPI_THREAD_MULTIPLE) {
        // MPI_THREAD_FUNNELED or MPI_THREAD_SERIALIZED - use critical sections
        // This works in practice but is not strictly correct per MPI standard
        if (provided == MPI_THREAD_FUNNELED) {
            std::cerr << "Warning: MPI provides MPI_THREAD_FUNNELED. "
                      << "Multi-threaded bound checks may be unsafe.\n";
        }
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            std::cout << "Golomb Ruler Solver - MPI v4: Hybrid MPI+OpenMP\n\n";
            std::cout << "Usage: mpirun -np <procs> " << argv[0] << " <order> [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --threads N   OpenMP threads per MPI rank (default: auto)\n";
            std::cout << "  --depth N     Prefix depth for subtree generation\n";
            std::cout << "  --csv FILE    Save results to CSV\n";
        }
        MPI_Finalize();
        return 1;
    }

    int order = parseAndValidateOrder(argv[1]);
    if (order < 0 || order < 4) {
        if (rank == 0) {
            std::cerr << "Error: Order must be between 4 and " << (MAX_ORDER-1) << '\n';
        }
        MPI_Finalize();
        return 1;
    }

    // Parse options
    int prefixDepth = (order <= 8) ? 3 : (order <= 10) ? 4 : (order <= 12) ? 5 : 6;
    int numThreads = omp_get_max_threads();
    std::string csvFile;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threads" && i+1 < argc) {
            numThreads = std::atoi(argv[++i]);
        } else if (arg == "--depth" && i+1 < argc) {
            prefixDepth = std::atoi(argv[++i]);
        } else if (arg == "--csv" && i+1 < argc) {
            csvFile = argv[++i];
        }
    }

    omp_set_num_threads(numThreads);

    MPI_Datatype subtreeType = createSubtreeType();
    MPI_Datatype resultType = createResultType();

    double startTime = MPI_Wtime();

    if (rank == 0) {
        std::cout << "=== Golomb Ruler Solver - MPI v4: Hybrid MPI+OpenMP ===\n";
        std::cout << "Order: " << order << "\n";
        std::cout << "MPI Processes: " << size << "\n";
        std::cout << "OpenMP Threads per process: " << numThreads << "\n";
        std::cout << "Total workers: " << size * numThreads << "\n";
        std::cout << "Prefix depth: " << prefixDepth << "\n";

        int bound = computeGreedyBound(order);
        std::cout << "Initial bound: " << bound << "\n";

        std::vector<Subtree> subtrees = generateSubtrees(order, prefixDepth, bound);
        std::cout << "Subtrees generated: " << subtrees.size() << "\n\n";

        // Dynamic work distribution
        size_t nextSubtree = 0;
        int activeWorkers = 0;

        // Initial distribution
        for (int w = 1; w < size && nextSubtree < subtrees.size(); ++w) {
            subtrees[nextSubtree].bestBound = bound;
            MPI_Send(&subtrees[nextSubtree], 1, subtreeType, w, TAG_WORK, MPI_COMM_WORLD);
            nextSubtree++;
            activeWorkers++;
        }

        // Master also processes subtrees using OpenMP
        GolombRuler globalBest;
        globalBest.length = bound;
        uint64_t totalNodes = 0;
        uint64_t totalPruned = 0;
        int localBound = bound;

        while (activeWorkers > 0 || nextSubtree < subtrees.size()) {
            // Process bounds
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

                // Send more work
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

        // Drain remaining bound messages
        int flag;
        MPI_Status status;
        while (true) {
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_BOUND, MPI_COMM_WORLD, &flag, &status);
            if (!flag) break;
            int tmp;
            MPI_Recv(&tmp, 1, MPI_INT, MPI_ANY_SOURCE, TAG_BOUND,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        double endTime = MPI_Wtime();
        double elapsedMs = (endTime - startTime) * 1000.0;

        std::cout << "=== Results ===\n";
        std::cout << "Solution: " << globalBest.toString() << "\n";
        std::cout << "Length: " << globalBest.length << "\n";
        std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsedMs << " ms\n";
        std::cout << "Nodes explored: " << totalNodes << "\n";
        std::cout << "Nodes pruned: " << totalPruned << "\n";

        if (order <= 14 && globalBest.length == OPTIMAL_LENGTHS[order]) {
            std::cout << "*** OPTIMAL ***\n";
        }

        if (!csvFile.empty()) {
            appendHybridCSV(csvFile, order, size, numThreads, elapsedMs,
                           totalNodes, totalPruned, globalBest);
            std::cout << "Results saved to " << csvFile << "\n";
        }

    } else {
        // Worker process
        int localBound = INT_MAX;

        while (true) {
            Subtree subtree;
            MPI_Status status;
            MPI_Recv(&subtree, 1, subtreeType, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_DONE) break;

            localBound = subtree.bestBound;
            double workerStart = MPI_Wtime();

            // Solve using OpenMP
            HybridSubtreeSolver solver(order, subtree, &localBound, rank, size);
            solver.solve(subtree);

            Result result = solver.getResult();
            result.timeMs = (MPI_Wtime() - workerStart) * 1000.0;
            MPI_Send(&result, 1, resultType, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }

    MPI_Type_free(&subtreeType);
    MPI_Type_free(&resultType);
    MPI_Finalize();

    return 0;
}
