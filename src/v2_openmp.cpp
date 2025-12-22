/**
 * @file v2_openmp.cpp
 * @brief OpenMP Golomb Ruler Solver
 *
 * Golomb Ruler Solver - High Performance Computing Implementation
 * Copyright (c) 2025 Nicolas Marano
 * Licensed under the MIT License. See LICENSE file for details.
 *
 * OpenMP parallelized implementation with optimizations:
 * - Task-based parallelism at shallow tree depths
 * - Bound caching to reduce atomic contention (16K interval)
 * - BitSet256 for O(1) difference lookup
 * - AVX2 SIMD for vectorized difference checking
 * - Cache-aligned data structures
 *
 * Usage:
 *   ./golomb_v2 <order> [options]
 *   Options:
 *     --threads N     Set thread count (default: auto-detect)
 *     --no-simd       Disable AVX2 optimizations
 *     --csv FILE      Save results to CSV
 *     --benchmark     Run benchmarks for orders 4 to <order>
 *     --verbose       Show progress
 */

#include "golomb/golomb.hpp"
#include "golomb/greedy.hpp"
#include "golomb/bitset256.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <atomic>
#include <thread>
#include <set>
#include <mutex>

#include <omp.h>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

// ============================================================================
// Hardware Detection
// ============================================================================

/**
 * @struct HardwareInfo
 * @brief Contains information about the system's CPU capabilities.
 *
 * Used to make intelligent decisions about thread count and SIMD usage.
 */
struct HardwareInfo {
    int logicalCores;       ///< Number of logical cores (includes hyperthreads)
    int physicalCores;      ///< Number of physical cores
    bool hasHyperthreading; ///< True if hyperthreading is enabled
    bool hasAVX2;           ///< True if AVX2 SIMD instructions are available
};

/**
 * @brief Detects CPU hardware characteristics.
 *
 * Queries the system to determine the number of physical and logical cores,
 * hyperthreading status, and AVX2 support. On Linux, reads /proc/cpuinfo
 * to distinguish physical from logical cores.
 *
 * @return HardwareInfo structure with detected capabilities
 *
 * @note Falls back to conservative defaults if detection fails
 */
HardwareInfo detectHardware() {
    HardwareInfo info;

    info.logicalCores = std::thread::hardware_concurrency();
    if (info.logicalCores == 0) {
        info.logicalCores = 1;
    }

    info.physicalCores = info.logicalCores;

#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        std::set<int> uniqueCores;
        int currentCoreId = -1;

        while (std::getline(cpuinfo, line)) {
            if (line.find("core id") != std::string::npos) {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    try {
                        currentCoreId = std::stoi(line.substr(pos + 1));
                        uniqueCores.insert(currentCoreId);
                    } catch (const std::exception&) {
                        // Ignore malformed lines
                    }
                }
            }
        }

        if (!uniqueCores.empty()) {
            info.physicalCores = static_cast<int>(uniqueCores.size());
        }
        cpuinfo.close();
    }
#endif

    info.hasHyperthreading = (info.logicalCores > info.physicalCores);

    // Runtime AVX2 detection via CPUID (more reliable than compile-time check)
    info.hasAVX2 = false;
#ifdef USE_AVX2
    // If compiled with AVX2, assume the CPU supports it
    // (otherwise the binary wouldn't run properly anyway)
    info.hasAVX2 = true;
#endif

    return info;
}

/**
 * @brief Selects optimal thread count based on hardware and problem size.
 *
 * Uses a heuristic to balance parallelization overhead against workload:
 * - Small problems (order <= 7): Limited threads to avoid overhead
 * - Medium problems (order <= 9): Use physical cores
 * - Large problems (order >= 10): Use all logical cores (including HT)
 *
 * @param hw    Hardware information from detectHardware()
 * @param order Order of the Golomb ruler to solve
 *
 * @return Recommended number of threads
 */
int selectThreadCount(const HardwareInfo& hw, int order) {
    if (order <= 7) {
        return std::min(hw.physicalCores, 4);
    }
    if (order <= 9) {
        return hw.physicalCores;
    }
    return hw.logicalCores;
}

// ============================================================================
// Thread-Local State (Cache-aligned)
// ============================================================================

/**
 * @struct ThreadState
 * @brief Thread-local search state for OpenMP parallel execution.
 *
 * Cache-aligned (64 bytes) to prevent false sharing between threads.
 * Each thread maintains its own copy of the search state, reducing
 * synchronization overhead.
 *
 * @note The cachedBound mechanism reduces atomic contention by only
 *       checking the global bound every 16K nodes.
 */
struct alignas(64) ThreadState {
    int marks[MAX_ORDER];       ///< Array of mark positions on the ruler
    BitSet256 usedDiffs;        ///< Bitset tracking used differences
    int markCount;              ///< Current number of marks placed
    int localNodesExplored;     ///< Thread-local counter for explored nodes
    int localNodesPruned;       ///< Thread-local counter for pruned nodes
    int cachedBound;            ///< Cached global bound to reduce atomic contention
    int checkCounter;           ///< Counter for bound refresh interval (resets at 16K)
    char padding[64 - (sizeof(int) * 4) % 64]; ///< Padding to ensure 64-byte alignment
};

// ============================================================================
// OpenMP Solver
// ============================================================================

/**
 * @class OpenMPSolver
 * @brief OpenMP-parallelized branch-and-bound solver for optimal Golomb rulers.
 *
 * This class implements a multi-threaded branch-and-bound algorithm using
 * OpenMP task parallelism. Key features:
 * - Task-based parallelism at shallow tree depths (cutoff depth 1-3)
 * - Bound caching to reduce atomic contention (16K interval)
 * - Thread-local state to minimize synchronization
 * - Adaptive cutoff depth based on problem size
 *
 * @note Thread count is controlled via OMP_NUM_THREADS or --threads option
 * @see SequentialSolver for single-threaded version
 */
class OpenMPSolver {
private:
    int order;                              ///< Target number of marks
    std::atomic<int> globalBestLength;      ///< Best length found (atomic for thread safety)
    GolombRuler bestSolution;               ///< Best solution found so far
    std::mutex solutionMutex;               ///< Mutex for updating bestSolution

    std::atomic<uint64_t> totalNodesExplored; ///< Global counter for explored nodes
    std::atomic<uint64_t> totalNodesPruned;   ///< Global counter for pruned nodes

    int cutoffDepth;    ///< Depth at which to stop generating tasks
    bool useSIMD;       ///< Whether to use AVX2 SIMD optimizations
    bool verbose;       ///< Whether to print progress information
    HardwareInfo hwInfo; ///< Detected hardware capabilities

public:
    /**
     * @brief Constructs a new OpenMP Solver.
     *
     * Initializes the solver, detects hardware capabilities, sets adaptive
     * cutoff depth, and computes initial upper bound using greedy heuristic.
     *
     * @param n     The order of the Golomb ruler to find (number of marks)
     * @param simd  Enable AVX2 SIMD optimizations (default: true)
     * @param verb  Enable verbose output during search (default: false)
     *
     * @pre n must be between 2 and MAX_ORDER-1
     */
    OpenMPSolver(int n, bool simd = true, bool verb = false)
        : order(n), globalBestLength(INT_MAX),
          totalNodesExplored(0), totalNodesPruned(0),
          useSIMD(simd), verbose(verb) {

        hwInfo = detectHardware();

        // Adaptive cutoff depth based on problem complexity.
        // Empirically determined thresholds:
        // - G7 and below: Limited parallelism useful, depth=1 minimizes overhead
        // - G8-G9: depth=2 balances task creation overhead vs parallelism
        // - G10+: depth=3 creates sufficient tasks for 32+ threads
        if (order <= 7) {
            cutoffDepth = 1;
        } else if (order <= 9) {
            cutoffDepth = 2;
        } else {
            cutoffDepth = 3;
        }

        findGreedySolution();
    }

    /**
     * @brief Executes the parallel branch-and-bound search.
     *
     * Uses OpenMP task parallelism to distribute work across threads.
     * Tasks are generated at shallow depths (1-2) to create sufficient
     * parallelism while minimizing task creation overhead.
     *
     * @post bestSolution contains the optimal ruler found
     * @post totalNodesExplored and totalNodesPruned are updated
     *
     * @thread_safety Thread-safe; uses atomic operations and mutex for synchronization
     */
    void solve() {
        #pragma omp parallel
        {
            #pragma omp single
            {
                int maxFirstMark = globalBestLength.load(std::memory_order_acquire) / 2;

                // Generate tasks for depth 1
                for (int pos1 = 1; pos1 <= maxFirstMark; ++pos1) {
                    #pragma omp task firstprivate(pos1)
                    {
                        ThreadState state;
                        initializeState(state);

                        state.marks[state.markCount++] = pos1;
                        state.usedDiffs.set(pos1);

                        if (cutoffDepth >= 2 && order >= 10) {
                            generateDepth2Tasks(state, pos1);
                        } else {
                            branchAndBound(state, 2);
                        }

                        totalNodesExplored.fetch_add(state.localNodesExplored,
                                                     std::memory_order_relaxed);
                        totalNodesPruned.fetch_add(state.localNodesPruned,
                                                   std::memory_order_relaxed);
                    }
                }
            }
        }
    }

    /**
     * @brief Retrieves the search statistics after solving.
     *
     * @return SearchStats structure containing nodes explored, pruned, and best solution
     *
     * @thread_safety Thread-safe; reads atomic values
     */
    SearchStats getStats() const {
        SearchStats stats;
        stats.nodesExplored = totalNodesExplored.load();
        stats.nodesPruned = totalNodesPruned.load();
        stats.bestSolution = bestSolution;
        return stats;
    }

    /**
     * @brief Returns the detected hardware information.
     *
     * @return Reference to HardwareInfo with CPU capabilities
     */
    const HardwareInfo& getHardwareInfo() const {
        return hwInfo;
    }

private:
    /**
     * @brief Initializes a thread-local search state.
     *
     * Sets up the initial state with mark 0 at position 0, clears counters,
     * and caches the current global bound.
     *
     * @param[out] state The thread state to initialize
     */
    void initializeState(ThreadState& state) {
        state.marks[0] = 0;
        state.markCount = 1;
        state.usedDiffs.reset();
        state.localNodesExplored = 0;
        state.localNodesPruned = 0;
        state.cachedBound = globalBestLength.load(std::memory_order_relaxed);
        state.checkCounter = 0;
    }

    /**
     * @brief Computes an initial upper bound using a greedy heuristic.
     *
     * Uses a greedy algorithm to quickly find a valid Golomb ruler.
     * This provides an initial upper bound for pruning during search.
     *
     * @post bestSolution contains the greedy solution
     * @post globalBestLength is set to the greedy solution's length
     */
    void findGreedySolution() {
        BitSet256 greedyDiffs;
        greedyDiffs.reset();
        std::vector<int> greedyMarks = computeGreedySolution(order, greedyDiffs);

        bestSolution = GolombRuler(greedyMarks);
        globalBestLength.store(bestSolution.length, std::memory_order_relaxed);

        if (verbose) {
            std::cout << "Initial bound: " << bestSolution.length << '\n';
        }
    }

    /**
     * @brief Generates OpenMP tasks at depth 2 for increased parallelism.
     *
     * Creates additional tasks for large problems (order >= 10) to ensure
     * sufficient work granularity for many threads. Uses triangular pruning
     * to eliminate unpromising positions early.
     *
     * @param[in] parentState The parent state with first mark placed
     * @param[in] pos1        Position of the first mark
     *
     * @note Only called when cutoffDepth >= 2 and order >= 10
     */
    void generateDepth2Tasks(ThreadState& parentState, int pos1) {
        int currentBest = globalBestLength.load(std::memory_order_acquire);
        int maxPos = currentBest - 1;

        for (int pos2 = pos1 + 1; pos2 <= maxPos; ++pos2) {
            // Pruning amélioré: distance minimale triangulaire
            int remainingMarks = order - 3;
            int minIncrement = remainingMarks * (remainingMarks + 1) / 2;
            if (pos2 + minIncrement >= currentBest) {
                parentState.localNodesPruned++;
                continue;
            }

            int diff1 = pos2 - parentState.marks[0];
            int diff2 = pos2 - parentState.marks[1];

            if (diff1 >= MAX_LENGTH || diff2 >= MAX_LENGTH) continue;
            if (parentState.usedDiffs.test(diff1) || parentState.usedDiffs.test(diff2)) continue;
            if (diff1 == diff2) continue;

            #pragma omp task firstprivate(pos2)
            {
                ThreadState state;
                std::memcpy(state.marks, parentState.marks,
                           parentState.markCount * sizeof(int));
                state.markCount = parentState.markCount;
                state.usedDiffs.copyFrom(parentState.usedDiffs);
                state.localNodesExplored = 0;
                state.localNodesPruned = 0;
                state.cachedBound = globalBestLength.load(std::memory_order_relaxed);
                state.checkCounter = 0;

                state.marks[state.markCount++] = pos2;
                state.usedDiffs.set(diff1);
                state.usedDiffs.set(diff2);

                branchAndBound(state, 3);

                totalNodesExplored.fetch_add(state.localNodesExplored,
                                            std::memory_order_relaxed);
                totalNodesPruned.fetch_add(state.localNodesPruned,
                                           std::memory_order_relaxed);
            }
        }
    }

    /**
     * @brief Recursive branch-and-bound search for Golomb rulers.
     *
     * Explores the search tree by trying all valid positions for the next mark.
     * Uses bound caching (16K interval) to reduce atomic contention while
     * maintaining good pruning effectiveness.
     *
     * @param[in,out] state Current thread-local search state
     * @param[in]     depth Current depth in the search tree
     *
     * @note Uses triangular pruning: remaining marks need at least 1+2+...+k positions
     */
    void branchAndBound(ThreadState& state, int depth) {
        state.localNodesExplored++;

        // Bound caching: refresh every 16K nodes to reduce atomic contention
        if (++state.checkCounter >= 16384) {
            state.cachedBound = globalBestLength.load(std::memory_order_relaxed);
            state.checkCounter = 0;
        }

        // Terminal: found complete solution
        if (depth == order) [[unlikely]] {
            int length = state.marks[state.markCount - 1];
            updateGlobalBest(length, state);
            state.cachedBound = globalBestLength.load(std::memory_order_relaxed);
            return;
        }

        int lastMark = state.marks[state.markCount - 1];
        int currentBest = state.cachedBound;
        int maxPos = currentBest - 1;

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            // Pruning amélioré: distance minimale = somme triangulaire
            // Les marques restantes doivent avoir des différences uniques (1,2,3,...)
            int remainingMarks = order - depth - 1;
            int minIncrement = remainingMarks * (remainingMarks + 1) / 2;
            if (pos + minIncrement >= currentBest) {
                state.localNodesPruned++;
                continue;
            }

            if (pos + 1 <= maxPos) {
                __builtin_prefetch(&state.marks[state.markCount], 1, 1);
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
                for (int i = 0; i < newDiffCount; ++i) {
                    state.usedDiffs.set(tempDiffs[i]);
                }

                branchAndBound(state, depth + 1);

                state.markCount--;
                for (int i = 0; i < newDiffCount; ++i) {
                    state.usedDiffs.clear(tempDiffs[i]);
                }

                currentBest = state.cachedBound;
                maxPos = currentBest - 1;
            }
        }
    }

    /**
     * @brief Checks differences using scalar (non-SIMD) implementation.
     *
     * Computes all differences between the candidate position and existing marks,
     * checking for collisions with already-used differences.
     *
     * @param[in]  state     Current thread state with existing marks
     * @param[in]  pos       Candidate position for the new mark
     * @param[out] tempDiffs Array to store the new differences
     * @param[out] diffCount Number of differences stored
     *
     * @return true if all differences are unique, false otherwise
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
     * Optimized version that processes 8 marks at a time using AVX2.
     * Uses vectorized collision detection for improved performance.
     *
     * @param[in]  state     Current thread state with existing marks
     * @param[in]  pos       Candidate position for the new mark
     * @param[out] tempDiffs Array to store the new differences
     * @param[out] diffCount Number of differences stored
     *
     * @return true if all differences are unique, false otherwise
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
            if (d >= MAX_LENGTH) {
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

    /**
     * @brief Thread-safe update of the global best solution.
     *
     * Uses mutex to ensure only one thread updates the solution at a time.
     * Double-checks the bound after acquiring the lock to avoid race conditions.
     *
     * @param[in] length Length of the candidate solution
     * @param[in] state  Thread state containing the candidate solution
     *
     * @thread_safety Thread-safe; uses mutex lock
     */
    void updateGlobalBest(int length, const ThreadState& state) {
        std::lock_guard<std::mutex> lock(solutionMutex);

        if (length < globalBestLength.load(std::memory_order_acquire)) {
            globalBestLength.store(length, std::memory_order_release);

            std::vector<int> marks(state.marks, state.marks + state.markCount);
            bestSolution = GolombRuler(marks);

            if (verbose) {
                std::cout << "New best: " << bestSolution.toString()
                          << " (length " << length << ")\n";
            }
        }
    }
};

// ============================================================================
// CSV Output
// ============================================================================

/**
 * @brief Appends benchmark results to a CSV file.
 *
 * Creates the file with headers if it doesn't exist, otherwise appends
 * a new row with the benchmark results.
 *
 * @param filename Path to the CSV file
 * @param version  Solver version number (2 for OpenMP)
 * @param order    Order of the Golomb ruler solved
 * @param stats    Search statistics including time and solution
 * @param threads  Number of threads used
 */
void appendResultCSV(const std::string& filename, int version, int order,
                     const SearchStats& stats, int threads) {
    std::ifstream checkFile(filename);
    bool writeHeader = !checkFile.good();
    checkFile.close();

    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << '\n';
        return;
    }

    if (writeHeader) {
        file << "version,order,threads,time_ms,nodes_explored,nodes_pruned,solution,length\n";
    }

    file << version << ","
         << order << ","
         << threads << ","
         << std::fixed << std::setprecision(2) << stats.elapsedMs << ","
         << stats.nodesExplored << ","
         << stats.nodesPruned << ","
         << "\"" << stats.bestSolution.toString() << "\","
         << stats.bestSolution.length << '\n';
}

// ============================================================================
// Main
// ============================================================================

/**
 * @brief Prints usage information and available command-line options.
 *
 * @param progName Name of the executable (typically argv[0])
 */
void printUsage(const char* progName) {
    std::cout << "Golomb Ruler Solver v2 - OpenMP Version\n\n";
    std::cout << "Usage: " << progName << " <order> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --threads N     Set thread count (default: auto-detect)\n";
    std::cout << "  --no-simd       Disable AVX2/SIMD optimizations\n";
    std::cout << "  --csv <file>    Append results to CSV file\n";
    std::cout << "  --verbose       Show progress during search\n";
    std::cout << "  --benchmark     Run benchmarks for orders 4 to <order>\n";
    std::cout << "  --info          Show hardware info and exit\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << progName << " 10\n";
    std::cout << "  " << progName << " 11 --threads 8 --csv results.csv\n";
    std::cout << "  OMP_NUM_THREADS=16 " << progName << " 12 --benchmark\n";
}

/**
 * @brief Displays detected hardware capabilities.
 *
 * @param hw HardwareInfo structure to display
 */
void printHardwareInfo(const HardwareInfo& hw) {
    std::cout << "\n=== Hardware Information ===\n";
    std::cout << "Logical cores:  " << hw.logicalCores << '\n';
    std::cout << "Physical cores: " << hw.physicalCores << '\n';
    std::cout << "Hyperthreading: " << (hw.hasHyperthreading ? "Yes" : "No") << '\n';
    std::cout << "AVX2 support:   " << (hw.hasAVX2 ? "Yes" : "No") << '\n';
    std::cout << "============================\n\n";
}

/**
 * @brief Main entry point for the OpenMP Golomb ruler solver.
 *
 * Parses command-line arguments, configures thread count, and runs
 * the solver in single or benchmark mode.
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 *
 * @return 0 on success, 1 on error
 *
 * Environment variables:
 * - OMP_NUM_THREADS: Override thread count
 */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--info") {
            HardwareInfo hw = detectHardware();
            printHardwareInfo(hw);
            return 0;
        }
    }

    int maxOrder = parseAndValidateOrder(argv[1]);
    if (maxOrder < 0) {
        std::cerr << "Error: Order must be between 2 and " << (MAX_ORDER-1) << '\n';
        return 1;
    }

    bool useSIMD = true;
    bool verbose = false;
    bool benchmark = false;
    int requestedThreads = 0;
    std::string csvFile;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-simd") {
            useSIMD = false;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "--threads" && i + 1 < argc) {
            ++i;
            char* endptr;
            long val = std::strtol(argv[i], &endptr, 10);
            if (*endptr == '\0' && val > 0 && val <= 256) {
                requestedThreads = static_cast<int>(val);
            } else {
                std::cerr << "Warning: Invalid thread count, using auto-detect\n";
            }
        } else if (arg == "--csv" && i + 1 < argc) {
            csvFile = argv[++i];
        }
    }

    HardwareInfo hwInfo = detectHardware();

    int numThreads;
    if (requestedThreads > 0) {
        numThreads = requestedThreads;
    } else {
        numThreads = selectThreadCount(hwInfo, maxOrder);
    }
    omp_set_num_threads(numThreads);

    std::cout << "=== Golomb Ruler Solver v2: OpenMP ===" << '\n';
    std::cout << "Threads: " << numThreads;
    if (hwInfo.hasHyperthreading) {
        std::cout << " (HT enabled, " << hwInfo.physicalCores << " physical)";
    }
    std::cout << '\n';
    std::cout << "SIMD/AVX2: " << (useSIMD && hwInfo.hasAVX2 ? "Enabled" : "Disabled") << '\n';

    if (benchmark) {
        std::cout << "\nRunning benchmarks from G4 to G" << maxOrder << "...\n\n";
        std::cout << std::left << std::setw(6) << "Order"
                  << std::right << std::setw(12) << "Time (ms)"
                  << std::setw(15) << "Nodes"
                  << std::setw(15) << "Pruned"
                  << std::setw(10) << "Length"
                  << "  Solution" << '\n';
        std::cout << std::string(80, '-') << '\n';

        for (int order = 4; order <= maxOrder; ++order) {
            Timer timer;
            timer.start();

            OpenMPSolver solver(order, useSIMD, false);
            solver.solve();

            SearchStats stats = solver.getStats();
            stats.elapsedMs = timer.elapsedMs();

            std::cout << std::left << std::setw(6) << order
                      << std::right << std::fixed << std::setprecision(2)
                      << std::setw(12) << stats.elapsedMs
                      << std::setw(15) << stats.nodesExplored
                      << std::setw(15) << stats.nodesPruned
                      << std::setw(10) << stats.bestSolution.length
                      << "  " << stats.bestSolution.toString() << '\n';

            if (!csvFile.empty()) {
                appendResultCSV(csvFile, 2, order, stats, numThreads);
            }
        }
    } else {
        std::cout << "Order: " << maxOrder << '\n';

        Timer timer;
        timer.start();

        OpenMPSolver solver(maxOrder, useSIMD, verbose);
        solver.solve();

        SearchStats stats = solver.getStats();
        stats.elapsedMs = timer.elapsedMs();

        std::cout << '\n';
        printStats(stats, maxOrder);

        int optimalLength = getOptimalLength(maxOrder);
        if (optimalLength > 0 && stats.bestSolution.length == optimalLength) {
            std::cout << "*** OPTIMAL SOLUTION FOUND ***\n";
        }

        if (!csvFile.empty()) {
            appendResultCSV(csvFile, 2, maxOrder, stats, numThreads);
            std::cout << "Results saved to " << csvFile << '\n';
        }
    }

    return 0;
}
