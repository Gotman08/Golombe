/**
 * Golomb Ruler Solver - Version 6: Hardware Optimized
 *
 * Advanced hardware optimizations:
 * - OpenMP task-based parallelism at shallow tree depths
 * - AVX2 SIMD for vectorized difference computation
 * - Cache-aligned data structures (64-byte alignment)
 * - Atomic global bound with relaxed memory ordering
 * - Hardware detection for optimal thread count
 * - Explicit prefetching hints
 * - Hyperthreading-aware thread management
 *
 * Usage:
 *   ./golomb_v6 <order> [options]
 *   Options:
 *     --threads N     Set thread count (default: auto-detect)
 *     --no-simd       Disable AVX2 optimizations
 *     --csv FILE      Save results to CSV
 *     --benchmark     Run benchmarks for orders 4 to <order>
 *     --verbose       Show progress
 */

#include "golomb.hpp"
#include "greedy.hpp"
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
#include <cassert>

#include <omp.h>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

// ============================================================================
// Hardware Detection
// ============================================================================

struct HardwareInfo {
    int logicalCores;
    int physicalCores;
    bool hasHyperthreading;
    bool hasAVX2;
};

HardwareInfo detectHardware() {
    HardwareInfo info;

    // Get logical core count
    info.logicalCores = std::thread::hardware_concurrency();
    if (info.logicalCores == 0) {
        info.logicalCores = 1;  // Fallback
    }

    // Detect physical cores (Linux-specific)
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
                        // Ignore malformed lines in /proc/cpuinfo
                    }
                }
            }
        }

        if (!uniqueCores.empty()) {
            // Multiply by number of sockets (assume 1 for desktop)
            info.physicalCores = static_cast<int>(uniqueCores.size());
        }
        cpuinfo.close();
    }
#endif

    info.hasHyperthreading = (info.logicalCores > info.physicalCores);

    // Detect AVX2 support
#ifdef USE_AVX2
    info.hasAVX2 = true;
#else
    info.hasAVX2 = false;
#endif

    return info;
}

int selectThreadCount(const HardwareInfo& hw, int order) {
    // For small problems, use fewer threads to avoid overhead
    if (order <= 7) {
        return std::min(hw.physicalCores, 4);
    }

    // For medium problems, use physical cores
    if (order <= 9) {
        return hw.physicalCores;
    }

    // For large problems, use all cores including hyperthreads
    return hw.logicalCores;
}

// ============================================================================
// Custom 256-bit Bitset (Cache-aligned, AVX2-friendly)
// ============================================================================

struct alignas(32) BitSet256 {
    uint64_t words[4];  // 256 bits = 4 x 64-bit words

    inline void reset() {
        words[0] = words[1] = words[2] = words[3] = 0;
    }

    inline bool test(int bit) const {
        assert(bit >= 0 && bit < 256 && "BitSet256::test out of bounds");
        if (bit < 0 || bit >= 256) return false;  // Safe bounds check in release
        return (words[bit >> 6] >> (bit & 63)) & 1;
    }

    inline void set(int bit) {
        assert(bit >= 0 && bit < 256 && "BitSet256::set out of bounds");
        if (bit < 0 || bit >= 256) return;  // Safe bounds check in release
        words[bit >> 6] |= (1ULL << (bit & 63));
    }

    inline void clear(int bit) {
        assert(bit >= 0 && bit < 256 && "BitSet256::clear out of bounds");
        if (bit < 0 || bit >= 256) return;  // Safe bounds check in release
        words[bit >> 6] &= ~(1ULL << (bit & 63));
    }

    // Copy from another BitSet256
    inline void copyFrom(const BitSet256& other) {
        words[0] = other.words[0];
        words[1] = other.words[1];
        words[2] = other.words[2];
        words[3] = other.words[3];
    }

#ifdef USE_AVX2
    /**
     * AVX2-accelerated collision detection
     * Checks if ANY of the bits in 'mask' are already set in this bitset
     * Uses _mm256_and_si256 to check 256 bits in a single operation
     * Returns true if collision detected (at least one bit is set in both)
     */
    inline bool hasCollisionAVX2(const BitSet256& mask) const {
        __m256i used = _mm256_load_si256((__m256i*)words);
        __m256i check = _mm256_load_si256((__m256i*)mask.words);
        __m256i collision = _mm256_and_si256(used, check);

        // _mm256_testz returns 1 if all bits are zero, 0 otherwise
        return !_mm256_testz_si256(collision, collision);
    }
#endif
};

// ============================================================================
// Thread-Local State (Cache-aligned)
// ============================================================================

struct alignas(64) ThreadState {
    int marks[MAX_ORDER];           // Mark positions
    BitSet256 usedDiffs;            // Difference bitset
    int markCount;                  // Current number of marks
    // NOTE: tempDiffs removed - must be local to avoid recursion overwrite!
    int localNodesExplored;         // Per-thread counter
    int localNodesPruned;           // Per-thread counter
    char padding[64 - (sizeof(int) * 2) % 64];  // Align to cache line
};

// ============================================================================
// Hardware Optimized Solver
// ============================================================================

class HardwareOptimizedSolver {
private:
    int order;
    std::atomic<int> globalBestLength;
    GolombRuler bestSolution;
    std::mutex solutionMutex;

    std::atomic<uint64_t> totalNodesExplored;
    std::atomic<uint64_t> totalNodesPruned;

    int cutoffDepth;
    bool useSIMD;
    bool verbose;
    HardwareInfo hwInfo;

public:
    HardwareOptimizedSolver(int n, bool simd = true, bool verb = false)
        : order(n), globalBestLength(INT_MAX),
          totalNodesExplored(0), totalNodesPruned(0),
          useSIMD(simd), verbose(verb) {

        hwInfo = detectHardware();

        // Adaptive cutoff depth based on order
        if (order <= 7) {
            cutoffDepth = 1;
        } else if (order <= 9) {
            cutoffDepth = 2;
        } else {
            cutoffDepth = 3;
        }

        // Find greedy initial solution
        findGreedySolution();
    }

    void solve() {
        #pragma omp parallel
        {
            #pragma omp single
            {
                ThreadState initialState;
                initializeState(initialState);

                int maxFirstMark = globalBestLength.load(std::memory_order_acquire) / 2;

                // Generate tasks for depth 1
                for (int pos1 = 1; pos1 <= maxFirstMark; ++pos1) {
                    #pragma omp task firstprivate(pos1)
                    {
                        ThreadState state;
                        initializeState(state);

                        // Place first mark
                        state.marks[state.markCount++] = pos1;
                        state.usedDiffs.set(pos1);  // diff from 0

                        if (cutoffDepth >= 2 && order >= 10) {
                            // Generate second level tasks for large problems
                            generateDepth2Tasks(state, pos1);
                        } else {
                            // Direct search
                            branchAndBound(state, 2);
                        }

                        // Aggregate counters
                        totalNodesExplored.fetch_add(state.localNodesExplored,
                                                     std::memory_order_relaxed);
                        totalNodesPruned.fetch_add(state.localNodesPruned,
                                                   std::memory_order_relaxed);
                    }
                }
            }
        }
    }

    SearchStats getStats() const {
        SearchStats stats;
        stats.nodesExplored = totalNodesExplored.load();
        stats.nodesPruned = totalNodesPruned.load();
        stats.bestSolution = bestSolution;
        return stats;
    }

    const HardwareInfo& getHardwareInfo() const {
        return hwInfo;
    }

private:
    void initializeState(ThreadState& state) {
        state.marks[0] = 0;
        state.markCount = 1;
        state.usedDiffs.reset();
        state.localNodesExplored = 0;
        state.localNodesPruned = 0;
    }

    // Greedy heuristic using shared template function with BitSet256
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

    void generateDepth2Tasks(ThreadState& parentState, int pos1) {
        int currentBest = globalBestLength.load(std::memory_order_acquire);
        int maxPos = currentBest - 1;

        for (int pos2 = pos1 + 1; pos2 <= maxPos; ++pos2) {
            // Pruning check
            int remainingMarks = order - 3;  // depth 2, placing 3rd mark
            if (pos2 + remainingMarks >= currentBest) {
                parentState.localNodesPruned++;
                continue;
            }

            // Check differences
            int diff1 = pos2 - parentState.marks[0];  // diff from 0
            int diff2 = pos2 - parentState.marks[1];  // diff from pos1

            if (diff1 >= MAX_LENGTH || diff2 >= MAX_LENGTH) continue;
            if (parentState.usedDiffs.test(diff1) || parentState.usedDiffs.test(diff2)) continue;
            if (diff1 == diff2) continue;

            #pragma omp task firstprivate(pos2)
            {
                ThreadState state;
                // Copy parent state
                std::memcpy(state.marks, parentState.marks,
                           parentState.markCount * sizeof(int));
                state.markCount = parentState.markCount;
                state.usedDiffs.copyFrom(parentState.usedDiffs);
                state.localNodesExplored = 0;
                state.localNodesPruned = 0;

                // Place mark
                state.marks[state.markCount++] = pos2;
                state.usedDiffs.set(diff1);
                state.usedDiffs.set(diff2);

                branchAndBound(state, 3);

                // Aggregate counters
                totalNodesExplored.fetch_add(state.localNodesExplored,
                                            std::memory_order_relaxed);
                totalNodesPruned.fetch_add(state.localNodesPruned,
                                           std::memory_order_relaxed);
            }
        }
    }

    void branchAndBound(ThreadState& state, int depth) {
        state.localNodesExplored++;

        // Bound caching: refresh every 16K nodes to reduce atomic contention
        static thread_local int cachedBound = INT_MAX;
        static thread_local int checkCounter = 0;
        if (++checkCounter >= 16384) {
            cachedBound = globalBestLength.load(std::memory_order_relaxed);
            checkCounter = 0;
        }

        // Terminal: found complete solution
        if (depth == order) [[unlikely]] {
            int length = state.marks[state.markCount - 1];
            updateGlobalBest(length, state);
            cachedBound = globalBestLength.load(std::memory_order_relaxed);
            return;
        }

        int lastMark = state.marks[state.markCount - 1];
        int currentBest = cachedBound;
        int maxPos = currentBest - 1;

        // Position iteration with prefetching
        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            // Early pruning
            int remainingMarks = order - depth - 1;
            if (pos + remainingMarks >= currentBest) {
                state.localNodesPruned++;
                continue;
            }

            // Prefetch for next iteration
            if (pos + 1 <= maxPos) {
                __builtin_prefetch(&state.marks[state.markCount], 1, 1);
            }

            // Check all differences - tempDiffs MUST be local to avoid recursion overwrite!
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
                // Place mark
                state.marks[state.markCount++] = pos;
                for (int i = 0; i < newDiffCount; ++i) {
                    state.usedDiffs.set(tempDiffs[i]);
                }

                // Recurse
                branchAndBound(state, depth + 1);

                // Backtrack
                state.markCount--;
                for (int i = 0; i < newDiffCount; ++i) {
                    state.usedDiffs.clear(tempDiffs[i]);
                }

                // Use cached bound (refreshed via counter mechanism)
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
    /**
     * AVX2-optimized difference checking with vectorized collision detection
     *
     * Optimization strategy:
     * 1. Calculate all differences using SIMD (8 at a time)
     * 2. Build a mask bitset with all differences
     * 3. Use _mm256_and_si256 to check all 256 bits in ONE operation
     * 4. Only if no collision, copy differences to output array
     *
     * This reduces branch mispredictions and leverages full SIMD width
     */
    inline bool checkDifferencesAVX2(ThreadState& state, int pos, int* tempDiffs, int& diffCount) {
        __m256i vpos = _mm256_set1_epi32(pos);

        // Phase 1: Calculate all differences and store them
        alignas(32) int allDiffs[MAX_ORDER];
        int totalDiffs = 0;
        int i = 0;

        // Process 8 marks at a time with AVX2
        for (; i + 8 <= state.markCount; i += 8) {
            __m256i vmarks = _mm256_loadu_si256((__m256i*)&state.marks[i]);
            __m256i vdiffs = _mm256_sub_epi32(vpos, vmarks);
            _mm256_storeu_si256((__m256i*)&allDiffs[totalDiffs], vdiffs);
            totalDiffs += 8;
        }

        // Handle remaining marks (scalar)
        for (; i < state.markCount; ++i) {
            allDiffs[totalDiffs++] = pos - state.marks[i];
        }

        // Phase 2: Build collision mask and check bounds
        BitSet256 checkMask;
        checkMask.reset();

        for (int j = 0; j < totalDiffs; ++j) {
            int d = allDiffs[j];
            if (d >= MAX_LENGTH) {
                return false;  // Out of bounds
            }
            checkMask.set(d);
        }

        // Phase 3: Vectorized collision detection (256 bits in ONE operation!)
        if (state.usedDiffs.hasCollisionAVX2(checkMask)) {
            return false;  // At least one difference already used
        }

        // Phase 4: No collision - copy differences to output
        diffCount = totalDiffs;
        for (int j = 0; j < totalDiffs; ++j) {
            tempDiffs[j] = allDiffs[j];
        }

        return true;
    }
#endif

    void updateGlobalBest(int length, const ThreadState& state) {
        // Use mutex to protect both atomic update and solution storage atomically
        // This prevents race condition where another thread could read stale solution
        std::lock_guard<std::mutex> lock(solutionMutex);

        // Check and update under mutex protection
        if (length < globalBestLength.load(std::memory_order_acquire)) {
            globalBestLength.store(length, std::memory_order_release);

            std::vector<int> marks(state.marks,
                                  state.marks + state.markCount);
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
        file << "version,order,threads,time_ms,nodes_explored,nodes_pruned,solution,length"
             << '\n';
        if (!file.good()) {
            std::cerr << "Error: Failed to write CSV header to " << filename << '\n';
            return;
        }
    }

    file << version << ","
         << order << ","
         << threads << ","
         << std::fixed << std::setprecision(2) << stats.elapsedMs << ","
         << stats.nodesExplored << ","
         << stats.nodesPruned << ","
         << "\"" << stats.bestSolution.toString() << "\","
         << stats.bestSolution.length << '\n';

    if (!file.good()) {
        std::cerr << "Error: Failed to write CSV data to " << filename << '\n';
    }

    file.close();
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* progName) {
    std::cout << "Golomb Ruler Solver v6 - Hardware Optimized Version\n\n";
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

void printHardwareInfo(const HardwareInfo& hw) {
    std::cout << "\n=== Hardware Information ===" << '\n';
    std::cout << "Logical cores:  " << hw.logicalCores << '\n';
    std::cout << "Physical cores: " << hw.physicalCores << '\n';
    std::cout << "Hyperthreading: " << (hw.hasHyperthreading ? "Yes" : "No") << '\n';
    std::cout << "AVX2 support:   " << (hw.hasAVX2 ? "Yes" : "No") << '\n';
    std::cout << "============================\n" << '\n';
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    // Check for --info flag first
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

    // Parse options
    bool useSIMD = true;
    bool verbose = false;
    bool benchmark = false;
    int requestedThreads = 0;  // 0 = auto-detect
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

    // Detect hardware
    HardwareInfo hwInfo = detectHardware();

    // Set thread count
    int numThreads;
    if (requestedThreads > 0) {
        numThreads = requestedThreads;
    } else {
        numThreads = selectThreadCount(hwInfo, maxOrder);
    }
    omp_set_num_threads(numThreads);

    std::cout << "=== Golomb Ruler Solver v6: Hardware Optimized ===" << '\n';
    std::cout << "Threads: " << numThreads;
    if (hwInfo.hasHyperthreading) {
        std::cout << " (HT enabled, " << hwInfo.physicalCores << " physical)";
    }
    std::cout << '\n';
    std::cout << "SIMD/AVX2: " << (useSIMD && hwInfo.hasAVX2 ? "Enabled" : "Disabled") << '\n';

    if (benchmark) {
        // Run benchmarks from order 4 to maxOrder
        std::cout << "\nRunning benchmarks from G4 to G" << maxOrder << "...\n" << '\n';
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

            HardwareOptimizedSolver solver(order, useSIMD, false);
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
                appendResultCSV(csvFile, 6, order, stats, numThreads);
            }
        }
    } else {
        // Single order solve
        std::cout << "Order: " << maxOrder << '\n';

        Timer timer;
        timer.start();

        HardwareOptimizedSolver solver(maxOrder, useSIMD, verbose);
        solver.solve();

        SearchStats stats = solver.getStats();
        stats.elapsedMs = timer.elapsedMs();

        std::cout << '\n';
        printStats(stats, maxOrder);

        int optimalLength = getOptimalLength(maxOrder);
        if (optimalLength > 0 && stats.bestSolution.length == optimalLength) {
            std::cout << "*** OPTIMAL SOLUTION FOUND ***" << '\n';
        }

        if (!csvFile.empty()) {
            appendResultCSV(csvFile, 6, maxOrder, stats, numThreads);
            std::cout << "Results saved to " << csvFile << '\n';
        }
    }

    return 0;
}
