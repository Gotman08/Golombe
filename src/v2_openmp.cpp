/**
 * Golomb Ruler Solver - Version 2: OpenMP
 *
 * OpenMP parallelized implementation with optimizations:
 * - Task-based parallelism at shallow tree depths
 * - Bound caching to reduce atomic contention (16K interval)
 * - BitSet256 for O(1) difference lookup
 * - AVX2 SIMD for vectorized difference checking
 * - Cache-aligned data structures
 *
 * Performance: G12 in ~500ms with 32 threads on Romeo
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

#include "common/golomb.hpp"
#include "common/greedy.hpp"
#include "common/bitset256.hpp"
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

struct HardwareInfo {
    int logicalCores;
    int physicalCores;
    bool hasHyperthreading;
    bool hasAVX2;
};

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

#ifdef USE_AVX2
    info.hasAVX2 = true;
#else
    info.hasAVX2 = false;
#endif

    return info;
}

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

struct alignas(64) ThreadState {
    int marks[MAX_ORDER];
    BitSet256 usedDiffs;
    int markCount;
    int localNodesExplored;
    int localNodesPruned;
    char padding[64 - (sizeof(int) * 2) % 64];
};

// ============================================================================
// OpenMP Solver
// ============================================================================

class OpenMPSolver {
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
    OpenMPSolver(int n, bool simd = true, bool verb = false)
        : order(n), globalBestLength(INT_MAX),
          totalNodesExplored(0), totalNodesPruned(0),
          useSIMD(simd), verbose(verb) {

        hwInfo = detectHardware();

        // Adaptive cutoff depth
        if (order <= 7) {
            cutoffDepth = 1;
        } else if (order <= 9) {
            cutoffDepth = 2;
        } else {
            cutoffDepth = 3;
        }

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
            int remainingMarks = order - 3;
            if (pos2 + remainingMarks >= currentBest) {
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

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            int remainingMarks = order - depth - 1;
            if (pos + remainingMarks >= currentBest) {
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

void printHardwareInfo(const HardwareInfo& hw) {
    std::cout << "\n=== Hardware Information ===\n";
    std::cout << "Logical cores:  " << hw.logicalCores << '\n';
    std::cout << "Physical cores: " << hw.physicalCores << '\n';
    std::cout << "Hyperthreading: " << (hw.hasHyperthreading ? "Yes" : "No") << '\n';
    std::cout << "AVX2 support:   " << (hw.hasAVX2 ? "Yes" : "No") << '\n';
    std::cout << "============================\n\n";
}

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
