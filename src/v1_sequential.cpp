/**
 * @file v1_sequential.cpp
 * @brief Sequential Golomb Ruler Solver
 *
 * Golomb Ruler Solver - High Performance Computing Implementation
 * Copyright (c) 2025 Nicolas Marano
 * Licensed under the MIT License. See LICENSE file for details.
 *
 * Pure sequential implementation with optimizations:
 * - Branch and Bound with greedy initial bound
 * - BitSet256 for O(1) difference lookup
 * - AVX2 SIMD for vectorized difference checking (optional)
 * - Bound caching to reduce memory accesses
 *
 * Usage:
 *   ./golomb_v1 <order> [options]
 *   Options:
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
#include <climits>

// ============================================================================
// Search State
// ============================================================================

/**
 * @struct SearchState
 * @brief Holds the current state of the branch-and-bound search.
 *
 * This structure maintains all information needed to represent a partial
 * solution during the backtracking search, including mark positions,
 * used differences, and performance counters.
 */
struct SearchState {
    int marks[MAX_ORDER];           ///< Array of mark positions on the ruler (marks[0] = 0 always)
    BitSet256 usedDiffs;            ///< Bitset tracking differences already used (for O(1) collision check)
    int markCount;                  ///< Current number of marks placed on the ruler
    uint64_t nodesExplored;         ///< Counter for total nodes explored in the search tree
    uint64_t nodesPruned;           ///< Counter for nodes pruned by bound or feasibility checks
};

// ============================================================================
// Sequential Solver
// ============================================================================

/**
 * @class SequentialSolver
 * @brief Sequential branch-and-bound solver for optimal Golomb rulers.
 *
 * This class implements a single-threaded branch-and-bound algorithm to find
 * optimal Golomb rulers. It uses several optimizations:
 * - Greedy heuristic for initial upper bound
 * - BitSet256 for O(1) difference collision detection
 * - AVX2 SIMD for vectorized difference checking (optional)
 * - Symmetry breaking by limiting first mark position
 *
 * @note For multi-threaded solving, use the OpenMP version (v2).
 * @see GolombRuler, BitSet256
 */
class SequentialSolver {
private:
    int order;                      ///< Target number of marks (order of the ruler)
    int bestLength;                 ///< Current best (shortest) ruler length found
    GolombRuler bestSolution;       ///< Best solution found so far
    uint64_t totalNodesExplored;    ///< Total nodes explored across all searches
    uint64_t totalNodesPruned;      ///< Total nodes pruned across all searches
    bool useSIMD;                   ///< Whether to use AVX2 SIMD optimizations
    bool verbose;                   ///< Whether to print progress information

public:
    /**
     * @brief Constructs a new Sequential Solver.
     *
     * Initializes the solver and computes an initial upper bound using
     * a greedy heuristic algorithm.
     *
     * @param n     The order of the Golomb ruler to find (number of marks)
     * @param simd  Enable AVX2 SIMD optimizations (default: true)
     * @param verb  Enable verbose output during search (default: false)
     *
     * @pre n must be between 2 and MAX_ORDER-1
     */
    SequentialSolver(int n, bool simd = true, bool verb = false)
        : order(n), bestLength(INT_MAX),
          totalNodesExplored(0), totalNodesPruned(0),
          useSIMD(simd), verbose(verb) {
        findGreedySolution();
    }

    /**
     * @brief Executes the branch-and-bound search for an optimal Golomb ruler.
     *
     * Performs a complete search of the solution space using branch-and-bound
     * with symmetry breaking. The first mark position is limited to [1, bestLength/2]
     * to eliminate symmetric solutions.
     *
     * @post bestSolution contains the optimal ruler found
     * @post totalNodesExplored and totalNodesPruned are updated with statistics
     *
     * @complexity Exponential in the worst case, but heavily pruned in practice
     */
    void solve() {
        SearchState state;
        initializeState(state);

        int maxFirstMark = bestLength / 2;

        // Symmetry breaking: first mark <= bestLength/2
        for (int pos1 = 1; pos1 <= maxFirstMark; ++pos1) {
            state.marks[state.markCount++] = pos1;
            state.usedDiffs.set(pos1);  // diff from 0

            branchAndBound(state, 2);

            // Backtrack
            state.markCount--;
            state.usedDiffs.clear(pos1);
        }

        totalNodesExplored = state.nodesExplored;
        totalNodesPruned = state.nodesPruned;
    }

    /**
     * @brief Retrieves the search statistics after solving.
     *
     * @return SearchStats structure containing nodes explored, pruned, and best solution
     *
     * @pre solve() should have been called first for meaningful results
     */
    SearchStats getStats() const {
        SearchStats stats;
        stats.nodesExplored = totalNodesExplored;
        stats.nodesPruned = totalNodesPruned;
        stats.bestSolution = bestSolution;
        return stats;
    }

private:
    /**
     * @brief Initializes a search state for a new search.
     *
     * Sets up the initial state with mark 0 at position 0, clears the
     * difference bitset, and resets all counters.
     *
     * @param[out] state The search state to initialize
     */
    void initializeState(SearchState& state) {
        state.marks[0] = 0;
        state.markCount = 1;
        state.usedDiffs.reset();
        state.nodesExplored = 0;
        state.nodesPruned = 0;
    }

    /**
     * @brief Computes an initial upper bound using a greedy heuristic.
     *
     * Uses a greedy algorithm to quickly find a valid (but not necessarily optimal)
     * Golomb ruler. This provides an initial upper bound for pruning during
     * the branch-and-bound search.
     *
     * @post bestSolution contains the greedy solution
     * @post bestLength is set to the greedy solution's length
     *
     * @see computeGreedySolution()
     */
    void findGreedySolution() {
        BitSet256 greedyDiffs;
        greedyDiffs.reset();
        std::vector<int> greedyMarks = computeGreedySolution(order, greedyDiffs);

        bestSolution = GolombRuler(greedyMarks);
        bestLength = bestSolution.length;

        if (verbose) {
            std::cout << "Initial bound: " << bestLength << '\n';
        }
    }

    /**
     * @brief Recursive branch-and-bound search for Golomb rulers.
     *
     * Explores the search tree by trying all valid positions for the next mark.
     * Uses several pruning strategies:
     * - Bound pruning: skip positions that cannot lead to better solutions
     * - Feasibility pruning: skip positions with duplicate differences
     *
     * @param[in,out] state Current search state (modified during recursion)
     * @param[in]     depth Current depth in the search tree (= number of marks to place)
     *
     * @note Updates bestLength and bestSolution when a better solution is found
     */
    void branchAndBound(SearchState& state, int depth) {
        state.nodesExplored++;

        // Terminal: found complete solution
        if (depth == order) [[unlikely]] {
            int length = state.marks[state.markCount - 1];
            if (length < bestLength) {
                bestLength = length;
                std::vector<int> marks(state.marks, state.marks + state.markCount);
                bestSolution = GolombRuler(marks);

                if (verbose) {
                    std::cout << "New best: " << bestSolution.toString()
                              << " (length " << length << ")\n";
                }
            }
            return;
        }

        int lastMark = state.marks[state.markCount - 1];
        int currentBest = bestLength;
        int maxPos = currentBest - 1;

        // Position iteration
        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            // Early pruning: check if remaining marks can fit
            int remainingMarks = order - depth - 1;
            if (pos + remainingMarks >= currentBest) {
                state.nodesPruned++;
                continue;
            }

            // Check all differences for collisions
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

                // Update bound (may have changed during recursion)
                currentBest = bestLength;
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
     * @param[in]  state     Current search state with existing marks
     * @param[in]  pos       Candidate position for the new mark
     * @param[out] tempDiffs Array to store the new differences (must have size >= MAX_ORDER)
     * @param[out] diffCount Number of differences stored in tempDiffs
     *
     * @return true if all differences are unique (position is valid), false otherwise
     *
     * @complexity O(markCount) - linear in the number of existing marks
     */
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
    /**
     * @brief Checks differences using AVX2 SIMD vectorization.
     *
     * Optimized version that processes 8 marks at a time using AVX2 instructions.
     * Uses a 4-phase approach:
     * 1. Vectorized difference computation (8 at a time)
     * 2. Build collision mask from new differences
     * 3. Vectorized collision detection using hasCollisionAVX2()
     * 4. Copy results to output array
     *
     * @param[in]  state     Current search state with existing marks
     * @param[in]  pos       Candidate position for the new mark
     * @param[out] tempDiffs Array to store the new differences (must have size >= MAX_ORDER)
     * @param[out] diffCount Number of differences stored in tempDiffs
     *
     * @return true if all differences are unique (position is valid), false otherwise
     *
     * @note Falls back to scalar for remaining marks when markCount % 8 != 0
     * @complexity O(markCount/8) SIMD operations + O(markCount % 8) scalar operations
     */
    inline bool checkDifferencesAVX2(SearchState& state, int pos, int* tempDiffs, int& diffCount) {
        __m256i vpos = _mm256_set1_epi32(pos);

        // Phase 1: Calculate all differences
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
                return false;
            }
            checkMask.set(d);
        }

        // Phase 3: Vectorized collision detection
        if (state.usedDiffs.hasCollisionAVX2(checkMask)) {
            return false;
        }

        // Phase 4: Copy differences to output
        diffCount = totalDiffs;
        for (int j = 0; j < totalDiffs; ++j) {
            tempDiffs[j] = allDiffs[j];
        }

        return true;
    }
#endif
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
 * CSV columns: version, order, threads, time_ms, nodes_explored, nodes_pruned, solution, length
 *
 * @param filename Path to the CSV file
 * @param version  Solver version number (1 for sequential)
 * @param order    Order of the Golomb ruler solved
 * @param stats    Search statistics including time and solution
 * @param threads  Number of threads used (1 for sequential)
 *
 * @note Thread-safe for different files, but not for concurrent writes to the same file
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
    std::cout << "Golomb Ruler Solver v1 - Sequential Version\n\n";
    std::cout << "Usage: " << progName << " <order> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --no-simd       Disable AVX2/SIMD optimizations\n";
    std::cout << "  --csv <file>    Append results to CSV file\n";
    std::cout << "  --verbose       Show progress during search\n";
    std::cout << "  --benchmark     Run benchmarks for orders 4 to <order>\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << progName << " 10\n";
    std::cout << "  " << progName << " 11 --csv results.csv\n";
}

/**
 * @brief Main entry point for the sequential Golomb ruler solver.
 *
 * Parses command-line arguments and runs the solver in one of two modes:
 * - Single solve: Find optimal ruler for the specified order
 * - Benchmark: Run benchmarks for orders 4 through the specified order
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 *
 * @return 0 on success, 1 on error (invalid arguments)
 *
 * Command-line options:
 * - `<order>`: Required. The order of the Golomb ruler to find (2 to MAX_ORDER-1)
 * - `--no-simd`: Disable AVX2 SIMD optimizations
 * - `--csv <file>`: Save results to CSV file
 * - `--verbose`: Show progress during search
 * - `--benchmark`: Run benchmarks for orders 4 to <order>
 */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
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
    std::string csvFile;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-simd") {
            useSIMD = false;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "--csv" && i + 1 < argc) {
            csvFile = argv[++i];
        }
    }

    std::cout << "=== Golomb Ruler Solver v1: Sequential ===" << '\n';
#ifdef USE_AVX2
    std::cout << "SIMD/AVX2: " << (useSIMD ? "Enabled" : "Disabled") << '\n';
#else
    std::cout << "SIMD/AVX2: Not compiled\n";
#endif

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

            SequentialSolver solver(order, useSIMD, false);
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
                appendResultCSV(csvFile, 1, order, stats, 1);
            }
        }
    } else {
        std::cout << "Order: " << maxOrder << '\n';

        Timer timer;
        timer.start();

        SequentialSolver solver(maxOrder, useSIMD, verbose);
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
            appendResultCSV(csvFile, 1, maxOrder, stats, 1);
            std::cout << "Results saved to " << csvFile << '\n';
        }
    }

    return 0;
}
