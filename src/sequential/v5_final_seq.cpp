/**
 * Golomb Ruler Solver - Version 5: Final Sequential
 *
 * This is the production-ready sequential solver combining all optimizations:
 * - Bitset for O(1) difference lookup
 * - Symmetry breaking to halve search space
 * - Branch and bound with greedy initial solution
 * - Stack allocation for hot paths
 * - CSV output support
 * - Benchmarking capabilities
 *
 * This version serves as the reference for parallel speedup calculations.
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

// Forward declarations
void printStats(const SearchStats& stats, int order);

class FinalSequentialSolver {
private:
    int order;
    int marks[MAX_ORDER];
    int markCount;
    std::bitset<MAX_LENGTH> usedDiffs;
    GolombRuler bestSolution;
    uint64_t nodesExplored;
    uint64_t nodesPruned;
    bool useSymmetry;
    bool verbose;

public:
    FinalSequentialSolver(int n, bool symmetry = true, bool verb = false)
        : order(n), markCount(1), nodesExplored(0), nodesPruned(0),
          useSymmetry(symmetry), verbose(verb) {
        marks[0] = 0;
        usedDiffs.reset();
        findGreedySolution();
    }

    void solve() {
        branchAndBound(1);
    }

    SearchStats getStats() const {
        SearchStats stats;
        stats.nodesExplored = nodesExplored;
        stats.nodesPruned = nodesPruned;
        stats.bestSolution = bestSolution;
        return stats;
    }

private:
    // Greedy heuristic using shared template function
    void findGreedySolution() {
        std::bitset<MAX_LENGTH> greedyDiffs;
        std::vector<int> greedyMarks = computeGreedySolution(order, greedyDiffs);

        bestSolution = GolombRuler(greedyMarks);
        if (verbose) {
            std::cout << "Initial bound: " << bestSolution.length << '\n';
        }
    }

    void branchAndBound(int depth) {
        nodesExplored++;

        if (depth == order) {
            int length = marks[markCount - 1];
            if (length < bestSolution.length) {
                std::vector<int> v(marks, marks + markCount);
                bestSolution = GolombRuler(v);
                if (verbose) {
                    std::cout << "New best: " << bestSolution.toString()
                              << " (length " << length << ")" << '\n';
                }
            }
            return;
        }

        int lastMark = marks[markCount - 1];
        int maxPos = bestSolution.length - 1;
        int startPos = lastMark + 1;

        if (useSymmetry && depth == 1) {
            maxPos = std::min(maxPos, bestSolution.length / 2);
        }

        for (int pos = startPos; pos <= maxPos; ++pos) {
            int remainingMarks = order - depth - 1;
            if (pos + remainingMarks >= bestSolution.length) {
                nodesPruned++;
                continue;
            }

            int newDiffs[MAX_ORDER];
            int newDiffCount = 0;
            bool valid = true;

            for (int i = 0; i < markCount; ++i) {
                int diff = pos - marks[i];
                if (diff >= MAX_LENGTH || usedDiffs.test(diff)) {
                    valid = false;
                    break;
                }
                newDiffs[newDiffCount++] = diff;
            }

            if (valid) {
                marks[markCount++] = pos;
                for (int i = 0; i < newDiffCount; ++i) {
                    usedDiffs.set(newDiffs[i]);
                }

                branchAndBound(depth + 1);

                markCount--;
                for (int i = 0; i < newDiffCount; ++i) {
                    usedDiffs.reset(newDiffs[i]);
                }
            }
        }
    }
};

// Write single result to CSV
void appendResultCSV(const std::string& filename, int version, int order,
                     const SearchStats& stats) {
    std::ifstream checkFile(filename);
    bool writeHeader = !checkFile.good();
    checkFile.close();

    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << '\n';
        return;
    }

    if (writeHeader) {
        file << "version,order,time_ms,nodes_explored,nodes_pruned,solution,length" << '\n';
        if (!file.good()) {
            std::cerr << "Error: Failed to write CSV header to " << filename << '\n';
            return;
        }
    }

    file << version << ","
         << order << ","
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

void printUsage(const char* progName) {
    std::cout << "Golomb Ruler Solver v5 - Final Sequential Version\n\n";
    std::cout << "Usage: " << progName << " <order> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --csv <file>    Append results to CSV file\n";
    std::cout << "  --no-symmetry   Disable symmetry breaking\n";
    std::cout << "  --verbose       Show progress during search\n";
    std::cout << "  --benchmark     Run benchmarks for orders 4 to <order>\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << progName << " 9\n";
    std::cout << "  " << progName << " 10 --csv results.csv\n";
    std::cout << "  " << progName << " 8 --benchmark --csv results.csv\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    int maxOrder = parseAndValidateOrder(argv[1]);
    if (maxOrder < 0) {
        std::cerr << "Error: Invalid order. Must be a number between 2 and " << (MAX_ORDER-1) << '\n';
        return 1;
    }

    // Parse options
    bool useSymmetry = true;
    bool verbose = false;
    bool benchmark = false;
    std::string csvFile;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-symmetry") {
            useSymmetry = false;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "--csv" && i + 1 < argc) {
            csvFile = argv[++i];
        }
    }

    std::cout << "=== Golomb Ruler Solver v5: Final Sequential ===" << '\n';

    if (benchmark) {
        // Run benchmarks from order 4 to maxOrder
        std::cout << "\nRunning benchmarks from G4 to G" << maxOrder << "...\n" << '\n';
        std::cout << std::left << std::setw(6) << "Order"
                  << std::right << std::setw(12) << "Time (ms)"
                  << std::setw(15) << "Nodes"
                  << std::setw(15) << "Pruned"
                  << std::setw(10) << "Length"
                  << "  Solution" << '\n';
        std::cout << std::string(75, '-') << '\n';

        for (int order = 4; order <= maxOrder; ++order) {
            Timer timer;
            timer.start();

            FinalSequentialSolver solver(order, useSymmetry, false);
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
                appendResultCSV(csvFile, 5, order, stats);
            }
        }
    } else {
        // Single order solve
        std::cout << "Order: " << maxOrder;
        if (useSymmetry) std::cout << " (symmetry enabled)";
        std::cout << '\n';

        Timer timer;
        timer.start();

        FinalSequentialSolver solver(maxOrder, useSymmetry, verbose);
        solver.solve();

        SearchStats stats = solver.getStats();
        stats.elapsedMs = timer.elapsedMs();

        std::cout << '\n';
        printStats(stats, maxOrder);

        if (maxOrder <= 14 && stats.bestSolution.length == OPTIMAL_LENGTHS[maxOrder]) {
            std::cout << "*** OPTIMAL SOLUTION FOUND ***" << '\n';
        }

        if (!csvFile.empty()) {
            appendResultCSV(csvFile, 5, maxOrder, stats);
            std::cout << "Results saved to " << csvFile << '\n';
        }
    }

    return 0;
}
