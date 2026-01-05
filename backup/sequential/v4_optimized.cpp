/**
 * Golomb Ruler Solver - Version 4: Optimized
 *
 * Algorithm: Branch and Bound with advanced optimizations.
 *
 * Optimizations:
 * 1. Bitset for differences: O(1) lookup instead of O(log n) with std::set
 * 2. Symmetry breaking: Only explore rulers where marks[1] <= best/2
 *    (exploits mirror symmetry of Golomb rulers)
 * 3. Improved lower bound estimation
 * 4. Stack-allocated difference tracking (no heap allocations in hot path)
 *
 * Expected to work well for G4-G10, possibly G11
 */

#include "golomb.hpp"
#include "greedy.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cstring>

// Forward declarations
void printStats(const SearchStats& stats, int order);

class OptimizedSolver {
private:
    int order;
    int marks[MAX_ORDER];       // Stack-allocated marks array
    int markCount;
    std::bitset<MAX_LENGTH> usedDiffs;
    GolombRuler bestSolution;
    uint64_t nodesExplored;
    uint64_t nodesPruned;
    bool useSymmetry;

public:
    OptimizedSolver(int n, bool symmetry = true)
        : order(n), markCount(1), nodesExplored(0), nodesPruned(0), useSymmetry(symmetry) {
        marks[0] = 0;  // First mark always at 0
        usedDiffs.reset();

        // Find initial bound using greedy heuristic
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
        std::cout << "Initial greedy solution: " << bestSolution.toString()
                  << " (length " << bestSolution.length << ")\n";
    }

    void branchAndBound(int depth) {
        nodesExplored++;

        // Base case: we have placed all marks
        if (depth == order) {
            int length = marks[markCount - 1];
            if (length < bestSolution.length) {
                std::vector<int> v(marks, marks + markCount);
                bestSolution = GolombRuler(v);
            }
            return;
        }

        // Get the last placed mark position
        int lastMark = marks[markCount - 1];

        // Calculate maximum allowed position based on current best
        int maxPos = bestSolution.length - 1;

        // For symmetry breaking: when placing second mark,
        // only consider positions up to half of the best length
        int startPos = lastMark + 1;
        if (useSymmetry && depth == 1) {
            maxPos = std::min(maxPos, bestSolution.length / 2);
        }

        // Try all valid positions for the next mark
        for (int pos = startPos; pos <= maxPos; ++pos) {

            // Quick check: can we fit remaining marks?
            int remainingMarks = order - depth - 1;
            if (pos + remainingMarks >= bestSolution.length) {
                nodesPruned++;
                continue;
            }

            // Check if this position creates any duplicate differences
            // Use stack-allocated array for new differences
            int newDiffs[MAX_ORDER];
            int newDiffCount = 0;
            bool valid = true;

            for (int i = 0; i < markCount; ++i) {
                int diff = pos - marks[i];

                // Bounds check for bitset
                if (diff >= MAX_LENGTH) {
                    valid = false;
                    break;
                }

                // Check if this difference already exists (O(1) with bitset)
                if (usedDiffs.test(diff)) {
                    valid = false;
                    break;
                }

                newDiffs[newDiffCount++] = diff;
            }

            if (valid) {
                // Add this position and its differences
                marks[markCount++] = pos;
                for (int i = 0; i < newDiffCount; ++i) {
                    usedDiffs.set(newDiffs[i]);
                }

                // Recurse
                branchAndBound(depth + 1);

                // Backtrack: remove the position and differences
                markCount--;
                for (int i = 0; i < newDiffCount; ++i) {
                    usedDiffs.reset(newDiffs[i]);
                }
            }
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <order> [--no-symmetry]" << '\n';
        std::cout << "  order: Number of marks in the Golomb ruler (4-10 recommended)" << '\n';
        std::cout << "  --no-symmetry: Disable symmetry breaking (for comparison)" << '\n';
        std::cout << '\n';
        std::cout << "Example: " << argv[0] << " 9" << '\n';
        return 1;
    }

    int order = parseAndValidateOrder(argv[1]);
    if (order < 0) {
        std::cerr << "Error: Invalid order. Must be a number between 2 and " << (MAX_ORDER-1) << '\n';
        return 1;
    }

    bool useSymmetry = true;
    if (argc >= 3 && std::string(argv[2]) == "--no-symmetry") {
        useSymmetry = false;
    }

    std::cout << "=== Golomb Ruler Solver v4: Optimized ===" << '\n';
    std::cout << "Order: " << order;
    if (useSymmetry) {
        std::cout << " (symmetry breaking enabled)";
    }
    std::cout << '\n';

    Timer timer;
    timer.start();

    OptimizedSolver solver(order, useSymmetry);
    solver.solve();

    SearchStats stats = solver.getStats();
    stats.elapsedMs = timer.elapsedMs();

    std::cout << '\n';
    printStats(stats, order);

    // Check if solution is optimal
    if (order <= 14 && stats.bestSolution.length == OPTIMAL_LENGTHS[order]) {
        std::cout << "*** OPTIMAL SOLUTION FOUND ***" << '\n';
    } else if (order <= 14) {
        std::cout << "Note: Optimal length for G" << order << " is " << OPTIMAL_LENGTHS[order] << '\n';
    }

    return 0;
}
