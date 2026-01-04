/**
 * Golomb Ruler Solver - Version 2: Backtracking
 *
 * Algorithm: Build the ruler incrementally, mark by mark.
 * Abandon a branch as soon as a duplicate difference is detected.
 *
 * Improvements over v1:
 * - No need to generate all combinations upfront
 * - Early pruning when a difference collision is found
 * - Uses std::bitset for O(1) difference lookup (instead of O(log n) with std::set)
 * - Stack-allocated arrays for hot path
 * - Symmetry breaking to halve search space
 *
 * Complexity: Still exponential, but explores far fewer nodes
 * Expected to work for G4-G9, maybe G10
 */

#include "golomb.hpp"
#include <iostream>
#include <bitset>
#include <cstdlib>
#include <string>
#include <algorithm>

// Forward declarations
void printStats(const SearchStats& stats, int order);

class BacktrackingSolver {
private:
    int order;
    int maxLength;
    int marks[MAX_ORDER];           // Stack-allocated marks array
    int markCount;
    std::bitset<MAX_LENGTH> usedDiffs;  // O(1) lookup
    GolombRuler bestSolution;
    uint64_t nodesExplored;
    bool useSymmetry;

public:
    BacktrackingSolver(int n, int maxLen, bool symmetry = true)
        : order(n), maxLength(maxLen), markCount(1), nodesExplored(0), useSymmetry(symmetry) {
        marks[0] = 0;  // First mark always at 0
        usedDiffs.reset();
        bestSolution.length = INT_MAX;
    }

    void solve() {
        backtrack(1);
    }

    SearchStats getStats() const {
        SearchStats stats;
        stats.nodesExplored = nodesExplored;
        stats.bestSolution = bestSolution;
        return stats;
    }

private:
    void backtrack(int depth) {
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

        // Symmetry breaking: when placing second mark, only consider
        // positions up to half of the best length (or maxLength/2 if no solution yet)
        int maxPos = (bestSolution.length < INT_MAX) ?
                     std::min(maxLength, bestSolution.length - 1) : maxLength;
        if (useSymmetry && depth == 1) {
            int halfBound = (bestSolution.length < INT_MAX) ?
                           bestSolution.length / 2 : maxLength / 2;
            maxPos = std::min(maxPos, halfBound);
        }

        // Try all positions for the next mark
        // Start from lastMark + 1 to maintain strictly increasing order
        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {

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
                backtrack(depth + 1);

                // Backtrack: remove the position and differences
                markCount--;
                for (int i = 0; i < newDiffCount; ++i) {
                    usedDiffs.reset(newDiffs[i]);
                }
            }
        }
    }
};

// Calculate a reasonable max length to search
int calculateMaxLength(int order) {
    // Use 2x optimal length as upper bound if known
    if (order >= 1 && order <= 14) {
        return OPTIMAL_LENGTHS[order] * 2;
    }
    // Otherwise use n^2 as a safe upper bound
    return order * order;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <order> [max_length] [--no-symmetry]\n";
        std::cout << "  order: Number of marks in the Golomb ruler (4-9 recommended)\n";
        std::cout << "  max_length: Optional maximum length to search (default: 2x optimal)\n";
        std::cout << "  --no-symmetry: Disable symmetry breaking (for comparison)\n";
        std::cout << "\nExample: " << argv[0] << " 7\n";
        return 1;
    }

    int order = parseAndValidateOrder(argv[1]);
    if (order < 0) {
        std::cerr << "Error: Invalid order. Must be a number between 2 and " << (MAX_ORDER-1) << '\n';
        return 1;
    }

    int maxLength = calculateMaxLength(order);
    bool useSymmetry = true;

    // Parse optional arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-symmetry") {
            useSymmetry = false;
        } else if (arg == "--benchmark") {
            // Ignore benchmark flag (not supported in v2)
        } else if (arg[0] != '-') {
            // Only parse as max_length if it's not a flag
            int parsedLen = parseAndValidateOrder(argv[i], MAX_LENGTH);
            if (parsedLen > 0) {
                maxLength = parsedLen;
            }
        }
    }

    std::cout << "=== Golomb Ruler Solver v2: Backtracking ===\n";
    std::cout << "Order: " << order << ", Max length: " << maxLength;
    if (useSymmetry) {
        std::cout << " (symmetry enabled)";
    }
    std::cout << "\nSearching...\n";

    Timer timer;
    timer.start();

    BacktrackingSolver solver(order, maxLength, useSymmetry);
    solver.solve();

    SearchStats stats = solver.getStats();
    stats.elapsedMs = timer.elapsedMs();

    std::cout << '\n';
    printStats(stats, order);

    // Check if solution is optimal
    if (order <= 14 && stats.bestSolution.length == OPTIMAL_LENGTHS[order]) {
        std::cout << "*** OPTIMAL SOLUTION FOUND ***\n";
    } else if (order <= 14) {
        std::cout << "Note: Optimal length for G" << order << " is " << OPTIMAL_LENGTHS[order] << '\n';
    }

    return 0;
}
