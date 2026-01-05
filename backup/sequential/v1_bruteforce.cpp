/**
 * Golomb Ruler Solver - Version 1: Brute Force
 *
 * Algorithm: Enumerate all C(L_max, n-1) combinations of positions
 * for n-1 marks (first mark is always 0), and check each for validity.
 *
 * Complexity: O(C(L, n) * n^2) - exponential in L
 * Expected to work for G4, G5, timeout for G6+
 */

#include "golomb.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>

// Forward declarations
void printStats(const SearchStats& stats, int order);

// Generate all combinations of k elements from [1, n] and process each
class CombinationGenerator {
private:
    std::vector<int> current;
    int n, k;
    bool finished;

public:
    CombinationGenerator(int n, int k) : n(n), k(k), finished(false) {
        current.resize(k);
        // Initialize to [1, 2, ..., k]
        for (int i = 0; i < k; ++i) {
            current[i] = i + 1;
        }
    }

    bool hasNext() const {
        return !finished;
    }

    const std::vector<int>& getCurrent() const {
        return current;
    }

    void next() {
        // Find the rightmost element that can be incremented
        int i = k - 1;
        while (i >= 0 && current[i] == n - k + i + 1) {
            --i;
        }

        if (i < 0) {
            finished = true;
            return;
        }

        // Increment this element and reset all elements to the right
        ++current[i];
        for (int j = i + 1; j < k; ++j) {
            current[j] = current[j - 1] + 1;
        }
    }
};

// Brute force search for optimal Golomb ruler
SearchStats bruteForce(int order, int maxLength) {
    SearchStats stats;
    Timer timer;
    timer.start();

    if (order < 2) {
        stats.bestSolution = GolombRuler({0});
        stats.elapsedMs = timer.elapsedMs();
        return stats;
    }

    if (order == 2) {
        stats.bestSolution = GolombRuler({0, 1});
        stats.nodesExplored = 1;
        stats.elapsedMs = timer.elapsedMs();
        return stats;
    }

    // We need to choose (order - 1) positions from [1, maxLength]
    // First mark is always 0
    int k = order - 1;  // Number of additional marks to place

    CombinationGenerator gen(maxLength, k);

    std::vector<int> ruler(order);
    ruler[0] = 0;  // First mark always at 0

    while (gen.hasNext()) {
        stats.nodesExplored++;

        // Build the ruler from the combination
        const std::vector<int>& combo = gen.getCurrent();
        for (int i = 0; i < k; ++i) {
            ruler[i + 1] = combo[i];
        }

        // Check if this is a valid Golomb ruler
        if (isValidGolombRuler(ruler)) {
            int length = ruler.back();
            if (length < stats.bestSolution.length) {
                stats.bestSolution = GolombRuler(ruler);
            }
        }

        gen.next();
    }

    stats.elapsedMs = timer.elapsedMs();
    return stats;
}

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
        std::cout << "Usage: " << argv[0] << " <order> [max_length]" << '\n';
        std::cout << "  order: Number of marks in the Golomb ruler (4-6 recommended)" << '\n';
        std::cout << "  max_length: Optional maximum length to search (default: 2x optimal)" << '\n';
        std::cout << '\n';
        std::cout << "Example: " << argv[0] << " 5" << '\n';
        return 1;
    }

    int order = parseAndValidateOrder(argv[1]);
    if (order < 0) {
        std::cerr << "Error: Invalid order. Must be a number between 2 and " << (MAX_ORDER-1) << '\n';
        return 1;
    }
    if (order > 8) {
        std::cerr << "Warning: Order " << order << " may be very slow or impractical for brute force." << '\n';
        std::cerr << "Recommended range: 4-5 (6 will be slow)" << '\n';
    }

    int maxLength;
    if (argc >= 3) {
        int parsedMaxLen = parseAndValidateOrder(argv[2], MAX_LENGTH);
        maxLength = (parsedMaxLen > 0) ? parsedMaxLen : calculateMaxLength(order);
    } else {
        maxLength = calculateMaxLength(order);
    }

    std::cout << "=== Golomb Ruler Solver v1: Brute Force ===" << '\n';
    std::cout << "Order: " << order << ", Max length: " << maxLength << '\n';
    std::cout << "Searching..." << '\n';

    SearchStats stats = bruteForce(order, maxLength);

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
