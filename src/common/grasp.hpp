#ifndef GRASP_HPP
#define GRASP_HPP

/**
 * GRASP (Greedy Randomized Adaptive Search Procedure) for Golomb Rulers
 *
 * This metaheuristic finds good initial bounds for Branch & Bound.
 * Works for ANY order (known or unknown optimal).
 *
 * Algorithm:
 * 1. Greedy Randomized Construction: Build solution using RCL (Restricted Candidate List)
 * 2. Local Search: Improve solution by trying small perturbations
 * 3. Repeat and keep best solution
 *
 * Reference: Feo & Resende (1995) - Greedy Randomized Adaptive Search Procedures
 */

#include "golomb.hpp"
#include <vector>
#include <bitset>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>

namespace grasp {

/**
 * Check if a position is valid for a partial Golomb ruler
 */
inline bool isValidPosition(int pos, const std::vector<int>& marks,
                           const std::bitset<MAX_LENGTH>& usedDiffs) {
    for (int m : marks) {
        int diff = pos - m;
        if (diff <= 0 || diff >= MAX_LENGTH) return false;
        if (usedDiffs.test(diff)) return false;
    }
    return true;
}

/**
 * Add a mark to the ruler and update differences
 */
inline void addMark(std::vector<int>& marks, std::bitset<MAX_LENGTH>& usedDiffs, int pos) {
    for (int m : marks) {
        int diff = pos - m;
        if (diff > 0 && diff < MAX_LENGTH) {
            usedDiffs.set(diff);
        }
    }
    marks.push_back(pos);
}

/**
 * Rebuild difference set from marks
 */
inline std::bitset<MAX_LENGTH> rebuildDiffs(const std::vector<int>& marks) {
    std::bitset<MAX_LENGTH> diffs;
    for (size_t i = 0; i < marks.size(); i++) {
        for (size_t j = i + 1; j < marks.size(); j++) {
            int diff = marks[j] - marks[i];
            if (diff > 0 && diff < MAX_LENGTH) {
                diffs.set(diff);
            }
        }
    }
    return diffs;
}

/**
 * Check if a ruler is valid (all differences unique)
 */
inline bool isValidRuler(const std::vector<int>& marks) {
    std::bitset<MAX_LENGTH> seen;
    for (size_t i = 0; i < marks.size(); i++) {
        for (size_t j = i + 1; j < marks.size(); j++) {
            int diff = marks[j] - marks[i];
            if (diff <= 0 || diff >= MAX_LENGTH) return false;
            if (seen.test(diff)) return false;
            seen.set(diff);
        }
    }
    return true;
}

/**
 * Greedy Randomized Construction
 *
 * @param order Number of marks
 * @param alpha RCL parameter (0.0 = pure greedy, 1.0 = pure random)
 * @param rng Random number generator
 * @return Constructed Golomb ruler
 */
inline std::vector<int> greedyRandomizedConstruction(int order, double alpha, std::mt19937& rng) {
    std::vector<int> marks;
    marks.reserve(order);
    marks.push_back(0);

    std::bitset<MAX_LENGTH> usedDiffs;

    // Estimate max length needed
    int maxLength = order * order;  // Rough upper bound
    if (maxLength > MAX_LENGTH - 1) maxLength = MAX_LENGTH - 1;

    while (static_cast<int>(marks.size()) < order) {
        // Find all valid candidate positions
        std::vector<int> candidates;
        int lastMark = marks.back();

        // Limit search range for efficiency
        int searchLimit = std::min(lastMark + maxLength / order + 50, maxLength);

        for (int pos = lastMark + 1; pos <= searchLimit; pos++) {
            if (isValidPosition(pos, marks, usedDiffs)) {
                candidates.push_back(pos);
                // Limit candidates for large orders
                if (candidates.size() >= 200) break;
            }
        }

        if (candidates.empty()) {
            // No valid position found, return partial solution
            break;
        }

        // Build Restricted Candidate List (RCL)
        // RCL contains candidates with cost in [cmin, cmin + alpha*(cmax-cmin)]
        // For Golomb, "cost" is the position value (smaller is better)
        int cmin = candidates.front();  // Already sorted by position
        int cmax = candidates.back();
        int threshold = cmin + static_cast<int>(alpha * (cmax - cmin));

        // Filter candidates within threshold
        std::vector<int> rcl;
        for (int c : candidates) {
            if (c <= threshold) {
                rcl.push_back(c);
            }
        }

        // Ensure RCL is not empty
        if (rcl.empty()) {
            rcl.push_back(candidates.front());
        }

        // Randomly select from RCL
        std::uniform_int_distribution<int> dist(0, rcl.size() - 1);
        int chosen = rcl[dist(rng)];

        addMark(marks, usedDiffs, chosen);
    }

    return marks;
}

/**
 * Local Search: Try to improve the solution
 *
 * Uses a simple neighborhood: try moving each mark by small deltas
 *
 * @param marks Current solution (will be modified)
 * @return Improved solution
 */
inline std::vector<int> localSearch(std::vector<int> marks) {
    if (marks.size() < 3) return marks;

    bool improved = true;
    int maxIterations = 100;
    int iteration = 0;

    while (improved && iteration++ < maxIterations) {
        improved = false;
        int currentLength = marks.back();

        // Try to reduce the last mark (most impactful)
        for (int delta = -1; delta >= -10 && !improved; delta--) {
            int newPos = marks.back() + delta;
            if (newPos <= marks[marks.size() - 2]) continue;

            // Check if new position is valid
            std::vector<int> testMarks = marks;
            testMarks.back() = newPos;

            if (isValidRuler(testMarks)) {
                marks = testMarks;
                improved = true;
            }
        }

        // Try moving middle marks to potentially allow smaller end
        if (!improved) {
            for (size_t i = marks.size() - 2; i >= 1; i--) {
                for (int delta = -3; delta <= 3; delta++) {
                    if (delta == 0) continue;

                    int newPos = marks[i] + delta;
                    if (newPos <= marks[i-1] || newPos >= marks[i+1]) continue;

                    std::vector<int> testMarks = marks;
                    testMarks[i] = newPos;

                    if (isValidRuler(testMarks) && testMarks.back() < currentLength) {
                        marks = testMarks;
                        improved = true;
                        break;
                    }
                }
                if (improved) break;
            }
        }
    }

    return marks;
}

/**
 * Main GRASP algorithm
 *
 * @param order Number of marks in the ruler
 * @param maxIterations Number of GRASP iterations
 * @param alpha RCL parameter (recommended: 0.1-0.3)
 * @param verbose Print progress
 * @return Best Golomb ruler found
 */
inline GolombRuler grasp(int order, int maxIterations = 1000, double alpha = 0.2, bool verbose = false) {
    // Initialize random generator with time-based seed
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);

    GolombRuler best;
    best.length = MAX_LENGTH;

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < maxIterations; iter++) {
        // Phase 1: Greedy Randomized Construction
        std::vector<int> candidate = greedyRandomizedConstruction(order, alpha, rng);

        // Skip incomplete solutions
        if (static_cast<int>(candidate.size()) != order) continue;

        // Phase 2: Local Search
        candidate = localSearch(candidate);

        // Update best solution
        if (!candidate.empty() && candidate.back() < best.length) {
            best = GolombRuler(candidate);

            if (verbose) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
                std::cout << "[GRASP] Iteration " << iter << ": Found length "
                          << best.length << " (" << elapsed << " ms)" << std::endl;
            }
        }

        // Progress reporting for long runs
        if (verbose && iter > 0 && iter % (maxIterations / 10) == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
            std::cout << "[GRASP] Progress: " << (iter * 100 / maxIterations) << "% "
                      << "Best: " << best.length << " (" << elapsed << " ms)" << std::endl;
        }
    }

    if (verbose) {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << "[GRASP] Completed " << maxIterations << " iterations in "
                  << totalTime << " ms. Best length: " << best.length << std::endl;
    }

    return best;
}

/**
 * Get recommended GRASP parameters based on order
 */
inline void getRecommendedParams(int order, int& iterations, double& alpha) {
    if (order <= 12) {
        iterations = 100;
        alpha = 0.3;
    } else if (order <= 15) {
        iterations = 500;
        alpha = 0.2;
    } else if (order <= 18) {
        iterations = 1000;
        alpha = 0.15;
    } else {
        iterations = 5000;
        alpha = 0.1;
    }
}

/**
 * Run GRASP with automatic parameter selection
 */
inline GolombRuler graspAuto(int order, bool verbose = false) {
    int iterations;
    double alpha;
    getRecommendedParams(order, iterations, alpha);

    if (verbose) {
        std::cout << "[GRASP] Auto params for G" << order
                  << ": iterations=" << iterations << ", alpha=" << alpha << std::endl;
    }

    return grasp(order, iterations, alpha, verbose);
}

} // namespace grasp

#endif // GRASP_HPP
