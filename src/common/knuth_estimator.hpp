/**
 * Knuth Tree Size Estimator for Golomb Ruler Search
 *
 * Uses Monte Carlo sampling to estimate the size of the search tree.
 * This allows displaying an approximate progress percentage during long searches.
 *
 * Algorithm (Knuth 1975):
 * 1. Perform random walks (dives) from root to leaves
 * 2. At each node, count valid children and pick one randomly
 * 3. Estimate tree size as product of branching factors along the path
 * 4. Average over many samples for accuracy
 *
 * Reference: Knuth, D.E. (1975) "Estimating the Efficiency of Backtrack Programs"
 */

#ifndef KNUTH_ESTIMATOR_HPP
#define KNUTH_ESTIMATOR_HPP

#include <cstdint>
#include <random>
#include <bitset>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "golomb.hpp"

class KnuthEstimator {
private:
    uint64_t estimatedTreeSize;
    double estimatedNodes;
    int order;
    int upperBound;
    std::mt19937 rng;
    bool calibrated;

    // Search state for sampling
    struct SampleState {
        int marks[MAX_ORDER];
        std::bitset<MAX_LENGTH> usedDiffs;
        int markCount;
    };

public:
    KnuthEstimator() : estimatedTreeSize(0), estimatedNodes(0), order(0),
                       upperBound(0), calibrated(false) {
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }

    /**
     * Calibrate the estimator by sampling the search tree
     *
     * @param n Order of the Golomb ruler
     * @param bound Initial upper bound
     * @param numSamples Number of random walks (default 1000)
     * @param maxTimeMs Maximum calibration time in milliseconds (default 1000)
     */
    void calibrate(int n, int bound, int numSamples = 1000, int maxTimeMs = 1000) {
        order = n;
        upperBound = bound;
        calibrated = true;

        auto startTime = std::chrono::high_resolution_clock::now();

        std::vector<double> estimates;
        estimates.reserve(numSamples);

        for (int sample = 0; sample < numSamples; ++sample) {
            // Check time limit
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
            if (elapsed >= maxTimeMs) break;

            double estimate = performRandomWalk();
            if (estimate > 0) {
                estimates.push_back(estimate);
            }
        }

        if (estimates.empty()) {
            estimatedNodes = 1000000;  // Fallback estimate
            return;
        }

        // Use median for robustness (more stable than mean for heavy-tailed distributions)
        std::sort(estimates.begin(), estimates.end());
        size_t mid = estimates.size() / 2;
        if (estimates.size() % 2 == 0) {
            estimatedNodes = (estimates[mid - 1] + estimates[mid]) / 2.0;
        } else {
            estimatedNodes = estimates[mid];
        }

        estimatedTreeSize = static_cast<uint64_t>(estimatedNodes);
    }

    /**
     * Get estimated progress as a percentage
     *
     * @param nodesVisited Number of nodes explored so far
     * @return Progress percentage (0-100), capped at 99% until completion
     */
    double getProgress(uint64_t nodesVisited) const {
        if (!calibrated || estimatedNodes <= 0) return 0.0;
        double progress = (static_cast<double>(nodesVisited) / estimatedNodes) * 100.0;
        return std::min(progress, 99.0);  // Cap at 99% until we're sure it's done
    }

    /**
     * Get estimated time remaining
     *
     * @param nodesVisited Nodes explored so far
     * @param elapsedMs Time elapsed so far in milliseconds
     * @return Formatted ETA string (e.g., "~2m 30s" or "~45s")
     */
    std::string formatETA(uint64_t nodesVisited, double elapsedMs) const {
        if (!calibrated || nodesVisited == 0 || estimatedNodes <= 0) {
            return "calculating...";
        }

        double progress = static_cast<double>(nodesVisited) / estimatedNodes;
        if (progress >= 1.0) {
            return "almost done";
        }

        double totalEstimatedMs = elapsedMs / progress;
        double remainingMs = totalEstimatedMs - elapsedMs;

        if (remainingMs < 0) remainingMs = 0;

        return formatTime(remainingMs);
    }

    /**
     * Get the estimated tree size
     */
    uint64_t getEstimatedSize() const { return estimatedTreeSize; }

    /**
     * Check if calibration was performed
     */
    bool isCalibrated() const { return calibrated; }

private:
    /**
     * Perform a single random walk from root to leaf
     * Returns estimated subtree size based on branching factors
     */
    double performRandomWalk() {
        SampleState state;
        state.marks[0] = 0;
        state.markCount = 1;
        state.usedDiffs.reset();

        double pathEstimate = 1.0;
        int currentBound = upperBound;

        for (int depth = 1; depth < order; ++depth) {
            // Count valid children at this node
            std::vector<int> validPositions;
            int lastMark = state.marks[state.markCount - 1];
            int maxPos = currentBound - 1;

            // Symmetry breaking for first mark
            if (depth == 1) {
                maxPos = std::min(maxPos, currentBound / 2);
            }

            for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
                // Pruning: check if remaining marks can fit
                int remaining = order - depth - 1;
                if (pos + remaining >= currentBound) break;

                // Check if position is valid (no duplicate differences)
                bool valid = true;
                for (int i = 0; i < state.markCount && valid; ++i) {
                    int diff = pos - state.marks[i];
                    if (diff >= MAX_LENGTH || state.usedDiffs.test(diff)) {
                        valid = false;
                    }
                }

                if (valid) {
                    validPositions.push_back(pos);
                }
            }

            // No valid children - this path is a dead end
            if (validPositions.empty()) {
                break;
            }

            // Multiply path estimate by branching factor
            pathEstimate *= validPositions.size();

            // Pick a random child to continue
            std::uniform_int_distribution<int> dist(0, validPositions.size() - 1);
            int chosenPos = validPositions[dist(rng)];

            // Add the chosen position to the state
            state.marks[state.markCount] = chosenPos;
            for (int i = 0; i < state.markCount; ++i) {
                int diff = chosenPos - state.marks[i];
                state.usedDiffs.set(diff);
            }
            state.markCount++;
        }

        return pathEstimate;
    }

    /**
     * Format milliseconds as a human-readable time string
     */
    static std::string formatTime(double ms) {
        std::ostringstream oss;
        oss << "~";

        if (ms < 1000) {
            oss << "<1s";
        } else if (ms < 60000) {
            oss << static_cast<int>(ms / 1000) << "s";
        } else if (ms < 3600000) {
            int minutes = static_cast<int>(ms / 60000);
            int seconds = static_cast<int>((ms - minutes * 60000) / 1000);
            oss << minutes << "m " << seconds << "s";
        } else {
            int hours = static_cast<int>(ms / 3600000);
            int minutes = static_cast<int>((ms - hours * 3600000) / 60000);
            oss << hours << "h " << minutes << "m";
        }

        return oss.str();
    }
};

#endif // KNUTH_ESTIMATOR_HPP
