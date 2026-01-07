/**
 * @file timing.cpp
 * @brief Performance timing and CSV export utilities
 *
 * Golomb Ruler Solver - High Performance Computing Implementation
 * Copyright (c) 2025 Nicolas Marano
 * Licensed under the MIT License. See LICENSE file for details.
 */

#include "golomb/golomb.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>

// ============================================================================
// Timer Implementation
// ============================================================================

/** @copydoc Timer::Timer() */
Timer::Timer() : running(false) {}

/** @copydoc Timer::start() */
void Timer::start() {
    startTime = std::chrono::high_resolution_clock::now();
    running = true;
}

/** @copydoc Timer::reset() */
void Timer::reset() {
    startTime = std::chrono::high_resolution_clock::now();
    running = false;
}

/** @copydoc Timer::elapsedMs() */
double Timer::elapsedMs() const {
    // Returns 0.0 if timer was never started via start().
    // This is intentional: callers should check timer state or ensure start() was called.
    if (!running) return 0.0;
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - startTime);
    return duration.count() / 1000.0;  // Convert to milliseconds
}

// ============================================================================
// CSV Export
// ============================================================================

/** @copydoc writeResultCSV() */
void writeResultCSV(const std::string& filename,
                    int version,
                    const std::vector<SearchStats>& results,
                    const std::vector<int>& orders) {

    // Open file in append mode, position at end to check if empty
    std::ofstream file(filename, std::ios::app | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << '\n';
        return;
    }

    // Write CSV header if file is empty (tellp() == 0)
    if (file.tellp() == 0) {
        file << "version,order,time_ms,nodes_explored,nodes_pruned,solution,length\n";
    }

    // Write each result row
    for (size_t i = 0; i < results.size() && i < orders.size(); ++i) {
        const auto& stats = results[i];
        file << version << ","
             << orders[i] << ","
             << std::fixed << std::setprecision(2) << stats.elapsedMs << ","
             << stats.nodesExplored << ","
             << stats.nodesPruned << ","
             << "\"" << stats.bestSolution.toString() << "\","
             << stats.bestSolution.length << '\n';
    }

    file.close();
    std::cout << "Results written to " << filename << '\n';
}

// ============================================================================
// Statistics Output
// ============================================================================

/** @copydoc printStats() */
void printStats(const SearchStats& stats, int order) {
    std::cout << "=== Golomb G" << order << " ===\n";
    std::cout << "Solution: " << stats.bestSolution.toString() << '\n';
    std::cout << "Length: " << stats.bestSolution.length << '\n';
    std::cout << "Time: " << std::fixed << std::setprecision(2) << stats.elapsedMs << " ms\n";
    std::cout << "Nodes explored: " << stats.nodesExplored << '\n';
    if (stats.nodesPruned > 0) {
        std::cout << "Nodes pruned: " << stats.nodesPruned << '\n';
        double pruneRatio = 100.0 * stats.nodesPruned / (stats.nodesExplored + stats.nodesPruned);
        std::cout << "Pruning ratio: " << std::fixed << std::setprecision(1) << pruneRatio << "%\n";
    }
    std::cout << '\n';
}
