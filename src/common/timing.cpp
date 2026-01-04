#include "golomb.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>

// Timer implementation

Timer::Timer() : running(false) {}

void Timer::start() {
    startTime = std::chrono::high_resolution_clock::now();
    running = true;
}

void Timer::reset() {
    startTime = std::chrono::high_resolution_clock::now();
    running = false;
}

double Timer::elapsedMs() const {
    if (!running) return 0.0;
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - startTime);
    return duration.count() / 1000.0;  // Convert to milliseconds
}

// Write results to CSV file
void writeResultCSV(const std::string& filename,
                    int version,
                    const std::vector<SearchStats>& results,
                    const std::vector<int>& orders) {

    // Open file and check if it's empty/new in one operation
    std::ofstream file(filename, std::ios::app | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << '\n';
        return;
    }

    // Write header if file is empty (tellp() == 0 means file is empty)
    if (file.tellp() == 0) {
        file << "version,order,time_ms,nodes_explored,nodes_pruned,solution,length\n";
    }

    // Write each result
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

// Print search statistics summary
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
