/**
 * @file csv_output.hpp
 * @brief Common CSV output functions for all solver versions
 *
 * Golomb Ruler Solver - High Performance Computing Implementation
 * Copyright (c) 2025 Nicolas Marano
 * Licensed under the MIT License. See LICENSE file for details.
 *
 * Provides shared CSV writing functions to eliminate duplication
 * across v1, v2, v3, and v4 implementations.
 */

#ifndef GOLOMB_CSV_OUTPUT_HPP
#define GOLOMB_CSV_OUTPUT_HPP

#include "golomb.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace golomb {

/**
 * @brief Appends benchmark results to a CSV file (sequential/OpenMP format).
 *
 * Creates the file with headers if it doesn't exist, otherwise appends
 * a new row with the benchmark results.
 *
 * CSV columns: version,order,threads,time_ms,nodes_explored,nodes_pruned,solution,length
 *
 * @param filename Path to the CSV file
 * @param version  Solver version number (1 for sequential, 2 for OpenMP)
 * @param order    Order of the Golomb ruler solved
 * @param stats    Search statistics including time and solution
 * @param threads  Number of threads used (1 for sequential)
 */
inline void appendResultCSV(const std::string& filename, int version, int order,
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

/**
 * @brief Appends benchmark results to a CSV file (MPI format).
 *
 * Creates the file with headers if it doesn't exist, otherwise appends
 * a new row with the benchmark results including MPI-specific columns.
 *
 * CSV columns: version,order,mpi_procs,omp_threads,total_workers,time_ms,nodes,pruned,solution,length
 *
 * @param filename   Path to the CSV file
 * @param version    Solver version number (3 for hybrid, 4 for hypercube)
 * @param order      Order of the Golomb ruler solved
 * @param mpiProcs   Number of MPI processes
 * @param ompThreads Number of OpenMP threads per process
 * @param timeMs     Total execution time in milliseconds
 * @param nodes      Total nodes explored
 * @param pruned     Total nodes pruned
 * @param solution   Best solution found
 */
inline void appendResultCSV_MPI(const std::string& filename, int version, int order,
                                int mpiProcs, int ompThreads,
                                double timeMs, uint64_t nodes, uint64_t pruned,
                                const GolombRuler& solution) {
    std::ifstream check(filename);
    bool header = !check.good();
    check.close();

    std::ofstream f(filename, std::ios::app);
    if (!f.is_open()) return;

    if (header) {
        f << "version,order,mpi_procs,omp_threads,total_workers,time_ms,nodes,pruned,solution,length\n";
    }

    f << version << "," << order << "," << mpiProcs << "," << ompThreads << ","
      << (mpiProcs * ompThreads) << ","
      << std::fixed << std::setprecision(2) << timeMs << ","
      << nodes << "," << pruned << ","
      << "\"" << solution.toString() << "\"," << solution.length << "\n";
}

}  // namespace golomb

#endif  // GOLOMB_CSV_OUTPUT_HPP
