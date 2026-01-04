/**
 * Golomb Ruler Solver - Parallel Version 3: Optimized MPI
 *
 * Enhancements over v2:
 * - Dynamic work distribution (workers request work)
 * - Finer-grained subtrees for better load balancing
 * - Hypercube communication for bounds
 * - Hypercube reduce for final result collection
 *
 * This is the production-ready parallel solver.
 */

#include <mpi.h>
#include "golomb.hpp"
#include "greedy.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <functional>
#include <iomanip>
#include <cmath>
#include <fstream>

// MPI error checking macro
#define MPI_CHECK(call) do { \
    int err = (call); \
    if (err != MPI_SUCCESS) { \
        char errstr[MPI_MAX_ERROR_STRING]; \
        int errlen; \
        MPI_Error_string(err, errstr, &errlen); \
        std::cerr << "MPI Error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << errstr << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, err); \
    } \
} while(0)

// MPI tags
const int TAG_WORK = 1;
const int TAG_RESULT = 2;
const int TAG_DONE = 3;
const int TAG_BOUND = 4;
const int TAG_REQUEST_WORK = 5;

// How often to check for new bounds (every N nodes)
// 100000 is standardized across all versions for consistency
const int CHECK_INTERVAL = 100000;

// Subtree structure
struct Subtree {
    int marks[MAX_ORDER];
    int markCount;
    unsigned char usedDiffs[MAX_LENGTH / 8 + 1];
    int bestBound;

    void setDiff(int d) {
        if (d < MAX_LENGTH) usedDiffs[d / 8] |= (1 << (d % 8));
    }
    bool testDiff(int d) const {
        if (d >= MAX_LENGTH) return false;
        return usedDiffs[d / 8] & (1 << (d % 8));
    }
    void clearDiffs() {
        std::memset(usedDiffs, 0, sizeof(usedDiffs));
    }
};

// Result structure
struct Result {
    int marks[MAX_ORDER];
    int length;
    int order;
    uint64_t nodesExplored;
    double timeMs;
};

// Hypercube utilities
int hypercubeDimension(int size) {
    int d = 0, s = size;
    while (s > 1) { s /= 2; d++; }
    return d;
}

std::vector<int> getHypercubeNeighbors(int rank, int size) {
    std::vector<int> neighbors;
    int d = hypercubeDimension(size);
    for (int i = 0; i < d; ++i) {
        int neighbor = rank ^ (1 << i);
        if (neighbor < size) neighbors.push_back(neighbor);
    }
    return neighbors;
}

bool checkForBoundUpdate(int& currentBound) {
    int flag;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_BOUND, MPI_COMM_WORLD, &flag, &status);
    if (flag) {
        int newBound;
        MPI_Recv(&newBound, 1, MPI_INT, status.MPI_SOURCE, TAG_BOUND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (newBound < currentBound) {
            currentBound = newBound;
            return true;
        }
    }
    return false;
}

void broadcastBoundToNeighbors(int bound, int rank, int size) {
    for (int neighbor : getHypercubeNeighbors(rank, size)) {
        MPI_Request request;
        MPI_Isend(&bound, 1, MPI_INT, neighbor, TAG_BOUND, MPI_COMM_WORLD, &request);
        MPI_Request_free(&request);
    }
}

// Solver with dynamic bounds
class OptimizedSubtreeSolver {
private:
    int order;
    int marks[MAX_ORDER];
    int markCount;
    std::bitset<MAX_LENGTH> usedDiffs;
    GolombRuler bestSolution;
    uint64_t nodesExplored;
    int* globalBound;
    int rank, size;

public:
    OptimizedSubtreeSolver(int n, const Subtree& subtree, int* bound, int r, int s)
        : order(n), nodesExplored(0), globalBound(bound), rank(r), size(s) {

        markCount = subtree.markCount;
        for (int i = 0; i < markCount; ++i) marks[i] = subtree.marks[i];

        usedDiffs.reset();
        for (int d = 0; d < MAX_LENGTH; ++d)
            if (subtree.testDiff(d)) usedDiffs.set(d);

        bestSolution.length = *globalBound;
    }

    void solve() { branchAndBound(markCount); }

    Result getResult() const {
        Result r;
        r.order = order;
        r.length = bestSolution.length;
        r.nodesExplored = nodesExplored;
        for (int i = 0; i < order && i < (int)bestSolution.marks.size(); ++i)
            r.marks[i] = bestSolution.marks[i];
        return r;
    }

private:
    void branchAndBound(int depth) {
        nodesExplored++;

        if (nodesExplored % CHECK_INTERVAL == 0) {
            if (checkForBoundUpdate(*globalBound)) {
                if (*globalBound < bestSolution.length)
                    bestSolution.length = *globalBound;
            }
        }

        if (depth == order) {
            int length = marks[markCount - 1];
            if (length < bestSolution.length) {
                std::vector<int> v(marks, marks + markCount);
                bestSolution = GolombRuler(v);
                if (length < *globalBound) {
                    *globalBound = length;
                    broadcastBoundToNeighbors(length, rank, size);
                }
            }
            return;
        }

        int lastMark = marks[markCount - 1];
        int maxPos = std::min(bestSolution.length - 1, *globalBound - 1);

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            if (pos + (order - depth - 1) >= *globalBound) continue;

            int newDiffs[MAX_ORDER];
            int newDiffCount = 0;
            bool valid = true;

            for (int i = 0; i < markCount; ++i) {
                int diff = pos - marks[i];
                if (diff >= MAX_LENGTH || usedDiffs.test(diff)) { valid = false; break; }
                newDiffs[newDiffCount++] = diff;
            }

            if (valid) {
                marks[markCount++] = pos;
                for (int i = 0; i < newDiffCount; ++i) usedDiffs.set(newDiffs[i]);
                branchAndBound(depth + 1);
                markCount--;
                for (int i = 0; i < newDiffCount; ++i) usedDiffs.reset(newDiffs[i]);
            }
        }
    }
};

// Generate subtrees with adaptive depth
std::vector<Subtree> generateSubtrees(int order, int prefixDepth, int bound) {
    std::vector<Subtree> subtrees;

    std::function<void(Subtree&, int)> generate = [&](Subtree& current, int depth) {
        if (depth == prefixDepth) {
            subtrees.push_back(current);
            return;
        }

        int lastMark = current.marks[current.markCount - 1];
        int maxPos = (depth == 1) ? std::min(bound - 1, bound / 2) : bound - 1;

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            if (pos + (order - depth - 1) >= bound) break;

            bool valid = true;
            std::vector<int> newDiffs;
            for (int i = 0; i < current.markCount; ++i) {
                int diff = pos - current.marks[i];
                if (diff >= MAX_LENGTH || current.testDiff(diff)) { valid = false; break; }
                newDiffs.push_back(diff);
            }

            if (valid) {
                Subtree next = current;
                next.marks[next.markCount++] = pos;
                for (int d : newDiffs) next.setDiff(d);
                generate(next, depth + 1);
            }
        }
    };

    Subtree root;
    root.clearDiffs();
    root.marks[0] = 0;
    root.markCount = 1;
    root.bestBound = bound;
    generate(root, 1);

    return subtrees;
}

// Greedy solution for initial bound - uses shared implementation from greedy.hpp
int greedyBound(int order) {
    return computeGreedyBound(order);
}

// MPI datatypes
MPI_Datatype createSubtreeType() {
    MPI_Datatype type;
    int bl[] = {MAX_ORDER, 1, MAX_LENGTH/8+1, 1};
    MPI_Aint disp[4] = {
        offsetof(Subtree, marks), offsetof(Subtree, markCount),
        offsetof(Subtree, usedDiffs), offsetof(Subtree, bestBound)
    };
    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_UNSIGNED_CHAR, MPI_INT};
    MPI_Type_create_struct(4, bl, disp, types, &type);
    MPI_Type_commit(&type);
    return type;
}

MPI_Datatype createResultType() {
    MPI_Datatype type;
    int bl[] = {MAX_ORDER, 1, 1, 1, 1};
    MPI_Aint disp[5] = {
        offsetof(Result, marks), offsetof(Result, length),
        offsetof(Result, order), offsetof(Result, nodesExplored),
        offsetof(Result, timeMs)
    };
    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_INT, MPI_UINT64_T, MPI_DOUBLE};
    MPI_Type_create_struct(5, bl, disp, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void appendParallelCSV(const std::string& filename, int version, int order, int procs,
                       double timeMs, double seqTime, uint64_t nodes, const GolombRuler& solution) {
    std::ifstream check(filename);
    bool header = !check.good();
    check.close();

    std::ofstream f(filename, std::ios::app);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open " << filename << '\n';
        return;
    }

    if (header) {
        f << "version,order,procs,time_ms,speedup,efficiency,nodes,solution,length\n";
        if (!f.good()) {
            std::cerr << "Error: Failed to write CSV header to " << filename << '\n';
            return;
        }
    }

    double speedup = (seqTime > 0) ? seqTime / timeMs : 0;
    double efficiency = speedup / procs;

    f << version << "," << order << "," << procs << ","
      << std::fixed << std::setprecision(2) << timeMs << ","
      << std::setprecision(2) << speedup << ","
      << std::setprecision(3) << efficiency << ","
      << nodes << ",\"" << solution.toString() << "\"," << solution.length << "\n";

    if (!f.good()) {
        std::cerr << "Error: Failed to write CSV data to " << filename << '\n';
    }

    f.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            std::cout << "Usage: mpirun -np <procs> " << argv[0] << " <order> [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --depth N     Prefix depth (default: auto)\n";
            std::cout << "  --csv FILE    Save results to CSV\n";
            std::cout << "  --seq-time T  Sequential time for speedup calc\n";
        }
        MPI_Finalize();
        return 1;
    }

    int order = parseAndValidateOrder(argv[1]);
    if (order < 0) {
        if (rank == 0) {
            std::cerr << "Error: Invalid order. Must be a number between 2 and " << (MAX_ORDER-1) << '\n';
        }
        MPI_Finalize();
        return 1;
    }
    if (order < 4) {
        if (rank == 0) {
            std::cerr << "Error: Order must be at least 4 for MPI version\n";
        }
        MPI_Finalize();
        return 1;
    }
    int prefixDepth = (order <= 8) ? 3 : (order <= 10) ? 4 : 5;
    std::string csvFile;
    double seqTime = 0;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--depth" && i+1 < argc) {
            int pd = parseAndValidateOrder(argv[++i], 10);
            if (pd > 0) prefixDepth = pd;
        }
        else if (arg == "--csv" && i+1 < argc) csvFile = argv[++i];
        else if (arg == "--seq-time" && i+1 < argc) {
            char* endptr;
            double val = std::strtod(argv[++i], &endptr);
            if (*endptr == '\0' && val >= 0) seqTime = val;
        }
    }

    MPI_Datatype subtreeType = createSubtreeType();
    MPI_Datatype resultType = createResultType();

    double startTime = MPI_Wtime();

    if (rank == 0) {
        std::cout << "=== Golomb Ruler Solver - MPI v3: Optimized ===\n";
        std::cout << "Order: " << order << ", Processes: " << size << "\n";
        std::cout << "Prefix depth: " << prefixDepth << "\n";

        int bound = greedyBound(order);
        std::cout << "Initial bound: " << bound << "\n";

        std::vector<Subtree> subtrees = generateSubtrees(order, prefixDepth, bound);
        std::cout << "Subtrees: " << subtrees.size() << "\n";

        // Dynamic distribution
        size_t nextSubtree = 0;
        int activeWorkers = 0;

        // Initial batch
        for (int w = 1; w < size && nextSubtree < subtrees.size(); ++w) {
            subtrees[nextSubtree].bestBound = bound;
            MPI_Send(&subtrees[nextSubtree], 1, subtreeType, w, TAG_WORK, MPI_COMM_WORLD);
            nextSubtree++;
            activeWorkers++;
        }

        GolombRuler globalBest;
        globalBest.length = bound;
        int pruningBound = bound;  // Separate bound for pruning
        uint64_t totalNodes = 0;
        double maxWorkerTime = 0, minWorkerTime = 1e9;

        while (activeWorkers > 0) {
            // Process bound updates (only for pruning, not for solution)
            int flag;
            MPI_Status status;
            while (true) {
                MPI_Iprobe(MPI_ANY_SOURCE, TAG_BOUND, MPI_COMM_WORLD, &flag, &status);
                if (!flag) break;
                int newBound;
                MPI_Recv(&newBound, 1, MPI_INT, status.MPI_SOURCE, TAG_BOUND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (newBound < pruningBound) {
                    pruningBound = newBound;
                }
            }

            // Check for results
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
                Result result;
                MPI_Recv(&result, 1, resultType, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                activeWorkers--;
                totalNodes += result.nodesExplored;
                maxWorkerTime = std::max(maxWorkerTime, result.timeMs);
                minWorkerTime = std::min(minWorkerTime, result.timeMs);

                // Only update best solution if we have actual marks
                if (result.length < globalBest.length && result.length < INT_MAX) {
                    globalBest.length = result.length;
                    globalBest.order = result.order;
                    globalBest.marks.assign(result.marks, result.marks + result.order);
                    pruningBound = std::min(pruningBound, result.length);
                }

                // Send more work with current pruning bound
                if (nextSubtree < subtrees.size()) {
                    subtrees[nextSubtree].bestBound = pruningBound;
                    MPI_Send(&subtrees[nextSubtree], 1, subtreeType, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
                    nextSubtree++;
                    activeWorkers++;
                }
            }
        }

        // Terminate workers
        Subtree dummy;
        for (int w = 1; w < size; ++w) {
            MPI_Send(&dummy, 1, subtreeType, w, TAG_DONE, MPI_COMM_WORLD);
        }

        // Drain bounds
        int flag;
        MPI_Status status;
        while (true) {
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_BOUND, MPI_COMM_WORLD, &flag, &status);
            if (!flag) break;
            int tmp;
            MPI_Recv(&tmp, 1, MPI_INT, MPI_ANY_SOURCE, TAG_BOUND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        double endTime = MPI_Wtime();
        double elapsedMs = (endTime - startTime) * 1000.0;

        std::cout << "\n=== Results ===\n";
        std::cout << "Solution: " << globalBest.toString() << "\n";
        std::cout << "Length: " << globalBest.length << "\n";
        std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsedMs << " ms\n";
        std::cout << "Nodes: " << totalNodes << "\n";

        if (seqTime > 0) {
            std::cout << "Speedup: " << std::setprecision(2) << seqTime/elapsedMs << "x\n";
            std::cout << "Efficiency: " << std::setprecision(1) << 100*seqTime/(elapsedMs*size) << "%\n";
        }

        if (order <= 14 && globalBest.length == OPTIMAL_LENGTHS[order]) {
            std::cout << "*** OPTIMAL ***\n";
        }

        if (!csvFile.empty()) {
            appendParallelCSV(csvFile, 3, order, size, elapsedMs, seqTime, totalNodes, globalBest);
        }

    } else {
        int localBound = INT_MAX;

        while (true) {
            Subtree subtree;
            MPI_Status status;
            MPI_Recv(&subtree, 1, subtreeType, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_DONE) break;

            localBound = subtree.bestBound;
            double workerStart = MPI_Wtime();

            OptimizedSubtreeSolver solver(order, subtree, &localBound, rank, size);
            solver.solve();

            Result result = solver.getResult();
            result.timeMs = (MPI_Wtime() - workerStart) * 1000.0;
            MPI_Send(&result, 1, resultType, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }

    MPI_Type_free(&subtreeType);
    MPI_Type_free(&resultType);
    MPI_Finalize();

    return 0;
}
