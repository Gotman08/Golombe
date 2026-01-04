/**
 * Golomb Ruler Solver - Parallel Version 2: Hypercube Communication
 *
 * Enhancement over v1:
 * - Hypercube broadcast of improved bounds between workers
 * - Workers check for new bounds periodically
 * - When a better solution is found, broadcast to neighbors
 *
 * Hypercube topology (for p = 2^d processes):
 * - Each process has d neighbors
 * - Neighbor i of process r = r XOR 2^i
 * - Broadcast in d = log2(p) steps
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

// MPI error checking macro
#define MPI_CHECK(call) do { \
    int err = (call); \
    if (err != MPI_SUCCESS) { \
        char errstr[MPI_MAX_ERROR_STRING]; \
        int errlen; \
        MPI_Error_string(err, errstr, &errlen); \
        std::cerr << "MPI Error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << errstr << '\n'; \
        MPI_Abort(MPI_COMM_WORLD, err); \
    } \
} while(0)

// MPI tags
const int TAG_WORK = 1;
const int TAG_RESULT = 2;
const int TAG_DONE = 3;
const int TAG_BOUND = 4;

// How often to check for new bounds (every N nodes)
// Lower values = fresher bounds, but more MPI overhead
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
};

// Hypercube helper: get dimension (log2 of size)
int hypercubeDimension(int size) {
    int d = 0;
    int s = size;
    while (s > 1) {
        s /= 2;
        d++;
    }
    return d;
}

// Get hypercube neighbors for a given rank
std::vector<int> getHypercubeNeighbors(int rank, int size) {
    std::vector<int> neighbors;
    int d = hypercubeDimension(size);
    for (int i = 0; i < d; ++i) {
        int neighbor = rank ^ (1 << i);
        if (neighbor < size) {
            neighbors.push_back(neighbor);
        }
    }
    return neighbors;
}

// Check for incoming bound updates (non-blocking)
bool checkForBoundUpdate(int& currentBound, int rank) {
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

// Broadcast new bound to hypercube neighbors
void broadcastBoundToNeighbors(int bound, int rank, int size) {
    std::vector<int> neighbors = getHypercubeNeighbors(rank, size);
    for (int neighbor : neighbors) {
        MPI_Request request;
        MPI_Isend(&bound, 1, MPI_INT, neighbor, TAG_BOUND, MPI_COMM_WORLD, &request);
        MPI_Request_free(&request);  // Fire and forget
    }
}

// Sequential solver with bound updates
class SubtreeSolverWithBounds {
private:
    int order;
    int marks[MAX_ORDER];
    int markCount;
    std::bitset<MAX_LENGTH> usedDiffs;
    GolombRuler bestSolution;
    uint64_t nodesExplored;
    int* globalBound;  // Pointer to shared bound
    int rank;
    int size;
    uint64_t broadcastCount;

public:
    SubtreeSolverWithBounds(int n, const Subtree& subtree, int* bound, int r, int s)
        : order(n), nodesExplored(0), globalBound(bound), rank(r), size(s), broadcastCount(0) {

        markCount = subtree.markCount;
        for (int i = 0; i < markCount; ++i) {
            marks[i] = subtree.marks[i];
        }

        usedDiffs.reset();
        for (int d = 0; d < MAX_LENGTH; ++d) {
            if (subtree.testDiff(d)) {
                usedDiffs.set(d);
            }
        }

        bestSolution.length = *globalBound;
    }

    void solve() {
        branchAndBound(markCount);
    }

    Result getResult() const {
        Result r;
        r.order = order;
        r.length = bestSolution.length;
        r.nodesExplored = nodesExplored;
        for (int i = 0; i < order && i < static_cast<int>(bestSolution.marks.size()); ++i) {
            r.marks[i] = bestSolution.marks[i];
        }
        return r;
    }

    uint64_t getBroadcastCount() const { return broadcastCount; }

private:
    void branchAndBound(int depth) {
        nodesExplored++;

        // Periodically check for bound updates
        if (nodesExplored % CHECK_INTERVAL == 0) {
            if (checkForBoundUpdate(*globalBound, rank)) {
                // Update local best if global bound improved
                if (*globalBound < bestSolution.length) {
                    bestSolution.length = *globalBound;
                }
            }
        }

        if (depth == order) {
            int length = marks[markCount - 1];
            if (length < bestSolution.length) {
                std::vector<int> v(marks, marks + markCount);
                bestSolution = GolombRuler(v);

                // Update global bound and broadcast
                if (length < *globalBound) {
                    *globalBound = length;
                    broadcastBoundToNeighbors(length, rank, size);
                    broadcastCount++;
                }
            }
            return;
        }

        int lastMark = marks[markCount - 1];
        int maxPos = std::min(bestSolution.length - 1, *globalBound - 1);

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            int remainingMarks = order - depth - 1;
            if (pos + remainingMarks >= *globalBound) {
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

// Generate subtrees
std::vector<Subtree> generateSubtrees(int order, int prefixDepth, int bound) {
    std::vector<Subtree> subtrees;

    std::function<void(Subtree&, int)> generate = [&](Subtree& current, int depth) {
        if (depth == prefixDepth) {
            subtrees.push_back(current);
            return;
        }

        int lastMark = current.marks[current.markCount - 1];
        int maxPos = bound - 1;

        if (depth == 1) {
            maxPos = std::min(maxPos, bound / 2);
        }

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            int remaining = order - depth - 1;
            if (pos + remaining >= bound) break;

            bool valid = true;
            std::vector<int> newDiffs;

            for (int i = 0; i < current.markCount; ++i) {
                int diff = pos - current.marks[i];
                if (diff >= MAX_LENGTH || current.testDiff(diff)) {
                    valid = false;
                    break;
                }
                newDiffs.push_back(diff);
            }

            if (valid) {
                Subtree next = current;
                next.marks[next.markCount++] = pos;
                for (int d : newDiffs) {
                    next.setDiff(d);
                }
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

// Create MPI datatypes
MPI_Datatype createSubtreeType() {
    MPI_Datatype type;
    int blocklengths[] = {MAX_ORDER, 1, MAX_LENGTH / 8 + 1, 1};
    MPI_Aint displacements[4];
    displacements[0] = offsetof(Subtree, marks);
    displacements[1] = offsetof(Subtree, markCount);
    displacements[2] = offsetof(Subtree, usedDiffs);
    displacements[3] = offsetof(Subtree, bestBound);
    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_UNSIGNED_CHAR, MPI_INT};
    MPI_Type_create_struct(4, blocklengths, displacements, types, &type);
    MPI_Type_commit(&type);
    return type;
}

MPI_Datatype createResultType() {
    MPI_Datatype type;
    int blocklengths[] = {MAX_ORDER, 1, 1, 1};
    MPI_Aint displacements[4];
    displacements[0] = offsetof(Result, marks);
    displacements[1] = offsetof(Result, length);
    displacements[2] = offsetof(Result, order);
    displacements[3] = offsetof(Result, nodesExplored);
    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_INT, MPI_UINT64_T};
    MPI_Type_create_struct(4, blocklengths, displacements, types, &type);
    MPI_Type_commit(&type);
    return type;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            std::cout << "Usage: mpirun -np <procs> " << argv[0] << " <order> [prefix_depth]" << '\n';
            std::cout << "  order: Number of marks (7-11 recommended)" << '\n';
            std::cout << "  prefix_depth: Subtree prefix depth (default: 3)" << '\n';
            std::cout << "\nNote: Number of processes should ideally be a power of 2 for hypercube." << '\n';
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
    int prefixDepth = 3;
    if (argc >= 3) {
        int pd = parseAndValidateOrder(argv[2], 10);
        if (pd > 0) prefixDepth = pd;
    }

    if (order < 4) {
        if (rank == 0) {
            std::cerr << "Error: Order must be at least 4 for MPI version\n";
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Datatype subtreeType = createSubtreeType();
    MPI_Datatype resultType = createResultType();

    double startTime = MPI_Wtime();

    if (rank == 0) {
        std::cout << "=== Golomb Ruler Solver - MPI v2: Hypercube ===" << '\n';
        std::cout << "Order: " << order << ", Processes: " << size << '\n';
        std::cout << "Hypercube dimension: " << hypercubeDimension(size) << '\n';
        std::cout << "Prefix depth: " << prefixDepth << '\n';

        int bound = greedyBound(order);
        std::cout << "Initial bound (greedy): " << bound << '\n';

        std::vector<Subtree> subtrees = generateSubtrees(order, prefixDepth, bound);
        std::cout << "Subtrees generated: " << subtrees.size() << '\n';

        int nextSubtree = 0;
        int activeWorkers = 0;

        // Initial distribution
        for (int w = 1; w < size && nextSubtree < static_cast<int>(subtrees.size()); ++w) {
            subtrees[nextSubtree].bestBound = bound;
            MPI_Send(&subtrees[nextSubtree], 1, subtreeType, w, TAG_WORK, MPI_COMM_WORLD);
            nextSubtree++;
            activeWorkers++;
        }

        GolombRuler globalBest;
        globalBest.length = bound;
        uint64_t totalNodes = 0;
        uint64_t totalBroadcasts = 0;

        while (activeWorkers > 0) {
            // Check for bound updates from workers
            int flag;
            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_BOUND, MPI_COMM_WORLD, &flag, &status);
            while (flag) {
                int newBound;
                MPI_Recv(&newBound, 1, MPI_INT, status.MPI_SOURCE, TAG_BOUND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (newBound < globalBest.length) {
                    globalBest.length = newBound;
                    // Propagate to all subtrees not yet distributed
                    for (int i = nextSubtree; i < static_cast<int>(subtrees.size()); ++i) {
                        subtrees[i].bestBound = newBound;
                    }
                }
                MPI_Iprobe(MPI_ANY_SOURCE, TAG_BOUND, MPI_COMM_WORLD, &flag, &status);
            }

            // Check for results
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
                Result result;
                MPI_Recv(&result, 1, resultType, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                activeWorkers--;
                totalNodes += result.nodesExplored;

                // Use <= to capture solution even if bound was already updated
                if (result.length <= globalBest.length && result.length < INT_MAX) {
                    globalBest.length = result.length;
                    globalBest.order = result.order;
                    globalBest.marks.assign(result.marks, result.marks + result.order);
                }

                if (nextSubtree < static_cast<int>(subtrees.size())) {
                    subtrees[nextSubtree].bestBound = globalBest.length;
                    MPI_Send(&subtrees[nextSubtree], 1, subtreeType, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
                    nextSubtree++;
                    activeWorkers++;
                }
            }
        }

        // Send termination
        Subtree dummy;
        for (int w = 1; w < size; ++w) {
            MPI_Send(&dummy, 1, subtreeType, w, TAG_DONE, MPI_COMM_WORLD);
        }

        // Drain remaining bound messages
        int flag;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_BOUND, MPI_COMM_WORLD, &flag, &status);
        while (flag) {
            int newBound;
            MPI_Recv(&newBound, 1, MPI_INT, MPI_ANY_SOURCE, TAG_BOUND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_BOUND, MPI_COMM_WORLD, &flag, &status);
        }

        double endTime = MPI_Wtime();
        double elapsedMs = (endTime - startTime) * 1000.0;

        std::cout << "\n=== Results ===" << '\n';
        std::cout << "Solution: " << globalBest.toString() << '\n';
        std::cout << "Length: " << globalBest.length << '\n';
        std::cout << "Total time: " << std::fixed << std::setprecision(2) << elapsedMs << " ms" << '\n';
        std::cout << "Total nodes: " << totalNodes << '\n';

        if (order <= 14 && globalBest.length == OPTIMAL_LENGTHS[order]) {
            std::cout << "*** OPTIMAL SOLUTION FOUND ***" << '\n';
        }

    } else {
        // Worker process
        int localBound = INT_MAX;

        while (true) {
            Subtree subtree;
            MPI_Status status;
            MPI_Recv(&subtree, 1, subtreeType, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_DONE) {
                break;
            }

            localBound = subtree.bestBound;

            SubtreeSolverWithBounds solver(order, subtree, &localBound, rank, size);
            solver.solve();

            Result result = solver.getResult();
            MPI_Send(&result, 1, resultType, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }

    MPI_Type_free(&subtreeType);
    MPI_Type_free(&resultType);
    MPI_Finalize();

    return 0;
}
