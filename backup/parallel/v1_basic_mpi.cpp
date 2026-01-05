/**
 * Golomb Ruler Solver - Parallel Version 1: Basic MPI
 *
 * Architecture:
 * - Master (rank 0): Generates subtrees, distributes to workers, collects results
 * - Workers (rank 1-N): Receive subtrees, solve using sequential algorithm, return best
 *
 * Distribution: Static round-robin assignment of subtrees to workers
 *
 * Known limitations:
 * - Load imbalance: some subtrees are much larger than others
 * - No bound sharing: workers use outdated bounds
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

// Subtree structure for distribution
struct Subtree {
    int marks[MAX_ORDER];      // Prefix marks
    int markCount;             // Number of marks in prefix
    unsigned char usedDiffs[MAX_LENGTH / 8 + 1];  // Bitset as bytes
    int bestBound;             // Initial upper bound

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

// Forward declarations
void printStats(const SearchStats& stats, int order);

// Sequential solver for a subtree
class SubtreeSolver {
private:
    int order;
    int marks[MAX_ORDER];
    int markCount;
    std::bitset<MAX_LENGTH> usedDiffs;
    GolombRuler bestSolution;
    uint64_t nodesExplored;
    int initialBound;

public:
    SubtreeSolver(int n, const Subtree& subtree)
        : order(n), nodesExplored(0), initialBound(subtree.bestBound) {

        // Copy prefix
        markCount = subtree.markCount;
        for (int i = 0; i < markCount; ++i) {
            marks[i] = subtree.marks[i];
        }

        // Copy used differences
        usedDiffs.reset();
        for (int d = 0; d < MAX_LENGTH; ++d) {
            if (subtree.testDiff(d)) {
                usedDiffs.set(d);
            }
        }

        bestSolution.length = initialBound;
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

private:
    void branchAndBound(int depth) {
        nodesExplored++;

        if (depth == order) {
            int length = marks[markCount - 1];
            if (length < bestSolution.length) {
                std::vector<int> v(marks, marks + markCount);
                bestSolution = GolombRuler(v);
            }
            return;
        }

        int lastMark = marks[markCount - 1];
        int maxPos = bestSolution.length - 1;

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            int remainingMarks = order - depth - 1;
            if (pos + remainingMarks >= bestSolution.length) {
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

// Generate subtrees with given prefix depth
std::vector<Subtree> generateSubtrees(int order, int prefixDepth, int bound) {
    std::vector<Subtree> subtrees;

    // Recursive generator
    std::function<void(Subtree&, int)> generate = [&](Subtree& current, int depth) {
        if (depth == prefixDepth) {
            subtrees.push_back(current);
            return;
        }

        int lastMark = current.marks[current.markCount - 1];
        int maxPos = bound - 1;

        // Symmetry breaking for second mark
        if (depth == 1) {
            maxPos = std::min(maxPos, bound / 2);
        }

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            // Check remaining space
            int remaining = order - depth - 1;
            if (pos + remaining >= bound) break;

            // Check differences
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

// Create MPI datatype for Subtree
MPI_Datatype createSubtreeType() {
    MPI_Datatype subtreeType;

    int blocklengths[] = {MAX_ORDER, 1, MAX_LENGTH / 8 + 1, 1};
    MPI_Aint displacements[4];
    displacements[0] = offsetof(Subtree, marks);
    displacements[1] = offsetof(Subtree, markCount);
    displacements[2] = offsetof(Subtree, usedDiffs);
    displacements[3] = offsetof(Subtree, bestBound);

    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_UNSIGNED_CHAR, MPI_INT};

    MPI_Type_create_struct(4, blocklengths, displacements, types, &subtreeType);
    MPI_Type_commit(&subtreeType);

    return subtreeType;
}

// Create MPI datatype for Result
MPI_Datatype createResultType() {
    MPI_Datatype resultType;

    int blocklengths[] = {MAX_ORDER, 1, 1, 1};
    MPI_Aint displacements[4];
    displacements[0] = offsetof(Result, marks);
    displacements[1] = offsetof(Result, length);
    displacements[2] = offsetof(Result, order);
    displacements[3] = offsetof(Result, nodesExplored);

    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_INT, MPI_UINT64_T};

    MPI_Type_create_struct(4, blocklengths, displacements, types, &resultType);
    MPI_Type_commit(&resultType);

    return resultType;
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

    // Create MPI datatypes
    MPI_Datatype subtreeType = createSubtreeType();
    MPI_Datatype resultType = createResultType();

    Timer timer;
    double startTime = MPI_Wtime();

    if (rank == 0) {
        // Master process
        std::cout << "=== Golomb Ruler Solver - MPI v1: Basic ===" << '\n';
        std::cout << "Order: " << order << ", Processes: " << size << '\n';
        std::cout << "Prefix depth: " << prefixDepth << '\n';

        // Get initial bound
        int bound = greedyBound(order);
        std::cout << "Initial bound (greedy): " << bound << '\n';

        // Generate subtrees
        std::vector<Subtree> subtrees = generateSubtrees(order, prefixDepth, bound);
        std::cout << "Subtrees generated: " << subtrees.size() << '\n';

        // Distribute subtrees to workers (round-robin)
        int nextSubtree = 0;
        int activeWorkers = 0;

        // Initial distribution
        for (int w = 1; w < size && nextSubtree < static_cast<int>(subtrees.size()); ++w) {
            MPI_Send(&subtrees[nextSubtree], 1, subtreeType, w, TAG_WORK, MPI_COMM_WORLD);
            nextSubtree++;
            activeWorkers++;
        }

        // Collect results and distribute remaining work
        GolombRuler globalBest;
        globalBest.length = bound;
        uint64_t totalNodes = 0;

        while (activeWorkers > 0) {
            Result result;
            MPI_Status status;
            MPI_Recv(&result, 1, resultType, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);

            activeWorkers--;
            totalNodes += result.nodesExplored;

            // Update global best
            if (result.length < globalBest.length) {
                globalBest.length = result.length;
                globalBest.order = result.order;
                globalBest.marks.assign(result.marks, result.marks + result.order);
            }

            // Send more work if available
            if (nextSubtree < static_cast<int>(subtrees.size())) {
                // Update bound in subtree
                subtrees[nextSubtree].bestBound = globalBest.length;
                MPI_Send(&subtrees[nextSubtree], 1, subtreeType, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
                nextSubtree++;
                activeWorkers++;
            }
        }

        // Send termination signal to all workers
        Subtree dummy;
        for (int w = 1; w < size; ++w) {
            MPI_Send(&dummy, 1, subtreeType, w, TAG_DONE, MPI_COMM_WORLD);
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
        while (true) {
            Subtree subtree;
            MPI_Status status;
            MPI_Recv(&subtree, 1, subtreeType, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_DONE) {
                break;
            }

            // Solve subtree
            SubtreeSolver solver(order, subtree);
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
