/**
 * Golomb Ruler Solver - Parallel Version 6: MPI Ring Shifter
 *
 * Uses MPI Cartesian topology (1D ring) for bound sharing:
 * - MPI_Cart_create with wrap-around for ring topology
 * - MPI_Cart_shift to get left/right neighbors
 * - Non-blocking token passing for bound propagation
 * - Token injection when finding better solutions
 *
 * Advantages over Hypercube (v2-v4):
 * - Simpler: only 2 neighbors (left/right) instead of log2(P)
 * - Less bandwidth: 1 message per update instead of log2(P)
 * - More elegant: uses MPI topology functions
 *
 * Trade-off: O(P) propagation time vs O(log P) for hypercube
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

// MPI tags for work distribution (MPI_COMM_WORLD)
const int TAG_WORK = 1;
const int TAG_RESULT = 2;
const int TAG_DONE = 3;

// MPI tag for ring bound passing (ring_comm)
const int TAG_RING_BOUND = 6;

// How often to check for new bounds (every N nodes)
const int CHECK_INTERVAL = 100000;

// Subtree structure for work distribution
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

// Result structure for returning solutions
struct Result {
    int marks[MAX_ORDER];
    int length;
    int order;
    uint64_t nodesExplored;
    double timeMs;
};

/**
 * RingCommunicator - Manages MPI Cartesian ring topology
 *
 * Creates a 1D ring with wrap-around where each process has
 * exactly 2 neighbors: left (source) and right (destination)
 */
class RingCommunicator {
private:
    MPI_Comm ring_comm;
    int left_neighbor;
    int right_neighbor;
    int ring_rank;
    int ring_size;

public:
    RingCommunicator() : ring_comm(MPI_COMM_NULL), left_neighbor(-1),
                          right_neighbor(-1), ring_rank(-1), ring_size(0) {}

    void initialize(int world_size) {
        int dims[1] = {world_size};
        int periods[1] = {1};  // Wrap-around enabled (ring)
        int reorder = 0;       // Keep original ranking

        MPI_CHECK(MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &ring_comm));
        MPI_CHECK(MPI_Cart_shift(ring_comm, 0, 1, &left_neighbor, &right_neighbor));
        MPI_CHECK(MPI_Comm_rank(ring_comm, &ring_rank));
        MPI_CHECK(MPI_Comm_size(ring_comm, &ring_size));
    }

    // Must be called before MPI_Finalize
    void cleanup() {
        if (ring_comm != MPI_COMM_NULL) {
            MPI_Comm_free(&ring_comm);
            ring_comm = MPI_COMM_NULL;
        }
    }

    ~RingCommunicator() {
        // Do not free here - must be done before MPI_Finalize via cleanup()
    }

    MPI_Comm getComm() const { return ring_comm; }
    int getLeft() const { return left_neighbor; }
    int getRight() const { return right_neighbor; }
    int getRank() const { return ring_rank; }
    int getSize() const { return ring_size; }
};

/**
 * RingBoundManager - Non-blocking bound sharing via ring topology
 *
 * Token passing logic:
 * - When process finds better bound: inject into ring (send to right)
 * - When receiving from left: if better, update and forward; else absorb
 * - All communication is non-blocking for overlap with computation
 */
class RingBoundManager {
private:
    RingCommunicator& ring;
    int currentBound;
    uint64_t messagesReceived;
    uint64_t messagesSent;

    // Persistent send buffer (must remain valid until Isend completes)
    int sendBuffer;
    MPI_Request sendRequest;

public:
    RingBoundManager(RingCommunicator& r, int initialBound)
        : ring(r), currentBound(initialBound),
          messagesReceived(0), messagesSent(0),
          sendBuffer(0), sendRequest(MPI_REQUEST_NULL) {}

    ~RingBoundManager() {
        // Wait for any pending send to complete
        if (sendRequest != MPI_REQUEST_NULL) {
            MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
        }
    }

    /**
     * Non-blocking check for incoming bound from left neighbor
     * Returns true if bound was improved
     */
    bool checkIncomingBound() {
        int flag;
        MPI_Status status;
        MPI_Iprobe(ring.getLeft(), TAG_RING_BOUND, ring.getComm(), &flag, &status);

        if (flag) {
            int receivedBound;
            MPI_Recv(&receivedBound, 1, MPI_INT, ring.getLeft(),
                     TAG_RING_BOUND, ring.getComm(), MPI_STATUS_IGNORE);
            messagesReceived++;

            if (receivedBound < currentBound) {
                // Improvement: update local and forward to right
                currentBound = receivedBound;
                forwardBound(receivedBound);
                return true;
            }
            // Absorb: received bound is not better, don't forward
        }
        return false;
    }

    /**
     * Inject new bound into ring when solution is found
     * Only sends if new bound is better than current
     */
    void injectBound(int newBound) {
        if (newBound < currentBound) {
            currentBound = newBound;
            forwardBound(newBound);
        }
    }

    /**
     * Update bound without forwarding (used by master when receiving results)
     */
    void updateBound(int newBound) {
        if (newBound < currentBound) {
            currentBound = newBound;
        }
    }

    int getCurrentBound() const { return currentBound; }
    uint64_t getMessagesReceived() const { return messagesReceived; }
    uint64_t getMessagesSent() const { return messagesSent; }

private:
    /**
     * Forward bound to right neighbor (non-blocking)
     */
    void forwardBound(int bound) {
        // Wait for any previous send to complete before reusing buffer
        if (sendRequest != MPI_REQUEST_NULL) {
            MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
        }

        sendBuffer = bound;
        MPI_Isend(&sendBuffer, 1, MPI_INT, ring.getRight(),
                  TAG_RING_BOUND, ring.getComm(), &sendRequest);
        messagesSent++;
    }
};

/**
 * RingSubtreeSolver - Branch and bound solver with ring-based bound sharing
 */
class RingSubtreeSolver {
private:
    int order;
    int marks[MAX_ORDER];
    int markCount;
    std::bitset<MAX_LENGTH> usedDiffs;
    GolombRuler bestSolution;
    uint64_t nodesExplored;
    RingBoundManager* boundManager;

public:
    RingSubtreeSolver(int n, const Subtree& subtree, RingBoundManager* bm)
        : order(n), nodesExplored(0), boundManager(bm) {

        markCount = subtree.markCount;
        for (int i = 0; i < markCount; ++i) marks[i] = subtree.marks[i];

        usedDiffs.reset();
        for (int d = 0; d < MAX_LENGTH; ++d)
            if (subtree.testDiff(d)) usedDiffs.set(d);

        bestSolution.length = boundManager->getCurrentBound();
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

        // Periodic check for ring token
        if (nodesExplored % CHECK_INTERVAL == 0) {
            if (boundManager->checkIncomingBound()) {
                // Update local best if ring gave us better bound
                if (boundManager->getCurrentBound() < bestSolution.length) {
                    bestSolution.length = boundManager->getCurrentBound();
                }
            }
        }

        // Terminal case: complete ruler
        if (depth == order) {
            int length = marks[markCount - 1];
            if (length < bestSolution.length) {
                std::vector<int> v(marks, marks + markCount);
                bestSolution = GolombRuler(v);

                // Inject improved bound into ring
                boundManager->injectBound(length);
            }
            return;
        }

        int lastMark = marks[markCount - 1];
        int currentBound = boundManager->getCurrentBound();
        int maxPos = std::min(bestSolution.length - 1, currentBound - 1);

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            // Pruning: remaining marks would exceed bound
            if (pos + (order - depth - 1) >= currentBound) continue;

            // Check all differences against existing marks
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
                // Add mark and recurse
                marks[markCount++] = pos;
                for (int i = 0; i < newDiffCount; ++i) usedDiffs.set(newDiffs[i]);

                branchAndBound(depth + 1);

                // Backtrack
                markCount--;
                for (int i = 0; i < newDiffCount; ++i) usedDiffs.reset(newDiffs[i]);

                // Update maxPos if bound improved during recursion
                currentBound = boundManager->getCurrentBound();
                maxPos = std::min(maxPos, currentBound - 1);
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
        // Symmetry breaking: when placing 2nd mark, only try up to bound/2
        int maxPos = (depth == 1) ? std::min(bound - 1, bound / 2) : bound - 1;

        for (int pos = lastMark + 1; pos <= maxPos; ++pos) {
            if (pos + (order - depth - 1) >= bound) break;

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

// Drain remaining ring messages before shutdown
void drainRingMessages(RingCommunicator& ring) {
    int flag;
    MPI_Status status;
    while (true) {
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_RING_BOUND, ring.getComm(), &flag, &status);
        if (!flag) break;
        int tmp;
        MPI_Recv(&tmp, 1, MPI_INT, MPI_ANY_SOURCE, TAG_RING_BOUND,
                 ring.getComm(), MPI_STATUS_IGNORE);
    }
}

// CSV output
void appendRingCSV(const std::string& filename, int order, int procs,
                   double timeMs, double seqTime, uint64_t nodes,
                   uint64_t ringMsgs, const GolombRuler& solution) {
    std::ifstream check(filename);
    bool header = !check.good();
    check.close();

    std::ofstream f(filename, std::ios::app);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open " << filename << '\n';
        return;
    }

    if (header) {
        f << "version,order,procs,time_ms,speedup,efficiency,nodes,ring_msgs,solution,length\n";
    }

    double speedup = (seqTime > 0) ? seqTime / timeMs : 0;
    double efficiency = speedup / procs;

    f << 6 << "," << order << "," << procs << ","
      << std::fixed << std::setprecision(2) << timeMs << ","
      << std::setprecision(2) << speedup << ","
      << std::setprecision(3) << efficiency << ","
      << nodes << "," << ringMsgs << ","
      << "\"" << solution.toString() << "\"," << solution.length << "\n";

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

    // Auto-select prefix depth based on order
    int prefixDepth = (order <= 8) ? 3 : (order <= 10) ? 4 : 5;
    std::string csvFile;
    double seqTime = 0;

    // Parse command line options
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

    // Create ring topology (all processes participate)
    RingCommunicator ring;
    ring.initialize(size);

    // Create MPI datatypes
    MPI_Datatype subtreeType = createSubtreeType();
    MPI_Datatype resultType = createResultType();

    double startTime = MPI_Wtime();

    if (rank == 0) {
        // ========== MASTER ==========
        std::cout << "=== Golomb Ruler Solver - MPI v6: Ring Shifter ===\n";
        std::cout << "Order: " << order << ", Processes: " << size << "\n";
        std::cout << "Prefix depth: " << prefixDepth << "\n";
        std::cout << "Ring topology: left=" << ring.getLeft() << ", right=" << ring.getRight() << "\n";

        int bound = computeGreedyBound(order);
        std::cout << "Initial bound: " << bound << "\n";

        std::vector<Subtree> subtrees = generateSubtrees(order, prefixDepth, bound);
        std::cout << "Subtrees: " << subtrees.size() << "\n";

        // Initialize ring bound manager
        RingBoundManager boundManager(ring, bound);

        // Dynamic work distribution
        size_t nextSubtree = 0;
        int activeWorkers = 0;

        // Initial batch distribution
        for (int w = 1; w < size && nextSubtree < subtrees.size(); ++w) {
            subtrees[nextSubtree].bestBound = boundManager.getCurrentBound();
            MPI_Send(&subtrees[nextSubtree], 1, subtreeType, w, TAG_WORK, MPI_COMM_WORLD);
            nextSubtree++;
            activeWorkers++;
        }

        GolombRuler globalBest;
        globalBest.length = bound;
        uint64_t totalNodes = 0;
        double maxWorkerTime = 0, minWorkerTime = 1e9;

        // Main distribution loop
        while (activeWorkers > 0) {
            // Check ring for incoming bounds
            boundManager.checkIncomingBound();

            // Check for results from workers
            int flag;
            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &flag, &status);

            if (flag) {
                Result result;
                MPI_Recv(&result, 1, resultType, status.MPI_SOURCE,
                         TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                activeWorkers--;
                totalNodes += result.nodesExplored;
                maxWorkerTime = std::max(maxWorkerTime, result.timeMs);
                minWorkerTime = std::min(minWorkerTime, result.timeMs);

                // Update best solution if improved
                if (result.length < globalBest.length && result.length < INT_MAX) {
                    globalBest.length = result.length;
                    globalBest.order = result.order;
                    globalBest.marks.assign(result.marks, result.marks + result.order);

                    // Inject improved bound into ring
                    boundManager.injectBound(result.length);
                }

                // Send more work with current bound
                if (nextSubtree < subtrees.size()) {
                    subtrees[nextSubtree].bestBound = boundManager.getCurrentBound();
                    MPI_Send(&subtrees[nextSubtree], 1, subtreeType,
                             status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
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

        // Barrier to ensure all workers have finished before draining
        MPI_Barrier(MPI_COMM_WORLD);

        // Drain remaining ring messages
        drainRingMessages(ring);

        double endTime = MPI_Wtime();
        double elapsedMs = (endTime - startTime) * 1000.0;

        uint64_t totalRingMsgs = boundManager.getMessagesSent() + boundManager.getMessagesReceived();

        std::cout << "\n=== Results ===\n";
        std::cout << "Solution: " << globalBest.toString() << "\n";
        std::cout << "Length: " << globalBest.length << "\n";
        std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsedMs << " ms\n";
        std::cout << "Nodes: " << totalNodes << "\n";
        std::cout << "Ring messages (master): " << totalRingMsgs << "\n";

        if (seqTime > 0) {
            std::cout << "Speedup: " << std::setprecision(2) << seqTime/elapsedMs << "x\n";
            std::cout << "Efficiency: " << std::setprecision(1) << 100*seqTime/(elapsedMs*size) << "%\n";
        }

        if (order <= 14 && globalBest.length == OPTIMAL_LENGTHS[order]) {
            std::cout << "*** OPTIMAL ***\n";
        }

        if (!csvFile.empty()) {
            appendRingCSV(csvFile, order, size, elapsedMs, seqTime, totalNodes, totalRingMsgs, globalBest);
        }

    } else {
        // ========== WORKER ==========
        RingBoundManager boundManager(ring, INT_MAX);

        while (true) {
            // Check ring tokens while waiting for work
            boundManager.checkIncomingBound();

            // Non-blocking check for work from master
            int flag;
            MPI_Status status;
            MPI_Iprobe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

            if (flag) {
                if (status.MPI_TAG == TAG_DONE) {
                    Subtree dummy;
                    MPI_Recv(&dummy, 1, subtreeType, 0, TAG_DONE,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    break;
                }

                Subtree subtree;
                MPI_Recv(&subtree, 1, subtreeType, 0, TAG_WORK,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Update bound manager with subtree's bound
                boundManager.updateBound(subtree.bestBound);

                double workerStart = MPI_Wtime();

                RingSubtreeSolver solver(order, subtree, &boundManager);
                solver.solve();

                Result result = solver.getResult();
                result.timeMs = (MPI_Wtime() - workerStart) * 1000.0;
                MPI_Send(&result, 1, resultType, 0, TAG_RESULT, MPI_COMM_WORLD);
            }
        }

        // Barrier before ring drain
        MPI_Barrier(MPI_COMM_WORLD);

        // Drain remaining ring messages
        drainRingMessages(ring);
    }

    // Cleanup ring communicator before MPI_Finalize
    ring.cleanup();

    MPI_Type_free(&subtreeType);
    MPI_Type_free(&resultType);
    MPI_Finalize();

    return 0;
}
