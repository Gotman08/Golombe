/**
 * Golomb Ruler Solver - Parallel Version 5: Work Stealing
 *
 * Architecture: Fully decentralized P2P work stealing
 * - Each process has a local work deque
 * - Idle processes steal from random victims
 * - Dijkstra token algorithm for termination detection
 * - GRASP for initial bound (optional)
 *
 * Expected efficiency: >90% (vs 60-75% for Master-Worker)
 *
 * Based on: Blumofe & Leiserson (1999) - Scheduling Multithreaded Computations by Work Stealing
 */

#include <mpi.h>
#include "golomb.hpp"
#include "greedy.hpp"
#include "grasp.hpp"
#include <iostream>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <random>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <bitset>
#include <functional>

// MPI Tags for Work Stealing Protocol
namespace Tags {
    const int STEAL_REQUEST = 10;    // Request work from victim
    const int STEAL_RESPONSE = 11;   // Response with subtree
    const int STEAL_EMPTY = 12;      // Victim has no work
    const int BOUND_UPDATE = 13;     // Broadcast new bound
    const int TOKEN = 14;            // Dijkstra termination token
    const int TERMINATE = 15;        // Global termination signal
}

// NOTE: Dijkstra token algorithm was removed - using MPI_Allreduce for termination
// which is simpler and sufficient for this use case.

// Constants - Tuned for performance
const int CHECK_INTERVAL = 10000;     // Check messages every N nodes - CRITICAL for bound updates
const int STEAL_BATCH_SIZE = 8;       // Steal multiple tasks at once
const int MIN_TASKS_TO_SHARE = 4;     // Don't share if less than this
const int STEAL_COOLDOWN = 5000;      // Iterations between steal attempts
const int WORK_BATCH_SIZE = 20;       // Process subtrees in batches
const int TERMINATION_CHECK_INTERVAL = 200;  // Check termination less frequently
const int BOUND_CHECK_INTERVAL = 5000; // Check bounds more frequently

// Subtree structure (same as v3 for compatibility)
struct Subtree {
    int marks[MAX_ORDER];
    int markCount;
    unsigned char usedDiffs[MAX_LENGTH / 8 + 1];
    int bound;

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

    // Serialization size
    static constexpr int serializedSize() {
        return sizeof(int) * (MAX_ORDER + 2) + (MAX_LENGTH / 8 + 1);
    }
};

// Statistics tracking
struct WorkerStats {
    uint64_t nodesExplored = 0;
    uint64_t nodesPruned = 0;
    uint64_t subtreesProcessed = 0;
    uint64_t stealAttempts = 0;
    uint64_t stealSuccesses = 0;
    uint64_t workShared = 0;
    double computeTime = 0;
    double idleTime = 0;
};

/**
 * Thread-safe work deque for work stealing
 */
class WorkDeque {
private:
    std::deque<Subtree> tasks;
    mutable std::mutex mtx;

public:
    // Worker pushes to bottom (LIFO for locality)
    void pushBottom(const Subtree& task) {
        std::lock_guard<std::mutex> lock(mtx);
        tasks.push_back(task);
    }

    // Worker pops from bottom (LIFO)
    bool popBottom(Subtree& task) {
        std::lock_guard<std::mutex> lock(mtx);
        if (tasks.empty()) return false;
        task = tasks.back();
        tasks.pop_back();
        return true;
    }

    // Thief steals from top (FIFO - larger subtrees)
    bool stealTop(Subtree& task) {
        std::lock_guard<std::mutex> lock(mtx);
        if (tasks.empty()) return false;
        task = tasks.front();
        tasks.pop_front();
        return true;
    }

    // Steal multiple tasks at once
    std::vector<Subtree> stealBatch(int maxCount) {
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<Subtree> stolen;
        int count = std::min(maxCount, (int)tasks.size() / 2);
        for (int i = 0; i < count && !tasks.empty(); i++) {
            stolen.push_back(tasks.front());
            tasks.pop_front();
        }
        return stolen;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx);
        return tasks.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return tasks.size();
    }
};

/**
 * Work Stealing Solver
 */
class WorkStealingSolver {
private:
    int order;
    int rank, worldSize;
    int globalBound;
    GolombRuler bestSolution;
    WorkDeque workDeque;
    WorkerStats stats;

    // Termination state
    bool terminated;
    bool idle;

    // Random generator for victim selection
    std::mt19937 rng;

    // MPI request tracking
    std::vector<MPI_Request> pendingRequests;

public:
    WorkStealingSolver(int n, int bound, int r, int s)
        : order(n), rank(r), worldSize(s), globalBound(bound),
          terminated(false), idle(false) {
        bestSolution.length = bound;
        rng.seed(rank + std::chrono::system_clock::now().time_since_epoch().count());
    }

    /**
     * Initialize work distribution
     * Rank 0 generates initial subtrees and distributes evenly
     */
    void initialize(int prefixDepth) {
        if (rank == 0) {
            std::cout << "Generating initial subtrees..." << std::endl;
            auto subtrees = generateSubtrees(order, prefixDepth, globalBound);
            std::cout << "Generated " << subtrees.size() << " subtrees" << std::endl;

            // Count subtrees per rank
            std::vector<int> counts(worldSize, 0);
            for (size_t i = 0; i < subtrees.size(); i++) {
                counts[i % worldSize]++;
            }

            // Send counts first
            for (int r = 1; r < worldSize; r++) {
                MPI_Send(&counts[r], 1, MPI_INT, r, 0, MPI_COMM_WORLD);
            }

            // Distribute subtrees round-robin
            for (size_t i = 0; i < subtrees.size(); i++) {
                int targetRank = i % worldSize;
                if (targetRank == 0) {
                    workDeque.pushBottom(subtrees[i]);
                } else {
                    sendSubtree(subtrees[i], targetRank);
                }
            }
        } else {
            // Receive count first
            int count;
            MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Receive subtrees
            for (int i = 0; i < count; i++) {
                Subtree subtree;
                receiveSubtree(subtree, 0);
                workDeque.pushBottom(subtree);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Initial distribution complete. Local queue: " << workDeque.size() << std::endl;
        }
    }

    /**
     * Main work stealing loop
     * Optimized version with reduced overhead
     */
    void run() {
        auto startTime = std::chrono::high_resolution_clock::now();
        auto lastProgressTime = startTime;
        int idleIterations = 0;
        int stealCooldown = 0;
        int msgCheckCounter = 0;
        int terminationCheckCounter = 0;

        while (!terminated) {
            // Phase 1: Process local work in large batches (minimize overhead)
            int processed = 0;
            while (!workDeque.empty() && processed < WORK_BATCH_SIZE) {
                Subtree task;
                if (workDeque.popBottom(task)) {
                    processSubtree(task);
                    stats.subtreesProcessed++;
                    processed++;
                }
            }

            if (processed > 0) {
                idle = false;
                idleIterations = 0;
            } else {
                idle = true;
                idleIterations++;
            }

            // Phase 2: Handle incoming messages less frequently when busy
            msgCheckCounter++;
            if (idle || msgCheckCounter >= 10) {
                handleIncomingMessages();
                msgCheckCounter = 0;
            }

            // Phase 3: If idle, try to steal work (with cooldown)
            if (idle && stealCooldown <= 0 && worldSize > 1) {
                trySteal();
                stealCooldown = STEAL_COOLDOWN;
            }
            if (stealCooldown > 0) stealCooldown -= (idle ? 100 : 1);  // Faster cooldown when idle

            // Phase 4: Collective termination check (very infrequent)
            terminationCheckCounter++;
            if (idle && idleIterations >= TERMINATION_CHECK_INTERVAL && terminationCheckCounter >= 10) {
                int localDone = workDeque.empty() ? 1 : 0;
                int globalDone = 0;
                MPI_Allreduce(&localDone, &globalDone, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

                if (globalDone == 1) {
                    terminated = true;
                } else {
                    idleIterations = 0;
                }
                terminationCheckCounter = 0;
            }

            // Progress reporting (rank 0 only, very infrequent)
            if (rank == 0 && !terminated && stats.subtreesProcessed % 500 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration<double>(now - lastProgressTime).count();
                if (elapsed > 30.0) {
                    printProgress();
                    lastProgressTime = now;
                }
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        stats.computeTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    }

    /**
     * Collect and print final results
     */
    void collectResults() {
        // Gather statistics
        uint64_t totalNodes, totalPruned, totalSubtrees;
        MPI_Reduce(&stats.nodesExplored, &totalNodes, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats.nodesPruned, &totalPruned, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats.subtreesProcessed, &totalSubtrees, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

        // Gather best solution
        struct {
            int length;
            int rank;
        } localBest = {bestSolution.length, rank}, globalBest;

        MPI_Allreduce(&localBest, &globalBest, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        // Get solution from winner
        if (globalBest.rank != 0) {
            if (rank == globalBest.rank) {
                MPI_Send(bestSolution.marks.data(), order, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
            if (rank == 0) {
                std::vector<int> marks(order);
                MPI_Recv(marks.data(), order, MPI_INT, globalBest.rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                bestSolution = GolombRuler(marks);
            }
        }

        if (rank == 0) {
            std::cout << "\n=== Results ===" << std::endl;
            std::cout << "Solution: [";
            for (size_t i = 0; i < bestSolution.marks.size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << bestSolution.marks[i];
            }
            std::cout << "]" << std::endl;
            std::cout << "Length: " << bestSolution.length << std::endl;
            std::cout << "Time: " << std::fixed << std::setprecision(2) << stats.computeTime << " ms" << std::endl;
            std::cout << "Nodes explored: " << totalNodes << std::endl;
            std::cout << "Nodes pruned: " << totalPruned << std::endl;
            std::cout << "Subtrees processed: " << totalSubtrees << std::endl;

            if (bestSolution.length == getOptimalLength(order)) {
                std::cout << "*** OPTIMAL ***" << std::endl;
            }
        }
    }

private:
    /**
     * Generate initial subtrees for distribution
     */
    std::vector<Subtree> generateSubtrees(int order, int depth, int bound) {
        std::vector<Subtree> subtrees;

        std::function<void(Subtree&, int)> generate = [&](Subtree& current, int d) {
            if (d == depth) {
                current.bound = bound;
                subtrees.push_back(current);
                return;
            }

            int lastMark = current.marks[current.markCount - 1];
            int maxPos = (d == 1) ? std::min(bound - 1, bound / 2) : bound - 1;

            for (int pos = lastMark + 1; pos <= maxPos; pos++) {
                if (pos + (order - d - 1) >= bound) break;

                bool valid = true;
                std::vector<int> newDiffs;
                for (int i = 0; i < current.markCount; i++) {
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
                    for (int diff : newDiffs) next.setDiff(diff);
                    generate(next, d + 1);
                }
            }
        };

        Subtree root;
        root.clearDiffs();
        root.marks[0] = 0;
        root.markCount = 1;
        generate(root, 1);

        return subtrees;
    }

    /**
     * Process a single subtree using Branch & Bound
     */
    void processSubtree(const Subtree& subtree) {
        // Initialize state from subtree
        int marks[MAX_ORDER];
        int markCount = subtree.markCount;
        std::bitset<MAX_LENGTH> usedDiffs;

        for (int i = 0; i < markCount; i++) marks[i] = subtree.marks[i];
        for (int d = 0; d < MAX_LENGTH; d++) {
            if (subtree.testDiff(d)) usedDiffs.set(d);
        }

        // Local copy of bound for this subtree - sync with global
        int localBound = globalBound;
        uint64_t localNodeCount = 0;

        // Branch and bound - optimized to check bounds frequently
        std::function<void(int)> solve = [&](int depth) {
            stats.nodesExplored++;
            localNodeCount++;

            // Check for bound updates frequently (critical for performance!)
            if (localNodeCount % BOUND_CHECK_INTERVAL == 0) {
                handleIncomingMessages();
                if (globalBound < localBound) {
                    localBound = globalBound;  // Update local bound
                }
            }

            if (depth == order) {
                int length = marks[depth - 1];
                if (length < localBound) {
                    localBound = length;
                    updateGlobalBound(length, marks);
                }
                return;
            }

            int lastMark = marks[depth - 1];
            int maxPos = localBound - (order - depth - 1) - 1;

            for (int pos = lastMark + 1; pos <= maxPos; pos++) {
                // Recheck bound frequently (might have been updated)
                if (localNodeCount % 1000 == 0 && globalBound < localBound) {
                    localBound = globalBound;
                    maxPos = localBound - (order - depth - 1) - 1;
                    if (pos > maxPos) break;
                }

                // Check validity
                bool valid = true;
                std::vector<int> newDiffs;
                newDiffs.reserve(depth);
                for (int i = 0; i < depth; i++) {
                    int diff = pos - marks[i];
                    if (diff >= MAX_LENGTH || usedDiffs.test(diff)) {
                        valid = false;
                        break;
                    }
                    newDiffs.push_back(diff);
                }

                if (valid) {
                    marks[depth] = pos;
                    for (int d : newDiffs) usedDiffs.set(d);

                    solve(depth + 1);

                    for (int d : newDiffs) usedDiffs.reset(d);
                } else {
                    stats.nodesPruned++;
                }
            }
        };

        solve(markCount);
    }

    /**
     * Update global bound and broadcast
     */
    void updateGlobalBound(int newBound, int* marks) {
        if (newBound < globalBound) {
            globalBound = newBound;
            bestSolution.marks.assign(marks, marks + order);
            bestSolution.length = newBound;

            // Broadcast to all other processes
            for (int r = 0; r < worldSize; r++) {
                if (r != rank) {
                    MPI_Request req;
                    MPI_Isend(&newBound, 1, MPI_INT, r, Tags::BOUND_UPDATE, MPI_COMM_WORLD, &req);
                    MPI_Request_free(&req);
                }
            }
        }
    }

    /**
     * Handle all incoming MPI messages
     */
    void handleIncomingMessages() {
        int flag;
        MPI_Status status;

        while (true) {
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
            if (!flag) break;

            switch (status.MPI_TAG) {
                case Tags::STEAL_REQUEST:
                    handleStealRequest(status.MPI_SOURCE);
                    break;

                case Tags::STEAL_RESPONSE:
                    handleStealResponse(status.MPI_SOURCE);
                    break;

                case Tags::STEAL_EMPTY:
                    handleStealEmpty(status.MPI_SOURCE);
                    break;

                case Tags::BOUND_UPDATE:
                    handleBoundUpdate(status.MPI_SOURCE);
                    break;

                case Tags::TOKEN:
                    // Token handling removed - using MPI_Allreduce for termination
                    {
                        int dummy;
                        MPI_Recv(&dummy, 1, MPI_INT, status.MPI_SOURCE, Tags::TOKEN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    break;

                case Tags::TERMINATE:
                    handleTerminate(status.MPI_SOURCE);
                    break;

                default:
                    // Unknown tag, consume it
                    char buf[1];
                    MPI_Recv(buf, 0, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    /**
     * Handle incoming steal request
     */
    void handleStealRequest(int thief) {
        int dummy;
        MPI_Recv(&dummy, 1, MPI_INT, thief, Tags::STEAL_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (workDeque.size() >= MIN_TASKS_TO_SHARE) {
            Subtree task;
            if (workDeque.stealTop(task)) {
                sendSubtree(task, thief);
                stats.workShared++;
                return;
            }
        }

        // Nothing to share
        int empty = 0;
        MPI_Send(&empty, 1, MPI_INT, thief, Tags::STEAL_EMPTY, MPI_COMM_WORLD);
    }

    /**
     * Handle incoming subtree (stolen work)
     */
    void handleStealResponse(int victim) {
        Subtree task;
        receiveSubtree(task, victim);
        workDeque.pushBottom(task);
        stats.stealSuccesses++;
        idle = false;
    }

    /**
     * Handle empty response from victim
     */
    void handleStealEmpty(int victim) {
        int dummy;
        MPI_Recv(&dummy, 1, MPI_INT, victim, Tags::STEAL_EMPTY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Will try another victim
    }

    /**
     * Handle bound update from another process
     */
    void handleBoundUpdate(int source) {
        int newBound;
        MPI_Recv(&newBound, 1, MPI_INT, source, Tags::BOUND_UPDATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (newBound < globalBound) {
            globalBound = newBound;
        }
    }

    /**
     * Handle global termination signal
     */
    void handleTerminate(int source) {
        int dummy;
        MPI_Recv(&dummy, 1, MPI_INT, source, Tags::TERMINATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        terminated = true;
    }

    /**
     * Try to steal work from a random victim
     */
    void trySteal() {
        stats.stealAttempts++;

        // Select random victim
        std::uniform_int_distribution<int> dist(0, worldSize - 2);
        int victim = dist(rng);
        if (victim >= rank) victim++;  // Skip self

        // Send steal request
        int request = 1;
        MPI_Send(&request, 1, MPI_INT, victim, Tags::STEAL_REQUEST, MPI_COMM_WORLD);
    }

    /**
     * Send subtree to another process
     */
    void sendSubtree(const Subtree& task, int dest) {
        // Validate buffer size to prevent out-of-bounds access
        if (task.markCount <= 0 || task.markCount > MAX_ORDER) {
            std::cerr << "Error: Invalid markCount in sendSubtree: " << task.markCount << std::endl;
            return;
        }

        // Send marks and count
        MPI_Send(&task.markCount, 1, MPI_INT, dest, Tags::STEAL_RESPONSE, MPI_COMM_WORLD);
        MPI_Send(task.marks, task.markCount, MPI_INT, dest, Tags::STEAL_RESPONSE, MPI_COMM_WORLD);
        MPI_Send(task.usedDiffs, MAX_LENGTH / 8 + 1, MPI_UNSIGNED_CHAR, dest, Tags::STEAL_RESPONSE, MPI_COMM_WORLD);
        MPI_Send(&task.bound, 1, MPI_INT, dest, Tags::STEAL_RESPONSE, MPI_COMM_WORLD);
    }

    /**
     * Receive subtree from another process
     */
    void receiveSubtree(Subtree& task, int source) {
        MPI_Recv(&task.markCount, 1, MPI_INT, source, Tags::STEAL_RESPONSE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(task.marks, task.markCount, MPI_INT, source, Tags::STEAL_RESPONSE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(task.usedDiffs, MAX_LENGTH / 8 + 1, MPI_UNSIGNED_CHAR, source, Tags::STEAL_RESPONSE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&task.bound, 1, MPI_INT, source, Tags::STEAL_RESPONSE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /**
     * Print progress (rank 0 only)
     */
    void printProgress() {
        std::cout << "[Progress] Nodes: " << stats.nodesExplored
                  << " | Subtrees: " << stats.subtreesProcessed
                  << " | Steals: " << stats.stealSuccesses << "/" << stats.stealAttempts
                  << " | Bound: " << globalBound
                  << " | Queue: " << workDeque.size()
                  << std::endl;
    }
};

/**
 * Calculate optimal prefix depth based on order and process count
 */
int calculatePrefixDepth(int order, int worldSize) {
    if (order <= 8) return 3;
    if (order <= 10) return 4;
    if (order <= 12) return 5;
    if (order <= 15) return 6;
    return 7;
}

/**
 * Main function
 */
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Parse arguments
    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <order> [--depth N] [--grasp] [--verbose]" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int order = std::atoi(argv[1]);
    int prefixDepth = calculatePrefixDepth(order, worldSize);
    bool useGrasp = false;
    bool verbose = false;

    for (int i = 2; i < argc; i++) {
        if (std::strcmp(argv[i], "--depth") == 0 && i + 1 < argc) {
            prefixDepth = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--grasp") == 0) {
            useGrasp = true;
        } else if (std::strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        }
    }

    // Calculate initial bound
    int initialBound;
    if (rank == 0) {
        std::cout << "=== Golomb Ruler Solver - MPI v5: Work Stealing ===" << std::endl;
        std::cout << "Order: " << order << std::endl;
        std::cout << "Processes: " << worldSize << std::endl;
        std::cout << "Prefix depth: " << prefixDepth << std::endl;

        if (useGrasp) {
            std::cout << "Running GRASP for initial bound..." << std::endl;
            auto graspResult = grasp::graspAuto(order, verbose);
            initialBound = graspResult.length;
            std::cout << "GRASP bound: " << initialBound << std::endl;
        } else {
            initialBound = computeGreedyBound(order);
            std::cout << "Greedy bound: " << initialBound << std::endl;
        }
    }

    // Broadcast initial bound
    MPI_Bcast(&initialBound, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Create solver and run
    WorkStealingSolver solver(order, initialBound, rank, worldSize);
    solver.initialize(prefixDepth);
    solver.run();
    solver.collectResults();

    MPI_Finalize();
    return 0;
}
