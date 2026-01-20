# Algorithm Documentation

## Golomb Ruler Problem

A **Golomb ruler** of order n is a set of n integers (marks) where all pairwise differences are unique.

```
Order 4 optimal ruler: [0, 1, 3, 6]
Differences: 1-0=1, 3-0=3, 6-0=6, 3-1=2, 6-1=5, 6-3=3
All unique: {1, 2, 3, 5, 6} ✓
```

**Goal**: Find the shortest ruler for a given order (NP-hard).

## Branch and Bound Algorithm

```
function branchAndBound(marks[], depth, usedDiffs):
    if depth == order:
        if length < bestLength:
            bestLength = length
            bestSolution = marks
        return

    for pos = marks[depth-1]+1 to upperBound:
        if canPlace(pos, marks, usedDiffs):
            marks[depth] = pos
            updateDiffs(usedDiffs, pos, marks)
            branchAndBound(marks, depth+1, usedDiffs)
            revertDiffs(usedDiffs, pos, marks)
```

## Key Optimizations

### 1. Bitset Difference Tracking
O(1) lookup instead of O(n) scanning:
```cpp
BitSet256 usedDiffs;
if (usedDiffs.test(diff)) return false;  // Collision
usedDiffs.set(diff);
```

### 2. Symmetry Breaking
Constraint: `marks[1] <= bestLength / 2`
- Halves the search space
- Exploits ruler symmetry (reversible)

### 3. Greedy Initial Bound
Compute greedy solution before search:
```cpp
int greedyBound = computeGreedySolution(order);
bestLength = greedyBound;  // Tighter initial bound
```

### 4. Early Pruning
```cpp
if (currentLength + minRemaining >= bestLength)
    return;  // Cannot improve, prune
```

### 5. AVX2 SIMD
Vectorized difference checking (8 integers per instruction):
```cpp
__m256i diffs = _mm256_sub_epi32(pos_vec, marks_vec);
// Check 8 differences simultaneously
```

## Complexity

- **Time**: O(L^n) worst case, heavily pruned in practice
- **Space**: O(n) for marks + O(L) for difference bitset
- **Parallel speedup**: Near-linear with proper load balancing
