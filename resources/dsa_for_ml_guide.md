# DSA for ML Guide

Data structures and algorithms knowledge required for ML engineering roles — interview patterns, time complexity, and ML-specific applications.

---

## Why DSA Matters for ML Engineers

ML interviews at top companies (Google, Meta, Amazon, Apple, Anthropic, OpenAI) include 1–2 LeetCode-style coding rounds alongside ML-specific questions. You need to be comfortable with:

- Arrays, strings, hash maps — every ML engineer's bread and butter
- Recursion, DFS/BFS — used in tree-structured models and graph ML
- Sorting and searching — feature ranking, nearest-neighbour search
- Dynamic programming — sequence models, optimal substructure in RL

---

## Priority Topics for ML Engineers

| Topic | Interview Frequency | Used in ML Code |
|---|---|---|
| Arrays & Hashing | ⭐⭐⭐⭐⭐ | Feature lookup, deduplication |
| Two Pointers | ⭐⭐⭐⭐ | Sliding window metrics |
| Binary Search | ⭐⭐⭐⭐ | Hyperparameter bounds, threshold search |
| Sliding Window | ⭐⭐⭐⭐ | Rolling statistics in time series |
| Trees | ⭐⭐⭐ | Decision tree internals |
| Graphs / BFS/DFS | ⭐⭐⭐ | GNNs, Airflow DAG validation |
| Dynamic Programming | ⭐⭐⭐ | RL (Bellman), sequence alignment |
| Heap/Priority Queue | ⭐⭐⭐ | Top-K features, beam search |
| Sorting | ⭐⭐⭐⭐ | Feature importance ranking |

---

## Part 1: Arrays & Hashing

```python
# Two Sum — O(n) with hash map
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i

# Contains Duplicate — O(n)
def contains_duplicate(nums):
    return len(nums) != len(set(nums))

# Valid Anagram — O(n)
from collections import Counter
def is_anagram(s, t):
    return Counter(s) == Counter(t)

# Group Anagrams — O(n·k log k)
def group_anagrams(strs):
    groups = {}
    for s in strs:
        key = tuple(sorted(s))
        groups.setdefault(key, []).append(s)
    return list(groups.values())

# Top K Frequent Elements — O(n log k)
import heapq
def top_k_frequent(nums, k):
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
```

---

## Part 2: Two Pointers & Sliding Window

```python
# Valid Palindrome — O(n)
def is_palindrome(s):
    s = ''.join(c.lower() for c in s if c.isalnum())
    l, r = 0, len(s) - 1
    while l < r:
        if s[l] != s[r]: return False
        l += 1; r -= 1
    return True

# Max Sliding Window Sum — O(n)
def max_subarray_sum(nums, k):
    window = sum(nums[:k])
    best   = window
    for i in range(k, len(nums)):
        window += nums[i] - nums[i-k]
        best    = max(best, window)
    return best

# ML application: rolling mean for time series
def rolling_mean(values, window):
    result = []
    total  = sum(values[:window])
    result.append(total / window)
    for i in range(window, len(values)):
        total += values[i] - values[i-window]
        result.append(total / window)
    return result

# Longest Substring Without Repeating — O(n)
def length_of_longest_substring(s):
    seen = {}
    left = best = 0
    for right, ch in enumerate(s):
        if ch in seen and seen[ch] >= left:
            left = seen[ch] + 1
        seen[ch] = right
        best = max(best, right - left + 1)
    return best
```

---

## Part 3: Binary Search

```python
# Classic binary search — O(log n)
def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target: return mid
        if nums[mid] < target:  lo = mid + 1
        else:                   hi = mid - 1
    return -1

# Search in rotated sorted array — O(log n)
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target: return mid
        if nums[lo] <= nums[mid]:                     # left half sorted
            if nums[lo] <= target < nums[mid]: hi = mid - 1
            else:                              lo = mid + 1
        else:                                         # right half sorted
            if nums[mid] < target <= nums[hi]: lo = mid + 1
            else:                              hi = mid - 1
    return -1

# Find minimum threshold — binary search on answer (common ML pattern)
def find_min_threshold(values, target_count):
    """Find minimum threshold t such that sum(v > t for v in values) <= target_count."""
    lo, hi = min(values), max(values)
    while lo < hi:
        mid = (lo + hi) / 2
        if sum(v > mid for v in values) > target_count:
            lo = mid + 1e-9
        else:
            hi = mid
    return lo
```

---

## Part 4: Trees & Recursion

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val  = val
        self.left = left
        self.right = right

# Inorder traversal (returns sorted values for BST) — O(n)
def inorder(root):
    if not root: return []
    return inorder(root.left) + [root.val] + inorder(root.right)

# Max depth — O(n)
def max_depth(root):
    if not root: return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

# Level-order BFS — O(n)
from collections import deque
def level_order(root):
    if not root: return []
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result

# Lowest Common Ancestor — O(n)
def lca(root, p, q):
    if not root or root == p or root == q: return root
    left  = lca(root.left, p, q)
    right = lca(root.right, p, q)
    return root if left and right else left or right

# Validate BST — O(n)
def is_valid_bst(root, lo=float('-inf'), hi=float('inf')):
    if not root: return True
    if not (lo < root.val < hi): return False
    return (is_valid_bst(root.left, lo, root.val) and
            is_valid_bst(root.right, root.val, hi))
```

---

## Part 5: Graphs (BFS / DFS)

```python
from collections import defaultdict, deque

# BFS — shortest path in unweighted graph
def bfs(graph, start, end):
    queue    = deque([(start, [start])])
    visited  = {start}
    while queue:
        node, path = queue.popleft()
        if node == end: return path
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

# DFS — topological sort (used for Airflow DAG validation)
def topological_sort(graph, nodes):
    visited = set(); order = []
    def dfs(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
        order.append(node)
    for node in nodes:
        if node not in visited: dfs(node)
    return order[::-1]

# Detect cycle in directed graph (Airflow DAG validation)
def has_cycle(graph, nodes):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    def dfs(node):
        color[node] = GRAY
        for nb in graph.get(node, []):
            if color[nb] == GRAY: return True    # back edge = cycle
            if color[nb] == WHITE and dfs(nb):   return True
        color[node] = BLACK
        return False
    return any(dfs(n) for n in nodes if color[n] == WHITE)

# Number of connected components
def count_components(n, edges):
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b); graph[b].append(a)
    visited = set()
    def dfs(node):
        visited.add(node)
        for nb in graph[node]:
            if nb not in visited: dfs(nb)
    return sum(1 for i in range(n) if i not in visited and not dfs(i) is None or i not in visited)
```

---

## Part 6: Dynamic Programming

```python
# Fibonacci — O(n) with memoisation
from functools import lru_cache
@lru_cache(maxsize=None)
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)

# Longest Increasing Subsequence — O(n log n)
import bisect
def lis_length(nums):
    tails = []
    for n in nums:
        pos = bisect.bisect_left(tails, n)
        if pos == len(tails): tails.append(n)
        else:                 tails[pos] = n
    return len(tails)

# Coin Change — O(amount × len(coins))
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

# Edit Distance (Levenshtein) — used in NLP
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
```

---

## Part 7: Heaps & Priority Queues

```python
import heapq

# K largest elements — O(n log k)
def k_largest(nums, k):
    return heapq.nlargest(k, nums)

# Merge K sorted lists — O(n log k)
def merge_k_lists(lists):
    result = []
    heap   = [(lst[0], i, 0) for i, lst in enumerate(lists) if lst]
    heapq.heapify(heap)
    while heap:
        val, list_i, elem_i = heapq.heappop(heap)
        result.append(val)
        if elem_i + 1 < len(lists[list_i]):
            heapq.heappush(heap, (lists[list_i][elem_i+1], list_i, elem_i+1))
    return result

# ML application: Beam Search (top-K candidates at each step)
def beam_search(scores_per_step, k=3):
    """scores_per_step: list of dicts {token: log_prob} for each step."""
    beams = [(0.0, [])]   # (cumulative_score, tokens_so_far)
    for step_scores in scores_per_step:
        candidates = []
        for score, seq in beams:
            for token, log_prob in step_scores.items():
                candidates.append((score + log_prob, seq + [token]))
        beams = heapq.nlargest(k, candidates, key=lambda x: x[0])
    return beams
```

---

## ML-Specific Coding Questions

These questions come up specifically in ML engineer interviews:

```python
# 1. Implement gradient descent
def gradient_descent(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(epochs):
        pred  = X @ theta
        grad  = (2/m) * X.T @ (pred - y)
        theta -= lr * grad
    return theta

# 2. Compute cosine similarity matrix efficiently
def cosine_similarity_matrix(A):
    # A: (n, d)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    A_norm = A / (norms + 1e-8)
    return A_norm @ A_norm.T   # (n, n)

# 3. K-Means from scratch
def kmeans(X, k, max_iter=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(max_iter):
        dists   = np.linalg.norm(X[:, None] - centroids[None], axis=2)
        labels  = dists.argmin(axis=1)
        new_c   = np.array([X[labels==i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, new_c): break
        centroids = new_c
    return labels, centroids

# 4. Implement softmax (numerically stable)
def softmax(z):
    e_z = np.exp(z - z.max(axis=-1, keepdims=True))   # stability trick
    return e_z / e_z.sum(axis=-1, keepdims=True)

# 5. Batch normalisation forward pass
def batch_norm(X, gamma, beta, eps=1e-8):
    mu    = X.mean(axis=0)
    sigma = X.var(axis=0)
    X_hat = (X - mu) / np.sqrt(sigma + eps)
    return gamma * X_hat + beta
```

---

## Time & Space Complexity Reference

| Algorithm | Time | Space |
|---|---|---|
| Sorting (Python `sorted`) | O(n log n) | O(n) |
| Binary search | O(log n) | O(1) |
| Hash map lookup | O(1) avg | O(n) |
| BFS / DFS | O(V + E) | O(V) |
| DP (2D grid) | O(m·n) | O(m·n) or O(n) |
| Heap push/pop | O(log n) | O(n) |
| Matrix multiply | O(n³) naive | O(n²) |
| K-Means (one iter) | O(n·k·d) | O(n·d) |

---

## Interview Strategy

1. **Clarify** — ask about input size, edge cases, expected output format
2. **Brute force first** — state the O(n²) solution before optimising
3. **Optimise** — identify bottleneck, apply the right data structure
4. **Code cleanly** — meaningful variable names, handle edge cases
5. **Test** — walk through your solution with the example + edge cases
6. **Analyse** — state time and space complexity at the end

**Practice platforms**: [LeetCode](https://leetcode.com) (top 150), [NeetCode](https://neetcode.io) (structured roadmap), [HackerRank](https://hackerrank.com)

**Focus list for ML interviews**: Two Sum, Valid Palindrome, Best Time to Buy/Sell Stock, Climbing Stairs, House Robber, Coin Change, Merge K Sorted Lists, Word Search, Number of Islands, Course Schedule (topological sort).
