# Graph Theory, Boolean Algebra, and Combinatorics

## Introduction
This document provides an overview of **Graph Theory, Boolean Algebra, and Combinatorics**, along with their implementations in code and practical applications. These mathematical concepts are widely used in Computer Science, especially in areas like Data Structures, Algorithms, Cryptography, and AI.

---

## 1. Graph Theory
Graph Theory deals with the study of graphs, which are mathematical structures used to model relationships between objects.

### Key Concepts:
- **Graph Types**: Undirected, Directed, Weighted
- **Graph Representations**: Adjacency Matrix, Adjacency List
- **Algorithms**:
  - BFS (Breadth-First Search)
  - DFS (Depth-First Search)
  - Dijkstra's Algorithm (Shortest Path)
  - Kruskal's Algorithm (Minimum Spanning Tree)

### Implementation in Python:
```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node, end=" ")
            visited.add(node)
            queue.extend(graph[node])

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

bfs(graph, 'A')
```

### Practical Applications:
- Social Network Analysis
- Pathfinding Algorithms (Google Maps)
- AI (Knowledge Graphs, Recommendation Systems)

---

## 2. Boolean Algebra
Boolean Algebra is the branch of algebra that deals with true/false values using logic operations.

### Key Concepts:
- **Boolean Operations**: AND, OR, NOT
- **Boolean Expressions and Simplification**
- **Logic Gates**: AND, OR, XOR, NAND, NOR

### Implementation in Python:
```python
def boolean_operations(a, b):
    print(f"A AND B: {a and b}")
    print(f"A OR B: {a or b}")
    print(f"NOT A: {not a}")

boolean_operations(True, False)
```

### Practical Applications:
- Digital Circuit Design
- Control Systems
- Search Engines (Boolean Queries)

---

## 3. Combinatorics
Combinatorics is the study of counting, arrangement, and combination of elements.

### Key Concepts:
- **Permutations**: Order matters
- **Combinations**: Order does not matter
- **Binomial Theorem**

### Implementation in Python:
```python
from itertools import permutations, combinations

def generate_combinations(lst, r):
    return list(combinations(lst, r))

def generate_permutations(lst, r):
    return list(permutations(lst, r))

print("Combinations:", generate_combinations([1, 2, 3], 2))
print("Permutations:", generate_permutations([1, 2, 3], 2))
```

### Practical Applications:
- Cryptography
- Game Theory
- Probability Calculations

---

## Conclusion
Graph Theory, Boolean Algebra, and Combinatorics are fundamental mathematical concepts in Computer Science. They have significant real-world applications, ranging from AI to digital circuits and network security. Understanding their implementation in code allows for solving complex computational problems efficiently.

