import numpy as np


class Graph:
    def __init__(self, vertex):
        self.V = vertex
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    def search(self, parent, i):
        if parent[i] == i:
            return i
        return self.search(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        xroot = self.search(parent, x)
        yroot = self.search(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def kruskal(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.search(parent, u)
            y = self.search(parent, v)
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.apply_union(parent, rank, x, y)
        total_weight = 0.0
        for u, v, weight in result:
            print("Edge:", u, v, end=" ")
            print("-", weight)
            total_weight += weight
        return total_weight

"""
A 0
B 1
C 2
D 3
E 4
F 5
G 6
H 7
"""


g = Graph(8)
g.add_edge(0, 1, 3)
g.add_edge(1, 2, 5)
g.add_edge(2, 3, 1)
g.add_edge(1, 5, 8)
g.add_edge(0, 3, 5)
g.add_edge(5, 4, 10)
g.add_edge(4, 7, 3)

g.add_edge(1, 4, 4)
g.add_edge(2, 4, 2)
g.add_edge(2, 7, 7)
g.add_edge(2, 6, 4)

g.add_edge(6, 7, 2)
g.add_edge(3, 6, 2)
g.add_edge(3, 6, 6)

weight = g.kruskal()
print(weight)