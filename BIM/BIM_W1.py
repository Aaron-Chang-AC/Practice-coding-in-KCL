import numpy as np
import pandas as pd

class MST:
    def __init__(self,input):
        self.input = input
        self.weight = 0.0
        self.vertices = 0
        self.df_bidirectional = None

    def read_graph_from_csv(self):
        df = pd.read_csv(self.input, encoding='unicode_escape')
        return df

    def make_graph(self):
        df = self.read_graph_from_csv()
        num_nodes = df[['node1', 'node2']].max().max() - df[['node1', 'node2']].min().min() + 1
        # print(f"Graph:\n{df}")
        self.vertices = num_nodes
        print(f"number of nodes: {num_nodes}\nnumber of edges: {len(df)}")

        # edge
        edge_temp = df[['node1', 'node2']].to_numpy()
        EDGES = np.append(edge_temp, df[['node2', 'node1']].to_numpy(), axis=0)
        # weight
        weight_temp = df['weight'].to_numpy()
        weight = np.append(weight_temp, df['weight'].to_numpy())

        # bidirectional graph
        df_bidirectional = pd.DataFrame(
            {'node1': EDGES[:, 0], 'node2': EDGES[:, 1], 'weight': weight})
        df_bidirectional = df_bidirectional.drop_duplicates(ignore_index=True)
        # print(df_bidirectional)
        self.df_bidirectional = df_bidirectional


    def alpha_num_convert(self, x):
        return ord(x) - 97

    def num_alpha_convert(self, x):
        return chr(x+97)

    def transform_alphabet_to_csv(self, alphabet_input=None):
        node1 = []
        node2 = []
        weight = []

        # alphabet to number
        for i in range(len(alphabet_input)):
            for j in range(len(alphabet_input[i])):
                if type(alphabet_input[i][j]) == str:
                    alphabet_input[i][j] = alphabet_input[i][j].lower()
                    alphabet_input[i][j] = self.alpha_num_convert(alphabet_input[i][j])
        # append
        for i in range(len(alphabet_input)):
            node1.append(alphabet_input[i][0])
            node2.append(alphabet_input[i][1])
            weight.append(alphabet_input[i][2])

        df = pd.DataFrame({'node1': node1,
                           'node2': node2,
                           'weight': weight})
        df.to_csv(self.input, index=False)


    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def kruskal(self):
        result_vertices = [] # store resultant MST
        total_weight = 0
        # sort all edges in non-decreasing order
        sorted_df = self.df_bidirectional.sort_values(by=['weight'], ascending=True)
        sorted_df.to_csv(self.input, index = False)

        # An index variable, used for sorted edges
        i = 0
        # An index variable, used for result[]
        e = 0
        parent = []
        rank = []

        for node in range(self.vertices):
            parent.append(node)
            rank.append(0)

        while e < self.vertices - 1:
            u, v, w= sorted_df.iloc[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            if x!= y:
                e = e + 1
                result_vertices.append([u, v, w])
                self.union(parent, rank, x, y)
        for u, v, weight in result_vertices:
            print(f"Final Answer {self.num_alpha_convert(int(u))} --> {self.num_alpha_convert(int(v))} and weight is {weight}")
            total_weight += weight
        print(f"Total weight is: {total_weight}")







alphabet_input= [
    ["A", "B", 3],
    ["A", "D", 5],
    ["B", "C", 5],
    ["B", "F", 8],
    ["B", "E", 4],
    ["C", "D", 1],
    ["C", "G", 2],
    ["C", "H", 7],
    ["C", "E", 2],
    ["D", "G", 2],
    ["D", "H", 6],
    ["E", "H", 3],
    ["E", "F", 10],
    ["G", "H", 2]
]

input = "graph.csv"
mst = MST(input)
mst.transform_alphabet_to_csv(alphabet_input=alphabet_input)
mst.make_graph()
mst.kruskal()