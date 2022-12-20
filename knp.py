import numpy as np
from sklearn.datasets import make_blobs
from pyvis.network import Network
import webbrowser
import random

class KNP:
    def __init__(self, points, k=3):

        self.Points = points
        self.K = k
        self.edges = []
        self.graph = []
        random.seed(15)

    def create_mst(self):

        fix = round(len(self.Points) / 2)
        print("fix: ", fix)
        for i in range(len(self.Points)):
            for j in range(len(self.Points)):
                if i != j and j < fix - 1:
                    weigth = random.randint(5, 100)

                    self.graph.append([weigth, i, j])
        print("граф ", self.graph)

        Rs = sorted(self.graph, key=lambda x: x[0])
        merged = set()
        isolated = {}

        for r in Rs:
            if r[1] not in merged or r[2] not in merged:
                if r[1] not in merged and r[2] not in merged:
                    isolated[r[1]] = [r[1], r[2]]
                    isolated[r[2]] = isolated[r[1]]
                else:
                    if not isolated.get(r[1]):
                        isolated[r[2]].append(r[1])
                        isolated[r[1]] = isolated[r[2]]
                    else:
                        isolated[r[1]].append(r[2])
                        isolated[r[2]] = isolated[r[1]]

                self.edges.append(r)
                merged.add(r[1])
                merged.add(r[2])

        for r in Rs:
            if r[2] not in isolated[r[1]]:
                self.edges.append(r)
                gr1 = isolated[r[1]]
                isolated[r[1]] += isolated[r[2]]
                isolated[r[2]] += gr1
        print("ребра в остове: ", self.edges)
        return self.edges

    def create_clasters(self):
        num_ribs_remove = self.K - 1
        print("Необходимо удалить ребер: ", num_ribs_remove)

        for i in range(num_ribs_remove):
            min_value = 0
            idx_rib = 0
            for idx_edge, edge in enumerate(self.edges):

                if idx_edge == 0:
                    min_value = edge[0]
                    idx_rib = 0
                if edge[2] <= min_value:
                    min_value = edge[0]
                    idx_rib = idx_edge
            self.edges.pop(idx_rib)
        return self.edges


if __name__ == '__main__':
    np.random.seed(42)
    X, y = make_blobs(10, random_state=0)
    print("количество точек: ", len(X))

    graph = KNP(points=X)
    gr_ribs = graph.create_mst()

    net = Network('1000px', directed=True)

    for i, p in enumerate(X):
        net.add_node(i, label=f'{i}', x=p[0], y=p[1])
    for idx, rib in enumerate(gr_ribs):
        net.add_edge(rib[1], rib[2], weight=rib[0])

    url = 'nx.html'
    net.show('nx.html')
    webbrowser.open(url, new=2)

    clusters = graph.create_clasters()

    net2 = Network('1000px', directed=True)

    for i2, p2 in enumerate(X):
        net2.add_node(i2, label=f'{i2}', x=p2[0], y=p2[1])

    for idx_cl, cl in enumerate(clusters):
        net2.add_edge(cl[1], cl[2], weight=cl[0])

    url = 'nx2.html'
    net2.show('nx2.html')
    webbrowser.open(url, new=2)
