import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs



def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))



class KMeans:
    def __init__(self, K=2, max_iters=100, plot_steps=False, GlobalItter=1):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.GlobalItter = GlobalItter
        self.Distance = 0


        self.clusters = [[] for _ in range(self.K)]

        self.centroids = []



    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for clusters_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = clusters_idx
        return labels

    def predict(self, X):

        self.X = X

        self.n_samples, self.n_features = X.shape

        random_samples_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_samples_idxs]

        for _ in range(self.max_iters):
            self.GlobalItter = self.GlobalItter + _
            if self.plot_steps:
                self.plot()
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self.plot_steps:
                self.plot()

            if self._is_converged(centroids_old, self.centroids):
                break
        return self._get_cluster_labels(self.clusters)




    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]

        for idx, sample in enumerate(self.X):


            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)

        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids


    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        for point in self.centroids:
            ax.scatter(*point, marker='x', color="black", linewidth=2)
        plt.show()

    def Dist(self):
        for i, index in enumerate(self.clusters):
            tempDistance = 0
            for j in index:
                tempDistance = tempDistance + euclidean_distance(self.X[j], self.centroids[i])
            self.Distance = self.Distance + tempDistance
        return self.Distance



if __name__ == '__main__':
    np.random.seed(42)
    X, y = make_blobs(250, random_state=0)
    print("количество точек: ", len(X))


    clusters = 3

    GlobalItter = 1
    k = KMeans(K=clusters, max_iters=150, plot_steps=False, GlobalItter=GlobalItter)
    y_pred = k.predict(X)
    k.Dist()
    k.plot()

    n = 2 * np.sqrt(len(X))
    Sum_of_distance = []
    print(round(n))
    for i in range(1, round(n)):
        temp_k = KMeans(K=i, max_iters=150, plot_steps=False, GlobalItter=GlobalItter)
        temp_y_pred = temp_k.predict(X)
        result_sum_klasters = temp_k.Dist()
        Sum_of_distance.append(result_sum_klasters)



    Optimum_klasters = float('inf')
    min = float('inf')
    for i in range(2, len(Sum_of_distance) - 1):
        tmp = abs(Sum_of_distance[i] - Sum_of_distance[i + 1]) / abs(Sum_of_distance[i - 1] - Sum_of_distance[i])
        if min > tmp:
            min = tmp
            Optimum_klasters = i
    print("Оптимальное количество кластеров: ", Optimum_klasters)


    GlobalItter = 1
    t = KMeans(K=Optimum_klasters, max_iters=150, plot_steps=True, GlobalItter=GlobalItter)
    new_pred = t.predict(X)
    t.plot()
