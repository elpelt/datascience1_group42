from clustering import Clustering
from sklearn_extra.cluster import KMedoids


class kmedoidsClustering(Clustering):
    def __init__(self, metric, dataset):
        super().__init__(metric, dataset)
        self.data = self.load_data()
        self.metric = metric
    
    def cluster(self, k, init="k-medoids++"):

        kmedoids = KMedoids(n_clusters=k, random_state=42, init=init, metric=self.metric, method='pam')
        kmedoids.fit(self.data)

        return self.package(kmedoids.labels_), kmedoids.cluster_centers_

    def package(self, labels):
        m = max(labels)
        clusters = [list() for i in range(m+1)]

        for i in range(len(labels)):
            clusters[labels[i]].append(i)

        return clusters
