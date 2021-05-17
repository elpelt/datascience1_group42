from clustering import Clustering

from sklearn.cluster import DBSCAN

"""
metrics can be given as a string, so no metric selection needed
"""
class DBSCANClustering(Clustering):
    def __init__(self, metric, dataset, path=""):
        super().__init__(metric, dataset)
        self.metric = metric
        self.data = self.load_data()

    def cluster(self, eps, minPts):
        """
        params are the same as in the DBSCAN paper
        @param eps Distance for the Eps-Neighbourhood
        @param minPts Minmal number of points in a cluster

        """

        clustering = DBSCAN(metric=self.metric, eps=eps, min_samples=minPts)
        clustering.fit(self.data)
 
        return self.package(clustering.labels_)
    
    def package(self, labels):
        noise = []
        m = max(labels)
        clusters = [list() for i in range(m+1)]

        for i in range(len(labels)):
            if labels[i] == -1:
                noise.append([i])
            else:
                clusters[labels[i]].append(i)

        return clusters + noise


if __name__ == "__main__":
    flarepath = "../datasets/solar_flares/flare.data1"
    c = DBSCANClustering("euclidean", "wine")
    c.load_data()
    print(c.cluster(50, 3))
