from clustering import Clustering

from sklearn.cluster import DBSCAN

class DBSCANClustering(Clustering):
    """
    implements DBSCAN Clustering
    uses the scikit-learn DBSCAN implementation
    """
    
    def __init__(self, metric, dataset):
        super().__init__(metric, dataset)
        self.metric = metric
        self.load_data()
    
    def cluster(self, eps, minPts):
        """
        clustering method. Will execute clustering on the data saved in self.data with the metric
        given in self.metric
        params are the same as in the DBSCAN paper
        @param eps Distance for the Eps-Neighbourhood
        @param minPts Minmal number of points in a cluster
        @returns formatted clustered data
        """
        clustering = DBSCAN(metric=self.metric, eps=eps, min_samples=minPts)
        clustering.fit(self.data)
 
        return self.package(clustering.labels_)
    
    def package(self, labels):
        """
        rearranges the result to a format similar to the one of the pyclustering algorithms
        allows for easier access in the streamlit interface
        @param labels cluster labels DBSCAN assigns to a point
        @returns clusters as list of lists of indices of points and noise as list of indices of points
        """
        noise = []
        m = max(labels)
        clusters = [list() for i in range(m+1)]

        for i in range(len(labels)):
            if labels[i] == -1:
                noise.append([i])
            else:
                clusters[labels[i]].append(i)

        return clusters, noise


if __name__ == "__main__":
    c = DBSCANClustering("chebyshev", "solarflare1")
    c.cluster(2, 3)
