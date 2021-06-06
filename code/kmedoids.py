from clustering import Clustering
from sklearn_extra.cluster import KMedoids

class kmedoidsClustering(Clustering):
    """
    implements k-Medians Clustering<br>
    uses the scikit-learn-extra k-medoids implementation<br>
    centers are set using the k++ initialiser if not set differently
    """

    def __init__(self, metric, dataset):
        """
        constructor
        @param metric metric description as string. allowed: "euclidean", "manhattan", "chebyshev", "cosine"
        @param dataset dataset given as string. allowed: "diabetes", "iris", "wine", "housevotes"
        """
        super().__init__(metric, dataset)

        ## metric name as string
        self.metric = self.pyc_metric(metric)

        ## dataset name as string
        self.dataset = dataset

        ## data that gets clustered
        self.data = []

        ## expected cluster values
        self.labels = []

    def cluster(self, k, init="k-medoids++"):
        """
        clustering method. Will execute clustering on the data saved in self.data with the metric
        given in self.metric
        @param k number of clusters that are generated
        @param init initialisation parameter. Default: "k-medoids++"
        @returns clusters as list of lists of indices of points, final cluster centers
        """

        if k == 1:
            return self.package([0 for i in range(len(self.data))]), [0]

        kmedoids = KMedoids(n_clusters=k, random_state=42, init=init, metric=self.metric, method='pam')
        kmedoids.fit(self.data)

        return self.package(kmedoids.labels_), kmedoids.cluster_centers_

    def package(self, labels):
        """
        rearranges the result to a format similar to the one of the pyclustering algorithms
        allows for easier access in the streamlit interface
        @param labels labels returned from the KMedoids algorithm
        @returns clusters formated similarly to the pyclustering algorithms
        """
        m = max(labels)
        clusters = [list() for i in range(m+1)]

        for i in range(len(labels)):
            clusters[labels[i]].append(i)

        return clusters

if __name__ == "__main__":
    c = kmedoidsClustering("cosine", "iris")
    c.load_data()
    c.cluster(1)
