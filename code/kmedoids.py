from clustering import Clustering
from sklearn_extra.cluster import KMedoids

class kmedoidsClustering(Clustering):
    """
    implements k-Medians Clustering
    uses the scikit-learn-extra k-medoids implementation
    centers are set using the k++ initialiser if not set differently
    """

    def __init__(self, metric, dataset):
        super().__init__(metric, dataset)
        self.data = self.load_data()
        self.metric = metric
        
    def cluster(self, k, init="k-medoids++"):
        """
        clustering method. Will execute clustering on the data saved in self.data with the metric
        given in self.metric
        @param k number of clusters that are generated
        @param init initialisation parameter. Standard: "k-medoids++"
        @returns clusters as list of lists of indices of points, final cluster centers
        """

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
