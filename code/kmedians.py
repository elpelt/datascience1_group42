from clustering import Clustering
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer

class kmediansClustering(Clustering):
    """
    implements k-Medians Clustering
    uses the pyclustering k-medians implementation
    centers are initialised using the random initialiser
    """

    def __init__(self, metric, dataset):
        """
        constructor
        @param metric metric description as string. allowed: "euclidean", "manhattan", "chebyshev", "cosine"
        @dataset dataset given as string. allowed: "diabetes", "iris", "wine", "housevotes"
        """
        super().__init__(metric, dataset)
        self.data = self.load_data()
        self.metric = self.pyc_metric(metric)
    
    def cluster(self, k):
        """
        clustering method. Will execute clustering on the data saved in self.data with the metric
        given in self.metric
        @param k number of clusters that are generated
        @returns clusters as list of lists of indices of points and final cluster medians
        """
        initial_centers = random_center_initializer(self.data, k).initialize()

        kmedians_instance = kmedians(self.data, initial_centers, metric=self.metric)
        # clustering
        kmedians_instance.process()

        clusters = kmedians_instance.get_clusters()
        medians = kmedians_instance.get_medians()

        return clusters, medians