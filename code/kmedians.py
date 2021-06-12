"""
@file kmedians.py
implementation of the k-medians algorithm.
"""

from clustering import Clustering
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer

class kmediansClustering(Clustering):
    """
    implements k-Medians Clustering
    uses the pyclustering k-medians implementation
    centers are initialised using the random initialiser
    """

    def __init__(self, metric, dataset, seed=None):
        """
        constructor
        @param metric metric description as string. allowed: "euclidean", "manhattan", "chebyshev", "cosine"
        @param dataset dataset given as string. allowed: "diabetes", "iris", "wine", "housevotes"
        """
        super().__init__(metric, dataset, seed)

        ## metric name as pyclustering distance_metric object
        self.metric = self.pyc_metric(metric)

        ## dataset name as string
        self.dataset = dataset

        ## data that gets clustered
        self.data = []

        ## expected cluster values
        self.labels = []
            
        ## seed for initializer, None if no seed is used
        self.seed = seed
    
    def cluster(self, k, plusplus=True):
        """
        clustering method. Will execute clustering on the data saved in self.data with the metric
        given in self.metric
        @param k number of clusters that are generated
        @returns clusters as list of lists of indices of points and final cluster medians
        """
        initializer_args = {"data" : self.data, "amount_centers" : k}
        if self.seed is not None:
            initializer_args["random_state"] = self.seed

        if plusplus:
            # k++ center initialiser
            initial_centers = kmeans_plusplus_initializer(**initializer_args).initialize()
        else:
            # random initialiser
            initial_centers = random_center_initializer(**initializer_args).initialize()

        kmedians_instance = kmedians(self.data, initial_centers, metric=self.metric)
        # clustering
        kmedians_instance.process()

        clusters = kmedians_instance.get_clusters()
        medians = kmedians_instance.get_medians()

        return clusters, medians
