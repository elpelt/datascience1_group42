"""
@file kmeans.py
implementation of the k-means algorithm.
"""

from clustering import Clustering
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer

class kmeansClustering(Clustering):
    """
    Class implementing k-Means Clustering<br>
    uses the pyclustering k-means implementation<br>
    centers can be initialised using the k++ or the random initialiser
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
        @param plusplus will use k++ initialiser if true
        @returns clusters as list of lists of indices of points and final cluster centers
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

        kmeans_instance = kmeans(self.data, initial_centers, metric=self.metric)

        #clustering
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()
        final_centers = kmeans_instance.get_centers()

        return clusters, final_centers
    