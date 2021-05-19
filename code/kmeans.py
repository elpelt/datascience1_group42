from clustering import Clustering
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer


class kmeansClustering(Clustering):
    def __init__(self, metric, dataset):
        super().__init__(metric, dataset)
        self.data = self.load_data()
        self.metric = self.pyc_metric(metric)
    
    def cluster(self, k, plusplus=True):

        if plusplus:
            # k++ center initialiser
            initial_centers = kmeans_plusplus_initializer(self.data, k).initialize()
        else:
            # random initialiser
            initial_centers = random_center_initializer(self.data, k).initialize()

        kmeans_instance = kmeans(self.data, initial_centers, metric=self.metric)

        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()
        final_centers = kmeans_instance.get_centers()

        return clusters, final_centers
    