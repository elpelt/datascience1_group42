from clustering import Clustering
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer

class kmediansClustering(Clustering):
    def __init__(self, metric, dataset):
        super().__init__(metric, dataset)
        self.data = self.load_data()
        self.metric = self.pyc_metric(metric)
    
    def cluster(self, k):

        initial_centers = random_center_initializer(self.data, k).initialize()

        kmedians_instance = kmedians(self.data, initial_centers, metric=self.metric)
        kmedians_instance.process()

        clusters = kmedians_instance.get_clusters()
        medians = kmedians_instance.get_medians()

        return clusters, medians