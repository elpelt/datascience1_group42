from clustering import Clustering
from pyclustering.cluster.kmedians import kmedians

class kmediansClustering(Clustering):
    def __init__(self, metric, dataset):
        super().__init__(metric, dataset)
        self.data = self.load_data()
        self.metric = self.pyc_metric(metric)
    
    def cluster(self, k, initial_medians):
        kmedians_instance = kmedians(self.data, initial_medians, metric=self.metric)
        kmedians_instance.process()

        clusters = kmedians_instance.get_clusters()
        medians = kmedians_instance.get_medians()

        return clusters, medians