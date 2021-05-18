from clustering import Clustering
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


class kmedoidsClustering(Clustering):
    def __init__(self, metric, dataset):
        super().__init__(metric, dataset)
        self.data = self.load_data()
        self.metric = self.pyc_metric(metric)
    
    def cluster(self, k, plusplus=True):

        if plusplus:
            # k++ center initialiser
            initial_medoids = kmeans_plusplus_initializer(self.data, k).initialize()
        else:
            # random initialiser
            initial_medoids = random_center_initializer(self.data, k).initialize()

        kmedoids_instance = kmedoids(self.data, initial_medoids, metric=self.metric)
 
        # Run cluster analysis and obtain results.
        kmedoids_instance.process()
        clusters = kmedoids_instance.get_clusters()
        medoids = kmedoids_instance.get_medoids()
        return clusters, medoids

if __name__ == "__main__":
    c = kmedoidsClustering("manhattan", "diabetis", "../datasets/solar_flares/flare.data2")
    c.load_data()
    print(c.cluster(4))
    