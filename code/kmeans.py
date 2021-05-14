from clustering import Clustering
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


class kmeansClustering(Clustering):
    def __init__(self, metric, dataset, path=""):
        super().__init__(metric, dataset, path)
        self.data = self.load_data()
        self.metric = self.pyc_metric(metric)
    
    def cluster(self):
        initial_centers = kmeans_plusplus_initializer(self.data, 2).initialize()

        kmeans_instance = kmeans(self.data, initial_centers, metric=self.metric)

        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()
        final_centers = kmeans_instance.get_centers()

        print(clusters)
        print(final_centers)

if __name__ == "__main__":
    c = kmeansClustering("manhattan", "wine")
    c.load_data()
    c.cluster()