# pyclustering: https://github.com/annoviko/pyclustering/
# documentation: https://pyclustering.github.io/docs/0.8.2/html/index.html
# own metrics: https://github.com/annoviko/pyclustering/issues/471
import numpy as np
from pyclustering.utils import read_sample
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.samples.definitions import FCPS_SAMPLES

class kmeansClustering():
    def __init__(self, path, metric):
        self.data = read_sample(path)
        self.metric = self.select_metric(metric)
    
    def select_metric(self, metric):
        # distance for plotting
        if metric == "euclidean" or 0:
            return distance_metric(type_metric.EUCLIDEAN)
        
        elif metric == "manhattan" or 1:
            return distance_metric(type_metric.MANHATTAN)
        
        elif metric == "chebyshev" or 2:
            return distance_metric(type_metric.CHEBYSHEV)
        
        elif metric == "mahalanobis" or 3:
            
            # todo
            def mahalanobis(x, y):
                return [(x[i]**2 + y[i]**2)**(0.5) for i in range(x)]

            return distance_metric(type_metric.USER_DEFINED, func=mahalanobis);
        
        else:
            print("wrong distance measure given")
            return None

    
    def cluster(self):
        # Load list of points for cluster analysis.
        self.data = read_sample(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS)

        # Prepare initial centers using K-Means++ method.
        initial_centers = kmeans_plusplus_initializer(self.data, 2).initialize()

        # Create instance of K-Means algorithm with prepared centers.
        kmeans_instance = kmeans(self.data, initial_centers, metric=self.metric)

        # Run cluster analysis and obtain results.
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()
        final_centers = kmeans_instance.get_centers()

        # Visualize obtained results
        kmeans_visualizer.show_clusters(self.data, clusters, final_centers)


if __name__ == "__main__":
    path = "../datasets/shuttle/shuttle.tst"
    test = kmeansClustering(path, "mahalanobis")
    test.cluster()