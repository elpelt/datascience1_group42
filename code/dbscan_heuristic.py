"""
@file dbscan_heuristic.py
implementation of DBSCAN parameter estimation heuristic
"""

from dbscan import DBSCANClustering
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt


class DBSCANHeuristic():
    """
    implements the DBSCAN heuristic proposed in the original DBSCAN paper:

    Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. A density-based algorithm for discovering clusters in large spatial databases
    with noise. In Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, KDD’96, page 226–231. AAAI Press, 1996.
    """

    def __init__(self):
        """
        constructor. uses a DBSCANClustering object for loading the datasets
        """
        
        ## DBSCANClustering object for loading datasets
        self.clustering = DBSCANClustering("euclidean", "iris")

        ## metric as string ("euclidean", "cosine", "manhattan", "chebyshev")
        self.metric = None

        ## variable k used in the k-dist calculation
        self.k = 0

    def set_metric(self, metric):
        """
        setter for the metric
        @param metric string containing name of the metric ("euclidean", "cosine", "manhattan", "chebyshev")
        """
        self.metric = metric
    
    def set_dataset(self, dataset):
        """
        sets and loads the dataset using the DBSCANClustering objects load_data() function
        @param dataset string with name of the dataset used ("iris", "wine", "diabetes", "housevotes")
        """
        self.clustering.dataset = dataset
        self.clustering.load_data()
    
    def kdist(self, k):
        """
        calculates all k-distances for the dataset
        @param k variable for the k-dist. Natural Number.
        """
        self.k = k
        distances = pairwise_distances(self.clustering.data, self.clustering.data, metric=self.metric)
        kdist = [sorted(distances[i])[k-1] for i in range(len(distances))]
        return kdist
    
    def plot_kdist(self, kdist):
        """
        plots the sorted kdist graph using matplotlib
        @param k-dist list containing the k-distances for every point of the dataset
        """
        p = [i for i in range(len(kdist))]

        kdist.sort(reverse=True)

        fig, ax = plt.subplots()
        ax.set_title(f"DBSCAN Heuristic for {self.clustering.dataset} dataset, {self.metric} distance, k = {self.k}")
        ax.plot(p, kdist)
        ax.grid(True)
        ax.set_xlabel("Points")
        ax.set_ylabel("k-dist")
        #fig.savefig("dbscanheuristic.png")
        return fig
    