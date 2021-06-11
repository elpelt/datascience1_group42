from dbscan import DBSCANClustering
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt


class DBSCANHeuristic():
    def __init__(self):
        self.clustering = DBSCANClustering("euclidean", "iris")
        self.metric = None

    def set_metric(self, metric):
        self.metric = metric
    
    def set_dataset(self, dataset):
        self.clustering.dataset = dataset
        self.clustering.load_data()
    
    def kdist(self, k):
        self.k = k
        distances = pairwise_distances(self.clustering.data, self.clustering.data, metric=self.metric)
        kdist = [sorted(distances[i])[k-1] for i in range(len(distances))]
        return kdist
    
    def percentages(self):
        nump = len(self.clustering.data)
        perc = []
        
        for pi in range(nump):
            perc.append(round(1-pi/nump, 3))
        
        return perc
    
    def plot_kdist(self, kdist):
        p = [i for i in range(len(kdist))]

        kdist.sort(reverse=True)

        fig, ax = plt.subplots()
        ax.set_title(f"DBSCAN Heuristic for {self.clustering.dataset} dataset, {self.metric} distance, k = {self.k}")
        ax.plot(p, kdist)
        ax.grid(True)
        ax.set_xlabel("Points")
        ax.set_ylabel("k-dist")
        #fig.savefig("tmp.png")
        return fig


if __name__ == "__main__":
    test = DBSCANHeuristic()
    test.set_metric("euclidean")
    test.set_dataset("iris")
    kdist = test.kdist(4)
    print(kdist)

    