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
        kdist = []
        for pi in range(len(self.clustering.data)):
            distances = []
            for pj in range(len(self.clustering.data)):
                if pj == pi:
                    continue
                
                distances.append(pairwise_distances([self.clustering.data[pi]], [self.clustering.data[pj]], metric=self.metric)[0][0])
            
            distances.sort()
            kdist.append(distances[k-1])
        
        return kdist
    
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
    test.set_metric("chebyshev")
    test.set_dataset("housevotes")
    kdist = test.kdist(4)
    test.plot_kdist(kdist)

    