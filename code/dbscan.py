from clustering import Clustering

from sklearn.cluster import DBSCAN

"""
metrics can be given as a string, so no metric selection needed
"""
class DBSCANClustering(Clustering):
    def __init__(self, metric, dataset, path=""):
        super().__init__(metric, dataset)
        self.metric = metric

    def cluster(self, eps, minPts):
        """
        params are the same as in the DBSCAN paper
        @param eps Distance for the Eps-Neighbourhood
        @param minPts Minmal number of points in a cluster

        """

        clustering = DBSCAN(metric=self.metric, eps=eps, min_samples=minPts)
        clustering.fit(self.data)



if __name__ == "__main__":
    flarepath = "../datasets/solar_flares/flare.data1"
    c = DBSCANClustering("manhattan", "wine")
    c.load_data()
    c.cluster()