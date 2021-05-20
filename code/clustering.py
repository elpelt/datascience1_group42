# pyclustering: https://github.com/annoviko/pyclustering/
# documentation: https://pyclustering.github.io/docs/0.8.2/html/index.html
# own metrics: https://github.com/annoviko/pyclustering/issues/471
from pyclustering.utils import read_sample
from pyclustering.utils.metric import type_metric, distance_metric

# datasets, maybe replace this, to reduce a dependency later
from sklearn.datasets import load_diabetes, load_iris, load_wine
from sklearn.preprocessing import StandardScaler
import numpy as np

"""
Meta Class for all subsequent clustering algorithms
implements all functions needed for 
"""
class Clustering():
    def __init__(self, metric, dataset):
        self.data = []
        self.dataset = dataset
        self.encoding = "integer"
        self.labels = None
    
    def pyc_metric(self, metric):
        # distance for plotting
        if metric == "euclidean":
            return distance_metric(type_metric.EUCLIDEAN)
        
        elif metric == "manhattan":
            return distance_metric(type_metric.MANHATTAN)
        
        elif metric == "chebyshev":
            return distance_metric(type_metric.CHEBYSHEV)
        
        # cosines' not converging for a centroid located at (0,0), needs fixing
        elif metric == "cosine":
            
            def cosine(x, y):
                def dot(x, y):
                    return sum([x[i]*y[i] for i in range(len(x))])

                def length(x):
                    return dot(x, x)**0.5


                cos_similarity = dot(x, y)/(length(x) * length(y))

                if cos_similarity > 1: # due to some rounding errors with floating point values
                    cos_similarity = 1

                if np.any(x<0) or np.any(y<0):
                    factor = 2
                else:
                    factor = 1

                return factor*np.arccos(cos_similarity)/np.pi

            return distance_metric(type_metric.USER_DEFINED, func=cosine)
        
        else:
            print("wrong distance measure given")
            return None
    
    def load_data(self):
        """
        loads in datasets
        """
        # sklearn data import
        if self.dataset == "diabetes":
            s = load_diabetes()
            self.labels = s["target"]
            self.data = s["data"]
        
        elif self.dataset == "iris":
            s = load_iris()
            self.labels = s["target"]
            self.data = s["data"]
        
        elif self.dataset == "wine":
            s = load_wine()
            self.labels = s["target"]
            self.data = s["data"]
        
        elif self.dataset == "solarflare1":
            path = "../datasets/solar_flares/flare.data1"
            self.solar_load(path)
        
        elif self.dataset == "solarflare2":
            path = "../datasets/solar_flares/flare.data2"
            self.solar_load(path)

        self.data = StandardScaler().fit_transform(self.data)

    def integer_encoding(self, objects):
        self.replacement = {}
        objects = sorted(list(objects))

        for i in range(len(objects)):
            self.replacement[objects[i]] = i
        
        for i in range(len(self.data)):
            for j in range(len(self.data[-1])):
                if not isinstance(self.data[i][j], int):
                    self.data[i][j] = self.replacement[self.data[i][j]]

    def one_hot_encoding(self):
        pass

    def solar_load(self, path, skip=1):
        groups = set()

        with open(path, 'r') as f:
            n = 0
            self.data = [i.split(" ") for i in f.read().splitlines()[skip:]]

        for i in range(len(self.data)):
            for j in range(len(self.data[-1])):
                try:
                    self.data[i][j] = int(self.data[i][j])
                except:
                    groups.add(self.data[i][j])
        
        if self.encoding == "integer":
            self.integer_encoding(groups)
        
    def cluster(self):
        pass

if __name__ == "__main__":
    pass