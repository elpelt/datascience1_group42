# pyclustering: https://github.com/annoviko/pyclustering/
# documentation: https://pyclustering.github.io/docs/0.8.2/html/index.html
# own metrics: https://github.com/annoviko/pyclustering/issues/471
from pyclustering.utils import read_sample
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.samples.definitions import FCPS_SAMPLES

# datasets, maybe replace this, to reduce a dependency later
from sklearn.datasets import load_diabetes, load_iris, load_wine


class Clustering():
    def __init__(self, metric, dataset, path=""):
        self.data = []
        self.path = path
        self.dataset = dataset
        self.encoding = "integer"
        self.path = path
    
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
                
                return dot(x, y)/(length(x) * length(y))

            return distance_metric(type_metric.USER_DEFINED, func=cosine);
        
        else:
            print("wrong distance measure given")
            return None
    
    def load_data(self):
        """
        loads in datasets
        """
        # sklearn data import
        if self.dataset == "diabetes":
            return load_diabetes()["data"]
        
        elif self.dataset == "iris":
            return load_iris()["data"]
        
        elif self.dataset == "wine":
            return load_wine()["data"]
        
        elif self.dataset == "solarflare":
            return self.solar_load()

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

    def solar_load(self, skip=1):
        groups = set()

        with open(self.path, 'r') as f:
            n = 0
            data = [i.split(" ") for i in f.read().splitlines()[skip:]]

        for i in range(len(data)):
            for j in range(len(data[-1])):
                try:
                    data[i][j] = int(data[i][j])
                except:
                    groups.add(data[i][j])
        
        if self.encoding == "integer":
            self.integer_encoding(groups)
        
        return data
        
    def cluster(self):
        pass

if __name__ == "__main__":
    pass