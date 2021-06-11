from pyclustering.utils import read_sample
from pyclustering.utils.metric import type_metric, distance_metric

from sklearn.datasets import load_diabetes, load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np
import pandas as pd


class Clustering():
    """
    Meta Class for all subsequent clustering algorithms<br>
    implements all functions needed for running the different<br>
    cluster algorithms
    """
    def __init__(self, metric, dataset, seed=None):
        """
        constructor
        @param metric metric description as string. allowed: "euclidean", "manhattan", "chebyshev", "cosine"
        @param dataset dataset given as string. allowed: "diabetes", "iris", "wine", "housevotes"
        """

        ## metric name as string or pyclustering distance_metric object
        self.metric = metric

        ## dataset name as string
        self.dataset = dataset

        ## data that gets clustered
        self.data = []

        ## expected cluster values
        self.labels = []

        ## seed for initializer, None if no seed is used
        self.seed = seed

        ## dataset as pandas frame. needed for web frontend later
        self.datadf = None
    
    def pyc_metric(self, metric):
        """
        returns a distance metric which is usable by the pyclustering algorithms
        @param distance metric string. allowed: "euclidean", "manhattan", "chebyshev", "cosine"
        @returns pyclustering distance_metric object, None when distance is not supported
        """
        # distance for plotting
        if metric == "euclidean":
            return distance_metric(type_metric.EUCLIDEAN)
        
        elif metric == "manhattan":
            return distance_metric(type_metric.MANHATTAN)
        
        elif metric == "chebyshev":
            return distance_metric(type_metric.CHEBYSHEV)
        
        # pyclustering does not have a cosine distance implementation
        # therefore it is defined here
        elif metric == "cosine":
            
            def cosine(x, y):

                def dot(x, y):
                    return sum([x[i]*y[i] for i in range(len(x))])

                def length(x):
                    return dot(x, x)**0.5

                if np.all((x == 0)) or np.all((y==0)): # case for 0-vectors (otherwise division by 0!)
                    cos_similarity = 1
                else:
                    cos_similarity = dot(x, y)/(length(x) * length(y))
                    if cos_similarity > 1: # due to some rounding errors with floating point values
                        cos_similarity = 1

                if np.any(np.array(x)<0) or np.any(np.array(y)<0):
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
        loads in a dataset, standardises it and sets it as self.data attribute
        """
        # sklearn data import
        if self.dataset == "diabetes":
            s = load_diabetes()
            self.labels = s["target"]
            self.data = s["data"]
            self.datadf = load_diabetes(as_frame=True)["data"]
        
        elif self.dataset == "iris":
            s = load_iris()
            self.labels = s["target"]
            self.data = s["data"]
            self.datadf = load_iris(as_frame=True)["data"]
        
        elif self.dataset == "wine":
            s = load_wine()
            self.labels = s["target"]
            self.data = s["data"]
            self.datadf = load_wine(as_frame=True)["data"]
        
        elif self.dataset == "housevotes":
            path = "./datasets/votes/house-votes-84.data"
            self.data, self.labels, self.datadf = self.house_load(path, 0)
                
        self.data = StandardScaler().fit_transform(self.data)

    def house_load(self, path, skip=1):
        """
        loads the housevotes dataset and encodes it using One-Hot-Encoding<br>
        democrats are labeled as 1, republicans as 0
        @param path filepath to the dataset
        @param skip number of lines that get skipped when reading in a file
        @return One-Hot-Encoded housevotes dataset and labels as array of 1s and 0s
        """
        data = []
        datadf = None

        with open(path, 'r') as f:
            row = -1
            first = True

            # list comprehension loading whole dataset
            data = [line.strip().split(',') for line in f][skip:]

            labels = np.zeros(len(data))
            
            # label extraction. labels are the first attribute
            # democrats are 1, republicans 0
            for i in range(len(data)):
                if data[i].pop(0) == "democrat":
                    labels[i] = 1
        
            datadf = pd.DataFrame(data, columns=[f"vote {i+1}" for i in range(len(data[0]))])
            
        # one hot encoding
        enc = OneHotEncoder().fit_transform(data).toarray()

        return enc, labels, datadf
    

    def cluster(self):
        """
        does nothing in the meta class.<br>
        needs to be implemented in the inheriting cluster algorithm classes
        """
        pass


if __name__ == "__main__":
    test = Clustering("cosine", "solarflare1")
    test.load_data()
    