import os
import json
import numpy as np

class Results():
    """
    database connector and utility wrapper for easily saving and loading already calculated simulations<br>
    <br>
    every dataset has a own table consisting of entries for the parameters (k or minpts and eps)
    and the resulting cluster index for every entry. 
    """
    def __init__(self, parentpath):
        self.parent = parentpath

    def get_path(self, dataset, algorithm, metric, **kwargs):
        path = f"{self.parent}/{dataset}/{algorithm}/{metric}/"
        
        if "k" in kwargs.keys():
            path += f'k_{kwargs["k"]}'
        else:
            path += f'minpts_{kwargs["minpts"]}_eps_{str(kwargs["eps"])}'
        
        path += ".json"

        return path

    def set_exists(self, dataset, algorithm, metric, **kwargs):
        path = self.get_path(dataset, algorithm, metric, **kwargs)
        return os.path.exists(path)
    
    def load_set(self, dataset, algorithm, metric, **kwargs):
        path = self.get_path(dataset, algorithm, metric, **kwargs)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return data["clusters"], data["centers"] 

    def save_set(self, dataset, algorithm, metric, clusters, centers, **kwargs):
        path = self.get_path(dataset, algorithm, metric, **kwargs)

        centers = centers.tolist() if isinstance(centers, np.ndarray) else centers
        
        with open(path, 'w') as f:
            json.dump({"clusters" : clusters, "centers" : centers}, f, ensure_ascii=False)

if __name__ == '__main__':
   db = Results("./code/results")
   import sys
   sys.path.insert(0, "code/") 

   from kmeans import kmeansClustering

   c = kmeansClustering("euclidean", "iris")
   c.load_data()
   
   results, centers = c.cluster(2)

   db.save_set("iris", "kmeans", "euclidean", results, centers, k=2)

   cs, cens = db.load_set("iris", "kmeans", "euclidean", k=2)
   print(cs)