"""
@file results.py
handler for saving and loading results.
"""

import os
import json
import numpy as np

class Results():
    """
    class for easily saving and loading already calculated clustering results<br>
    <br>
    every dataset has a folder containing subfolders for every clustering algorithm 
    containing more subfolders for every distance measure. Clustering results are saved as
    json files in their respective folders.
    """
    def __init__(self, parentpath):
        """
        constructor. needs the filepath to the parent directory where the json files are
        suposed to be saved
        @param parentpath filepath to the parent directory
        """
        self.parent = parentpath

    def get_path(self, dataset, algorithm, metric, **kwargs):
        """
        builds and returns the filepath to the json file fitting the given parameters
        @param dataset string with the name of the dataset ("iris", "wine", "diabetes", "DBSCAN")
        @param algorithm string with the name of the algorithm ("kmeans", "kmedians", "kmedoids", "DBSCAN")
        @param metric string with the name of the distance measure ("euclidean", "cosine", "chebyshev", "manhattan")
        @param **kwargs algorithm specific parameters. Needs to be either "k" or "minpts" and "eps"
        @returns filepath to the correct json file
        """
        path = f"{self.parent}/{dataset}/{algorithm}/{metric}/"
        
        if "k" in kwargs.keys():
            path += f'k_{kwargs["k"]}'
        else:
            path += f'minpts_{kwargs["minpts"]}_eps_{str(kwargs["eps"])}'
        
        path += ".json"

        return path

    def set_exists(self, dataset, algorithm, metric, **kwargs):
        """
        checks if a file for a result defined by the parameters exists
        @param dataset string with the name of the dataset ("iris", "wine", "diabetes", "DBSCAN")
        @param algorithm string with the name of the algorithm ("kmeans", "kmedians", "kmedoids", "DBSCAN")
        @param metric string with the name of the distance measure ("euclidean", "cosine", "chebyshev", "manhattan")
        @param **kwargs algorithm specific parameters. Needs to be either "k" or "minpts" and "eps"
        @returns True if file exists, False if not
        """
        path = self.get_path(dataset, algorithm, metric, **kwargs)
        return os.path.exists(path)
    
    def load_set(self, dataset, algorithm, metric, **kwargs):
        """
        loads results fitting the given parameters from a json file
        @param dataset string with the name of the dataset ("iris", "wine", "diabetes", "DBSCAN")
        @param algorithm string with the name of the algorithm ("kmeans", "kmedians", "kmedoids", "DBSCAN")
        @param metric string with the name of the distance measure ("euclidean", "cosine", "chebyshev", "manhattan")
        @param **kwargs algorithm specific parameters. Needs to be either "k" or "minpts" and "eps"
        @returns loaded clustering results (clusters and centers)
        """
        path = self.get_path(dataset, algorithm, metric, **kwargs)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return data["clusters"], data["centers"] 

    def save_set(self, dataset, algorithm, metric, clusters, centers, **kwargs):
        """
        saves cluster results in a json file
        @param dataset string with the name of the dataset ("iris", "wine", "diabetes", "DBSCAN")
        @param algorithm string with the name of the algorithm ("kmeans", "kmedians", "kmedoids", "DBSCAN")
        @param metric string with the name of the distance measure ("euclidean", "cosine", "chebyshev", "manhattan")
        @param **kwargs algorithm specific parameters. Needs to be either "k" or "minpts" and "eps"
        """
        path = self.get_path(dataset, algorithm, metric, **kwargs)

        centers = centers.tolist() if isinstance(centers, np.ndarray) else centers
        
        with open(path, 'w') as f:
            json.dump({"clusters" : clusters, "centers" : centers}, f, ensure_ascii=False)
