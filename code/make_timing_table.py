from kmeans import kmeansClustering
from kmedians import kmediansClustering
from kmedoids import kmedoidsClustering
from dbscan import DBSCANClustering
import pandas as pd
import numpy as np

from results import Results

from datetime import datetime
import json

kalgos = ['kmeans', 'kmedians', 'kmedoids']
kalgoclass = {'kmeans': kmeansClustering, 'kmedians': kmediansClustering, 'kmedoids': kmedoidsClustering}

distances = ["euclidean", "manhattan", "chebyshev", "cosine"]

datasets = ["iris", "wine", "diabetes", "housevotes"]

dataset_cluster = {"iris": 3, "wine" : 3, "diabetes" : 2, "housevotes" : 2}

# dbscan combinations are chosen because they produced the best scores
dbscan_comb = {"euclideaniris" : {"minpts" : 3, "eps" : 0.4}, "euclideanwine" : {"minpts" : 18, "eps" : 2.4},
"euclideandiabetes" : {"minpts" : 3, "eps" : 0.1}, "euclideanhousevotes" : {"minpts" : 18, "eps" : 2.5},

"manhattaniris" : {"minpts" : 1, "eps" : 1.2}, "manhattanwine" : {"minpts" : 18, "eps" : 6.9},
"manhattandiabetes" : {"minpts" : 2, "eps" : 0.3}, "manhattanhousevotes" : {"minpts" : 18, "eps" : 6.0},

"chebysheviris" : {"minpts" : 1, "eps" : 0.7}, "chebyshevwine" : {"minpts" : 17, "eps" : 1.3},
"chebyshevdiabetes" : {"minpts" : 1, "eps" : 0.1}, "chebyshevhousevotes" : {"minpts" : 4, "eps" : 0.1},

"cosineiris" : {"minpts" : 1, "eps" : 0.1}, "cosinewine" : {"minpts" : 16, "eps" : 0.3},
"cosinediabetes" : {"minpts" : 10, "eps" : 0.2}, "cosinehousevotes" : {"minpts" : 18, "eps" : 0.2},
}


with open("code/timing.json", 'r') as f:
    timing_results = json.load(f)



for d in datasets:
    table_array = np.zeros((4,4))
    for ik, kalgo in enumerate(kalgos+['DBSCAN']):
        for idd, dist in enumerate(distances):
            table_array[ik, idd] = timing_results[f'{kalgo}{dist}{d}']


    timing_table = pd.DataFrame(table_array, columns=["Euclidean", "Manhattan", "Chebyshev", "Cosine"], index=['K-Means', 'K-Medians', 'K-Medoids']+['DBSCAN'])
    timing_table.loc['Total', :] = timing_table.sum(axis=0)
    timing_table.loc[:, 'Total'] = timing_table.sum(axis=1)
    print(timing_table.to_latex())

