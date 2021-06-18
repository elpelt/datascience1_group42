from kmeans import kmeansClustering
from kmedians import kmediansClustering
from kmedoids import kmedoidsClustering
from dbscan import DBSCANClustering

from results import Results

from datetime import datetime
import json

kalgos = ['kmeans', 'kmedians', 'kmedoids']
kalgoclass = {'kmeans': kmeansClustering, 'kmedians': kmediansClustering, 'kmedoids': kmedoidsClustering}

distances = ["euclidean", "manhattan", "chebyshev", "cosine"]

dataset_cluster = {"iris": 3, "wine" : 3, "diabetes" : 2, "housevotes" : 2}

timing = {}

n = 10

for c in kalgos:
    for d in distances:
        for s in dataset_cluster.keys():
            time = 0
            alg = kalgoclass[c](d, s)
            alg.load_data()
            for t in range(0, n):
                before = datetime.now()
                clusters, centers = alg.cluster(dataset_cluster[s])
                after = datetime.now()
                time += (after - before).total_seconds()
            
            time = time / n
            timing[f"{c}{d}{s}"] = time
            print(time)

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

for d in distances:
    for s in dataset_cluster.keys():
        time = 0
        alg = DBSCANClustering(d, s)
        alg.load_data()
        for t in range(0, n):
            before = datetime.now()
            clusters, centers = alg.cluster(**dbscan_comb[f"{d}{s}"])
            after = datetime.now()
            time += (after - before).total_seconds()
            
        time = time / n
        timing[f"DBSCAN{d}{s}"] = time
    
                
with open("code/timing.json", 'w') as f:
    json.dump(timing, f)


