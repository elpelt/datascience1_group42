import numpy as np

from dbscan import DBSCANClustering
from indices import Indices



distances = ["euclidean", "manhattan", "chebyshev", "cosine"]

datasets = ["iris", "wine", "diabetes", "housevotes"]

seed = 42

external = ["ARI", "AMI", "Completeness Score", "Homogeneity Score"]
internal = ["Silhouette Score"]

best_results = {}

def save_score(key, score, m, e):
    try:
        if best_results[key][0] < score:
            best_results[key] = [score, m, e]
    except KeyError:
        best_results[key] = [score, m, e]

for d in distances:
    for s in datasets:
        alg = DBSCANClustering(d, s, seed)
        alg.load_data()
        
        for m in range(1, 21):
            for e in [round(0.1 + 0.1*i, 2) for i in range(200)]:
                clusters, centers = alg.cluster(e, m)
                
                clustered_data = np.zeros(len(alg.data))
                for ic,c in enumerate(clusters):
                    clustered_data[c] = ic+1

                indices = Indices(clustered_data.tolist(), alg.labels.tolist())
                if len(set(clustered_data)) == 1:
                    continue
                elif len(set(clustered_data)) == len(clustered_data):
                    continue
                
                for ind in external:
                    score = indices.index_external(ind)
                    save_score(f"{d}{s}{ind}", score, m, e)
                        
                for ind in internal:
                    score = indices.index_internal(ind, alg.data.tolist(), d)
                    save_score(f"{d}{s}{ind}", score, m, e)

    print(f"finished {s}")

print(best_results)
with open("optimal_solutions.json", 'w') as f:
    json.dump(best_results, f)