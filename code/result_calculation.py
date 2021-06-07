from kmeans import kmeansClustering
from kmedians import kmediansClustering
from kmedoids import kmedoidsClustering
from dbscan import DBSCANClustering

kalgos = ['kmeans', 'kmedians', 'kmedoids']
kalgoclass = {'kmeans': kmeansClustering, 'kmedians': kmediansClustering, 'kmedoids': kmedoidsClustering}


distances = ["euclidean", "manhattan", "chebyshev", "cosine"]

datasets = ["iris", "wine", "diabetes", "housevotes"]

seed = 42

results = Results("./results")

for c in kalgos:
    for d in distances:
        for s in datasets:
            for k in range(1, 11):
                alg = kalgoclass[c](d, s, seed)
                clusters, centers = alg.cluster(k)
                results.save_set(s, c, d, clusters, centers, k=k)
                print(f"saved {dataset}, {cluster_algo}, {cluster_dist}, k={k_value}")
    
    print(f"finished {c} algorithm")

for d in distances:
    for s in datasets:
        for m in range(1, 21):
            for e in [round(0.1 + 0.1*i, 2) in range(200)]
                alg = DBSCANClustering(d, s, seed)
                clusters, centers = alg.cluster(k)
                results.save_set(s, c, d, clusters, centers, k=k)
                print(f"saved {dataset}, {cluster_algo}, {cluster_dist}, minpts={minpts}, eps={epsilon}")
print(f"finished DBSCAN algorithm")
