from kmeans import kmeansClustering
from kmedians import kmediansClustering
from kmedoids import kmedoidsClustering
from dbscan import DBSCANClustering

from results import Results

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
                if not results.set_exists(s, c, d, k=k):
                    alg = kalgoclass[c](d, s, seed)
                    alg.load_data()
                    clusters, centers = alg.cluster(k)
                    results.save_set(s, c, d, clusters, centers, k=k)
                    print(f"saved {s}, {c}, {d}, k={k}")
    
    print(f"finished {c} algorithm")


for d in distances:
    for s in datasets:
        for m in [1]:#range(1, 21):
            for e in [0.1]:#[round(0.1 + 0.1*i, 2) for i in range(200)]:
                if not results.set_exists(s, "DBSCAN", d, minpts=m, eps=e):
                    alg = DBSCANClustering(d, s, seed)
                    alg.load_data()
                    clusters, centers = alg.cluster(e, m)
                    results.save_set(s, "DBSCAN", d, clusters, centers, minpts=m, eps=e)
                    print(f"saved {s}, DBSCAN, {d}, minpts={m}, eps={e}")

print(f"finished DBSCAN algorithm")
