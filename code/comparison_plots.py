from indices import Indices
from results import Results
from kmeans import kmeansClustering
from kmedians import kmediansClustering
from kmedoids import kmedoidsClustering
from dbscan import DBSCANClustering
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

kalgos = ['kmeans', 'kmedians', 'kmedoids']
kalgoclass = {'kmeans': kmeansClustering, 'kmedians': kmediansClustering, 'kmedoids': kmedoidsClustering}


distances = ["euclidean", "manhattan", "chebyshev", "cosine"]

datasets = ["iris", "wine", "diabetes", "housevotes"]

index_eval = ["ARI", "NMI", "Completeness Score", "Homogeneity Score"]

seed = 42

results = Results("./results")

for s in datasets:
    print(f'start {s}')
    for c in kalgos:
        all_dist = np.zeros((4, 10,len(index_eval)))
        for id,d in enumerate(distances):
            all_k = np.zeros((10,len(index_eval)))
            for k in range(1, 11):
                clusters, stuff = results.load_set(s, c, d, k=k)
                cluster = kalgoclass[c](d, s, seed)
                cluster.load_data()

                clustered_data = np.zeros(len(cluster.data))
                for ic, cl in enumerate(clusters):
                    clustered_data[cl] = ic + 1

                labels = cluster.labels.tolist()
                predicted = clustered_data.tolist()
                I1 = Indices(labels, predicted)
                index_scores = np.zeros_like(index_eval, dtype=float)
                for i in range(0, 4):
                    index_scores[i] = I1.index_external(index_eval[i])
                all_k[k-1] = index_scores

            all_dist[id] = all_k


        for i in range(0,4):
            if not os.path.exists(f'../plots/{s}/{c}/{index_eval[i]}'):
                os.makedirs(f'../plots/{s}/{c}/{index_eval[i]}')
            plt.figure(figsize=(15, 10))
            for d in range(4):
                plt.plot(range(1, 11), all_dist[d,:,i], )
            plt.legend(distances)
            plt.xlabel('k')
            plt.ylabel('score')
            plt.savefig(f'../plots/{s}/{c}/{index_eval[i]}/k_1to10.png')
        print(f'plotted {c}')






