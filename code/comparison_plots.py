from indices import Indices
from results import Results
from kmeans import kmeansClustering
from kmedians import kmediansClustering
from kmedoids import kmedoidsClustering
from dbscan import DBSCANClustering
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 30})

kalgos = ['kmeans', 'kmedians', 'kmedoids']
kalgoclass = {'kmeans': kmeansClustering, 'kmedians': kmediansClustering, 'kmedoids': kmedoidsClustering}


distances = ["euclidean", "manhattan", "chebyshev", "cosine"]

datasets = ["iris", "wine", "diabetes", "housevotes"]

index_eval = ["ARI", "NMI", "Completeness Score", "Homogeneity Score"]

num_of_classes = [3,3,2,2]

seed = 42

results = Results("./results")

for isx,s in enumerate(datasets):
    print(f'start {s}')
    all_kalgos = np.zeros((3,4, 10,len(index_eval)))
    for icc, c in enumerate(kalgos):
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
            fig = plt.figure(figsize=(15, 10))
            for d in range(4):
                plt.plot(range(1, 11), all_dist[d,:,i], 'o-')
            plt.legend(distances)
            plt.title(f'{index_eval[i]}')
            plt.xlabel('k')
            plt.ylabel('score')
            plt.tight_layout()
            plt.savefig(f'../plots/{s}/{c}/{index_eval[i]}/k_1to10.png')
            plt.close(fig)
        print(f'plotted {c}')

        all_kalgos[icc] = all_dist

    for i in range(0, 4):
        if not os.path.exists(f'../plots/{s}/combined/{index_eval[i]}'):
            os.makedirs(f'../plots/{s}/combined/{index_eval[i]}')
        fig, ax = plt.subplots(figsize=(15, 10))
        all_kalgos_df = pd.DataFrame(all_kalgos[:,:,num_of_classes[isx],i], columns=distances, index=kalgos)
        all_kalgos_df.plot(kind='bar', colormap="winter", ax=ax)
        plt.legend(distances)
        plt.title(f'{index_eval[i]}, k = {num_of_classes[isx]}')
        plt.xlabel('algorithm')
        plt.ylabel('score')
        plt.tight_layout()
        plt.savefig(f'../plots/{s}/combined/{index_eval[i]}/algdist_for_given_k.png')
        plt.close(fig)
        print(f'plotted for k = {num_of_classes[isx]}')








