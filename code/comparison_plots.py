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

index_ext_eval = ["ARI", "AMI", "Completeness Score", "Homogeneity Score"]
index_int_eval = ["Silhouette Score"]

num_of_classes = [3,3,2,2]

seed = 42

results = Results("./code/results")

for isx,s in enumerate(datasets):
    print(f'start {s}')
    all_kalgos = np.zeros((3,4, 9,len(index_ext_eval)))
    all_kalgos_int = pd.DataFrame(columns=['k', 'Distance (Silhouette Score)', 'Distance (Clustering)', 'sil_score', 'kalgo'])
    for icc, c in enumerate(kalgos):
        all_dist = np.zeros((4, 9,len(index_ext_eval)))
        for id,d in enumerate(distances):
            all_k = np.zeros((9,len(index_ext_eval)))
            for k in range(2, 11):
                clusters, stuff = results.load_set(s, c, d, k=k)
                cluster = kalgoclass[c](d, s, seed)
                cluster.load_data()

                clustered_data = np.zeros(len(cluster.data))
                for ic, cl in enumerate(clusters):
                    clustered_data[cl] = ic + 1

                labels = cluster.labels.tolist()
                predicted = clustered_data.tolist()
                I1 = Indices(predicted, labels)
                index_scores = np.zeros_like(index_ext_eval, dtype=float)
                for i in range(0, len(index_ext_eval)):
                    index_scores[i] = I1.index_external(index_ext_eval[i])
                for i,di in enumerate(distances):
                    index_score = I1.index_internal(index=index_int_eval[0], points=cluster.data.tolist(), metric=di)
                    all_kalgos_int.loc[-1] = [k, di, d, index_score, c]
                    all_kalgos_int.index = all_kalgos_int.index + 1

                all_k[k-2] = index_scores

            all_dist[id] = all_k


        for i in range(0,len(index_ext_eval)):
            if not os.path.exists(f'../plots/{s}/{c}/{index_ext_eval[i]}'):
                os.makedirs(f'../plots/{s}/{c}/{index_ext_eval[i]}')
            fig = plt.figure(figsize=(15, 10))
            for d in range(4):
                plt.plot(range(2, 11), all_dist[d,:,i], 'o-')
            plt.legend(distances, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.title(f'{index_ext_eval[i]}')
            plt.xlabel('k')
            plt.ylabel('score')
            plt.tight_layout()
            plt.savefig(f'../plots/{s}/{c}/{index_ext_eval[i]}/k_1to10.png')
            plt.close(fig)


        if not os.path.exists(f'../plots/{s}/{c}/{index_int_eval[0]}'):
            os.makedirs(f'../plots/{s}/{c}/{index_int_eval[0]}')
        fig,ax = plt.subplots(figsize=(25, 10))
        sns.lineplot(data=all_kalgos_int.loc[all_kalgos_int['kalgo'] == c], x='k', y='sil_score', hue='Distance (Clustering)', style='Distance (Silhouette Score)', ax=ax, legend='full', )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,)
        plt.title(f'{index_int_eval[0]}')
        plt.xlabel('k')
        plt.ylabel('score')
        plt.tight_layout()
        plt.savefig(f'../plots/{s}/{c}/{index_int_eval[0]}/k_1to10.png')
        plt.close(fig)
        print(f'plotted {c}')

        all_kalgos[icc] = all_dist

    for i in range(0, len(index_ext_eval)):
        if not os.path.exists(f'../plots/{s}/combined/{index_ext_eval[i]}'):
            os.makedirs(f'../plots/{s}/combined/{index_ext_eval[i]}')
        fig, ax = plt.subplots(figsize=(15, 10))
        all_kalgos_df = pd.DataFrame(all_kalgos[:,:,num_of_classes[isx]-2,i], columns=distances, index=kalgos)
        all_kalgos_df.plot(kind='bar', ax=ax)
        plt.legend(distances, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Distance (Clustering)")
        plt.title(f'{index_ext_eval[i]}, k = {num_of_classes[isx]}')
        plt.xlabel('algorithm')
        plt.ylabel('score')
        plt.tight_layout()
        plt.savefig(f'../plots/{s}/combined/{index_ext_eval[i]}/algdist_for_given_k.png')
        plt.close(fig)
        print(f'plotted for k = {num_of_classes[isx]}')

    if not os.path.exists(f'../plots/{s}/combined/{index_int_eval[0]}'):
        os.makedirs(f'../plots/{s}/combined/{index_int_eval[0]}')
    fig, ax = plt.subplots(figsize=(15, 10))
    #all_kalgos_int.loc[all_kalgos_int['k'] == num_of_classes[isx]].plot(kind='bar', x='kalgo',y='sil_score', colormap="winter", ax=ax)
    sns.barplot(data=all_kalgos_int.loc[all_kalgos_int['k'] == num_of_classes[isx]], x='kalgo',y='sil_score', hue='Distance (Clustering)', ax=ax)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Distance (Clustering)")
    plt.title(f'{index_int_eval[0]}, k = {num_of_classes[isx]}')
    plt.xlabel('algorithm')
    plt.ylabel('score')
    plt.tight_layout()
    plt.savefig(f'../plots/{s}/combined/{index_int_eval[0]}/algdist_for_given_k.png')
    plt.close(fig)
    print(f'plotted for k = {num_of_classes[isx]}')








