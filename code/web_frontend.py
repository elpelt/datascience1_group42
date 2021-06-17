"""
@file web_frontend.py
webfrontend for project.
implemented using streamlit. displays parameter selection, clustering results, 
data and the evaluation module. Charts can be displayed using seaborn or altair
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import altair as alt
import random

from kmeans import kmeansClustering
from kmedians import kmediansClustering
from kmedoids import kmedoidsClustering
from dbscan import DBSCANClustering
from indices import Indices
from results import Results
import SessionState

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Group 42", page_icon=":koala:")
st.title('Datascience: Group 42')

# session state for saving parameters for every in browser opened instance
session_state = SessionState.get(indices_data=pd.DataFrame())

seeded = st.checkbox('Use precalculated results (with random seed for reproduction).', value=True)
seed = None

seaplots = st.checkbox('Use interactive charts', value=True)


# result handler for set seed clusters
resulthandler = Results("./code/results")

if seeded:
    seed = 42

# Settings tab
st.header("Settings")
col1, col2 = st.beta_columns(2)
dataset = col1.selectbox('Choose a beautiful dataset',['iris', 'wine', 'diabetes', 'housevotes'])

cluster_dist_desc = {'euclidean': 'd(x,y) = \sqrt{\sum_{i=1}^{n}(|x_i-y_i|)^2}',
                     'manhattan': 'd(x,y) = \sum\limits_{i=1}^{n}|x_i - y_i|',
                     'chebyshev': 'd(x,y) = \max(|x_i - y_i|)',
                     'cosine': 'd(x,y) = \\frac{\\arccos(\\frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum_{i=1}^{n} x_i^2 \sum_{i=1}^{n} y_i^2}})}{\pi}'}
cluster_dist = col1.selectbox('Choose an awesome distance measure', list(cluster_dist_desc.keys()))
col1.latex(cluster_dist_desc[cluster_dist])

cluster_algo_class = {'kmeans': kmeansClustering, 'kmedians': kmediansClustering, 'kmedoids': kmedoidsClustering, 'DBSCAN': DBSCANClustering}
cluster_algo = col2.selectbox('Choose a lovely clustering algorithm', list(cluster_algo_class.keys()))

params = {}
if cluster_algo == 'DBSCAN':
    epsilon = col2.slider("Choose a nice value for epsilon", min_value=0.1, max_value=20.0, step=0.1)
    minpts = col2.slider("Choose a minimal number of nearest points", min_value=1, max_value=20, step=1, value=5)
    params = {"eps" : epsilon, "minpts" : minpts}
    st.write("*DBSCAN heuristic for estimating minPts and eps parameters: https://share.streamlit.io/elpelt/datascience1_group42/main/code/heuristic_web.py*")

else:
    k_value = col2.slider("Choose a nice value for k (number of clusters)", min_value=2, max_value=10, step=1, value=3)
    params = {"k" : k_value}

@st.cache()
def create_cluster(cluster_algo, cluster_dist, dataset, seed):
    """
    creates a cluster algorithm instance. takes all parameters needed for creating such instance. uses the streamlit caching decorator
    @param cluster_algo string containing name of cluster algorithm used ("kmeans", "kmedians", "kmedoids", "DBSCAN")
    @param cluster_dist string containing the distance measure used ("euclidean", "manhattan", "chebyshev", "cosine")
    @param dataset string containing the datasets name ("iris", "wine", "diabetes", "housevotes")
    @param seed seed for cluster algorithm, None if a random seed should be used
    @returns cluster results, cluster centers, clustered data
    """
    cluster = cluster_algo_class[cluster_algo](cluster_dist, dataset, seed)
    cluster.load_data()
    return cluster

cluster = create_cluster(cluster_algo, cluster_dist, dataset, seed)

@st.cache()
def clustering(cluster, params):
    """
    calculates clustering results. uses the streamlit caching decorator
    @param cluster cluster algorithm object
    @param params dictionary containing parameters needed for the cluster algorithm, either k or minpts and eps
    @returns cluster results, cluster centers, clustered data
    """
    if seeded and resulthandler.set_exists(dataset, cluster_algo, cluster_dist, **params):
        clusters, centers = resulthandler.load_set(dataset, cluster_algo, cluster_dist, **params)
        print(f"loaded {dataset}, {cluster_algo}, {cluster_dist}, {params}")
    
    else:
        clusters, centers = cluster.cluster(epsilon, minpts)
        
        if seeded:
            resulthandler.save_set(dataset, cluster_algo, cluster_dist, clusters, centers, **params)
            print(f"saved {dataset}, {cluster_algo}, {cluster_dist}, {params}")
    
    clustered_data = np.zeros(len(cluster.data))
    for ic,c in enumerate(clusters):
        clustered_data[c] = ic+1

    return clusters, centers, clustered_data

clusters, centers, clustered_data = clustering(cluster, params)

st.success('Great choice! Here are the results!!!!')

# Projections
col1, col2 = st.beta_columns(2)
col1.header("Projection with t-SNE")
perp = col1.slider("Perplexity for t-SNE", 5, 50, 25)
col1.write("t-SNE is a nonlinear dimension reduction. The outcome will depend on the perplexity you have chosen. ")


col2.header("Projection with PCA")
col2.markdown("#")
col2.write("PCA is a linear dimension reduction. The data will be projected on the first 2 principal components, "
           "which capture the most variance in the data. ")

if cluster_algo == 'kmedoids' and k_value>1:
    marking_centroids = np.ones(cluster.data.shape[0])
    marking_centroids[centers] = 25

# actual projecting and plot generating
col1, col2 = st.beta_columns(2)

dfclusterdata = pd.DataFrame()
dfclusterdata["c"] = clustered_data
dfclusterdata["i"] = [i for i in range(len(clustered_data))]

@st.cache(allow_output_mutation=True)
def plotting():
    """
    generates the plots for the frontend. uses the streamlit chacheing decorator
    @returns TSNE and PCA projections of clustering results either as seaborn or altair plots
    """
    projected_data_tsne = TSNE(random_state=42, perplexity=perp).fit_transform(cluster.data)
    projected_data_pca = PCA(random_state=42, n_components=2).fit_transform(cluster.data)

    if not seaplots:
        # seaborn color palette
        if cluster_algo == "DBSCAN":
            color_palette = ['black'] + sns.color_palette("husl", len(set(clustered_data))-1)
        else:
            color_palette = sns.color_palette("husl", len(set(clustered_data)))

        # plot building
        fig1, ax1 = plt.subplots()
        if cluster_algo == 'kmedoids' and k_value>1:
            sns.scatterplot(x=projected_data_tsne[:, 0], y=projected_data_tsne[:, 1], hue=clustered_data, ax=ax1,
                            palette=color_palette, legend=False, style=marking_centroids, size=marking_centroids*30, markers=["o", "P"])
        else:
            sns.scatterplot(x=projected_data_tsne[:,0], y=projected_data_tsne[:,1], hue=clustered_data, ax=ax1, palette=color_palette, legend=False)
        
        fig2, ax2 = plt.subplots()
        
        if cluster_algo == 'kmedoids' and k_value>1:
            sns.scatterplot(x=projected_data_pca[:, 0], y=projected_data_pca[:, 1], hue=clustered_data, ax=ax2,
                                palette=color_palette, legend=False, style=marking_centroids, size=marking_centroids*30, markers=["o", "P"])
        else:
            sns.scatterplot(x=projected_data_pca[:, 0], y=projected_data_pca[:, 1], hue=clustered_data, ax=ax2,
                            palette=color_palette, legend=False)
        return fig1, fig2

    else:
        dfclusterdata[["xt", "yt"]] = pd.DataFrame(projected_data_tsne)
        dfclusterdata[["xp", "yp"]] = pd.DataFrame(projected_data_pca)
        dfclusterdata["c"] = clustered_data
        dfclusterdata["i"] = [i for i in range(len(clustered_data))]

        # plot building
        cluster_label = alt.Tooltip("c", title="Cluster ID")
        point_label = alt.Tooltip("i", title="Point ID")
        altcolor=alt.Color("c", legend=None, scale=alt.Scale(domain=[0, 1 if max(clustered_data) == 0 else max(clustered_data)], scheme="turbo"))

        xaxis = alt.X("xt", axis=alt.Axis(title=None))
        yaxis = alt.Y("yt", axis=alt.Axis(title=None))
        tsnealt = alt.Chart(dfclusterdata).mark_circle().encode(x=xaxis, y=yaxis, tooltip=[cluster_label, point_label], color=altcolor).interactive()

        xaxis = alt.X("xp", axis=alt.Axis(title=None))
        yaxis = alt.Y("yp", axis=alt.Axis(title=None))
        pcaalt = alt.Chart(dfclusterdata).mark_circle().encode(x=xaxis, y=yaxis, tooltip=[cluster_label, point_label], color=altcolor).interactive()
            
        return tsnealt, pcaalt

with st.spinner('Please wait a second. Some colorful plots are generated...'):
    fig1, fig2 = plotting()

    if seaplots:
        col1.altair_chart(fig1, use_container_width=True)
        col2.altair_chart(fig2, use_container_width=True)

    else:
        col1.pyplot(fig1)
        col2.pyplot(fig2)

if cluster_algo == 'DBSCAN':
    st.write()
    st.write(f"*Please notice for the DBSCAN clustering algorithm: Data points classified as noise are plotted as black points*. In this clustering {np.count_nonzero(clustered_data == 0)} points are marked as noise")

dataexpander = st.beta_expander("data")
#cluster.datadf["cluster ID"] = clustered_data
dataexpander.write(cluster.datadf)

# Clustering evaluation
st.header("Clustering evaluation")
st.write("Clustering results can be stored in a cluster-table and used for comparative evaluation.")

# generates button to add or reset the calculated clustering
col1, col2 = st.beta_columns(2)
add_result = col1.button('Add')
reset_tmp = col2.button('Reset')

# write empty CSV to clear CSV
if reset_tmp:
    session_state.indices_data = pd.DataFrame()
    st.write("Cluster-table cleared succesfully!")

# add clustering result to CSV with new column and characteristics as header
if add_result:

    if len(set(clustered_data)) ==1:
        st.warning('All datapoints belong to the same cluster. Please choose different parameter settings to get a more useful clustering.')

    else:
        # epsilon or k, depends on clustering algorithm
        if cluster_algo == "DBSCAN":
            val="epsilon="+str(epsilon)+", np="+str(minpts)
        else:
            val="k="+str(k_value)

        df = session_state.indices_data

        if str((cluster_algo, cluster_dist, val, dataset)) in df.columns:
            st.write("Cluster", (cluster_algo, cluster_dist, val, dataset), "already in cluster-table!")
        else:
            labels = cluster.labels.tolist()
            predicted = clustered_data.tolist()
            precalc = []
            index_eval = ["ARI", "AMI", "Completeness Score", "Homogeneity Score", "Silhouette Score"]
            for i in range(0,4):
                I1 = Indices(predicted, labels)
                score = I1.index_external(index_eval[i])
                precalc.append(score)
            precalc.append(I1.index_internal(index_eval[4], cluster.data.tolist(), cluster_dist))
            df[(cluster_algo, cluster_dist, val, dataset)] = pd.Series(precalc)
            st.write("Cluster", (cluster_algo, cluster_dist, val, dataset), "added succesfully!")

if not session_state.indices_data.empty:
    st.write("The cluster-table contains the following cluster:", session_state.indices_data.columns)
else:
    st.write("Cluster-table is empty!")

index_eval = st.selectbox('Choose an adorable index',["ARI", "AMI", "Completeness Score", "Homogeneity Score", "Silhouette Score"])

datasets = []
# iterate over cluster results and calculate score with chosen index
results = [[1, "maximum reference value"]]
if not session_state.indices_data.empty:
    df = session_state.indices_data
    if index_eval in ["ARI", "AMI", "Completeness Score", "Homogeneity Score", "Silhouette Score"]:
        for i in range(0, len(df.columns)):
            if index_eval == "ARI":
                results.append([df.iloc[:,i].values[0], df.columns[i]])
            elif index_eval == "AMI":
                results.append([df.iloc[:, i].values[1], df.columns[i]])
            elif index_eval == "Completeness Score":
                results.append([df.iloc[:,i].values[2], df.columns[i]])
            elif index_eval == "Silhouette Score":
                results.append([df.iloc[:, i].values[4], df.columns[i]])
            else:
                results.append([df.iloc[:, i].values[3], df.columns[i]])
    for i in range(0, len(results)):
        st.write(results[i][0], results[i][1])
    datasets = []
    for i in range(1, len(results)):
        results[i][1] = str(results[i][1]).replace("(", "").replace(")", "").replace("'", "").split(",")
        if results[i][1][0] == "DBSCAN":
            if (results[i][1][4] not in datasets):
                datasets.append(results[i][1][4])
        elif results[i][1][3] not in datasets:
            datasets.append(results[i][1][3])

# if list is empty or two diff. datasets were chosen
else:
    st.write("")

# preprocess radar plot
desc_list = []
if len(datasets) == 0:
    st.write("Plot not possible.")

else:
    if len(datasets) == 1:
        for j in range(0, len(results)):
            if (j != 0) and (results[j][1][0] == "DBSCAN"):
                desc_list.append(str(results[j][1][0:4]))
            elif j != 0:
                desc_list.append(str(results[j][1][0:3]))
            else:
                desc_list.append(str(results[j][1]))
    else:
        for j in range(0, len(results)):
            desc_list.append(str(results[j][1]))

desc = np.array(desc_list)
stats = np.zeros(len(results))
for i in range(0, len(results)):
    stats[i] = results[i][0]

# if only reference value is in results
if len(results) <= 1:
    pass

# if length of results smaller than 3, barplot instead of radar chart
elif len(results) <= 2:
    fig, ax = plt.subplots()
    labels, ys = list(desc), list(stats)
    xs = np.arange(len(labels))
    width = 0.5
    ax.bar(xs[0], ys[0], width, align='center', color='red')
    ax.bar(xs[1:], ys[1:], width, align='center')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_yticks(ys)
    st.pyplot(fig)

#if length of results higher than 2 radar chart
else:
    # define angles
    angles=np.linspace(0, 2*np.pi, len(desc), endpoint=False)
    stats=np.concatenate((stats,[stats[0]]))
    angles=np.concatenate((angles,[angles[0]]))

    # print radar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.plot(angles[0], stats[0], 'o-', linewidth=2, color='red')
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids((angles * 180 / np.pi)[0:len(results)], desc)
    if len(datasets) == 1:
        ax.set_title("Index:"+" "+index_eval+","+" "+"Dataset:"+" "+datasets[0])
    else:
        ax.set_title("Index:" + " " + index_eval)
    ax.grid(True)
    st.pyplot(fig)
