import streamlit as st
from kmeans import kmeansClustering
from kmedians import kmediansClustering
from kmedoids import kmedoidsClustering
from dbscan import DBSCANClustering
from indices import Indices
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score

st.set_page_config(page_title="Group 42", page_icon=":koala:")
st.title('Datascience: Group 42')

st.header("Settings")
col1, col2 = st.beta_columns(2)
dataset = col1.selectbox('Choose a beautiful dataset',['iris', 'wine', 'diabetes', 'solarflare'])
cluster_dist = col1.selectbox('Choose an awesome distance measure',['euclidean', 'manhattan', 'chebyshev', 'cosine'])

cluster_algo = col2.selectbox('Choose a lovely clustering algorithm',['kmeans', 'kmedians', 'kmedoids', 'DBSCAN'])


cluster_algo_class = {'kmeans': kmeansClustering, 'kmedians': kmediansClustering, 'kmedoids': kmedoidsClustering, 'DBSCAN': DBSCANClustering}

cluster = cluster_algo_class[cluster_algo](cluster_dist, dataset)
cluster.load_data()

if cluster_algo == 'DBSCAN':
    epsilon = col2.slider("Choose a nice value for epsilon", min_value=1.0, max_value=2.0, step=0.1)
    minpts = col2.slider("Choose a minimal number of nearest points", min_value=1, max_value=20, step=1, value=5)
    clusters, stuff = cluster.cluster(epsilon, minpts)
else:
    k_value = col2.slider("Choose a nice value for k", min_value=1, max_value=10, step=1, value=3)

    if cluster_algo in  ['kmedoids', 'kmeans']:
        clusters, stuff = cluster.cluster(k_value)
    elif cluster_algo == 'kmedians':
        clusters, stuff = cluster.cluster(k_value, initial_medians=[])

clustered_data = np.zeros_like(cluster.labels)
for ic,c in enumerate(clusters):
    clustered_data[c] = ic+1

color_palette = sns.color_palette("husl", len(set(clustered_data)))

st.success('Here are the results!!!!')
st.balloons()
col1, col2 = st.beta_columns(2)
col1.header("Projection with TSNE")
perp = col1.slider("Perplexity for TSNE", 5, 50, 5)
col1.write("TSNE is a nonlinear dimension reduction. The outcome will depend on the perplexity you have chosen. ")
projected_data_tsne = TSNE(random_state=42, perplexity=perp).fit_transform(cluster.data)

col2.header("Projection with PCA")
col2.markdown("#")
col2.write("PCA is a linear dimension reduction. The data will be projected on the first 2 principal components. ")


col1, col2 = st.beta_columns(2)
fig, ax = plt.subplots()
sns.scatterplot(x=projected_data_tsne[:,0], y=projected_data_tsne[:,1], hue=clustered_data, ax=ax, palette=color_palette)
col1.pyplot(fig)

projected_data_pca = PCA(random_state=42, n_components=3).fit_transform(cluster.data)
fig, ax = plt.subplots()
sns.scatterplot(x=projected_data_pca[:,0], y=projected_data_pca[:,1], hue=clustered_data, ax=ax, palette=color_palette)
col2.pyplot(fig)

if cluster_algo == 'DBSCAN':
    st.write("*Please notice for DBSCAN clustering algorithm: Noise is labeled with 0 in the plots.*")

st.header("Clustering evaluation")
true_labels = cluster.labels
pred_labels = clustered_data
st.write('homogeneity_score', homogeneity_score(true_labels, pred_labels))
st.write('completeness_score', completeness_score(true_labels, pred_labels))