import streamlit as st
from kmeans import kmeansClustering
from kmedians import kmediansClustering
from kmedoids import kmedoidsClustering
from dbscan import DBSCANClustering
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

st.set_page_config(page_title="Group 42", page_icon=":koala:")
st.title('Datascience: Group 42')
c = kmeansClustering("manhattan", "wine")


dataset = st.selectbox('Choose a beautiful dataset',['iris', 'wine', 'diabetes', 'solarflare'])
cluster_dist = st.selectbox('Choose an awesome distance measure',['euclidean', 'manhattan', 'chebyshev', 'cosine'])

cluster_algo = st.selectbox('Choose a lovely clustering algorithm',['kmeans', 'kmedians', 'kmedoids', 'DBSCAN'])


cluster_algo_class = {'kmeans': kmeansClustering, 'kmedians': kmediansClustering, 'kmedoids': kmedoidsClustering, 'DBSCAN': DBSCANClustering}

cluster = cluster_algo_class[cluster_algo](cluster_dist, dataset)
cluster.load_data()
if cluster_algo == 'DBSCAN':
    epsilon = st.slider("Choose a nice value for epsilon", min_value=1, max_value=50, step=0.5)
    minpts = st.slider("Choose a minimal number of nearest points", min_value=1, max_value=20, step=1, value=5)
    clusters, stuff = cluster.cluster(epsilon, minpts)
else:
    k_value = st.slider("Choose a nice value for k", min_value=1, max_value=10, step=1, value=3)

    if cluster_algo in  ['kmedoids', 'kmeans']:
        clusters, stuff = cluster.cluster(k_value)
    elif cluster_algo == 'kmedians':
        clusters, stuff = cluster.cluster(k_value, initial_medians=[])

clustered_data = np.zeros(len(cluster.data))
for ic,c in enumerate(clusters):
    clustered_data[c] = ic+1



st.success('Here are the results!!!!')
st.balloons()
col1, col2 = st.beta_columns(2)
col1.header("Projection with TSNE")
perp = col1.slider("Perplexity for TSNE", 5, 50, 5)
projected_data = TSNE(random_state=42, perplexity=perp).fit_transform(cluster.data)
fig, ax = plt.subplots()
sns.scatterplot(x=projected_data[:,0], y=projected_data[:,1], hue=clustered_data, ax=ax)
col1.pyplot(fig)

col2.header("Projection with PCA")
col2.markdown("#")
col2.markdown("#")
projected_data = PCA(random_state=42, n_components=2).fit_transform(cluster.data)
fig, ax = plt.subplots()
sns.scatterplot(x=projected_data[:,0], y=projected_data[:,1], hue=clustered_data, ax=ax)
col2.pyplot(fig)