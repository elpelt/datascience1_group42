import streamlit as st
from kmeans import kmeansClustering
from kmedians import kmediansClustering
from kmedoids import kmedoidsClustering
from dbscan import DBSCANClustering
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

st.title('Datascience: Group 42')
c = kmeansClustering("manhattan", "wine")


dataset = st.selectbox('Choose a beautiful dataset',['iris', 'wine', 'diabetes', 'solarflare'])
cluster_dist = st.selectbox('Choose an awesome distance measure',['euclidean', 'manhattan', 'chebyshev', 'cosine'])

cluster_algo = st.selectbox('Choose a lovely clustering algorithm',['kmeans', 'kmedians', 'kmedoids', 'DBSCAN'])


cluster_algo_class = {'kmeans': kmeansClustering, 'kmedians': kmediansClustering, 'kmedoids': kmedoidsClustering, 'DBSCAN': DBSCANClustering}

cluster = cluster_algo_class[cluster_algo](cluster_dist, dataset)
cluster.load_data()
if cluster_algo == 'DBSCAN':
    epsilon = st.slider("Choose a nice value for epsilon", 1, 50, 3)
    minpts = st.slider("Choose a minimal number of nearest points", 1, 20, 1)
    clusters, stuff = cluster.cluster(epsilon, minpts)
else:
    k_value = st.slider("Choose a nice value for k", 1, 10, 1)

    if cluster_algo in  ['kmedoids', 'kmeans']:
        clusters, stuff = cluster.cluster(k_value, plusplus=True)
    elif cluster_algo == 'kmedians':
        clusters, stuff = cluster.cluster(k_value, initial_medians=[])

clustered_data = np.zeros(len(cluster.data))
for ic,c in enumerate(clusters):
    print(ic)
    clustered_data[c] = ic+1
print(len(clustered_data))
print(clustered_data)


st.text('Here are the results!!!!')
st.balloons()
perp = st.slider("Perplexity for TSNE", 5, 50, 5)
projected_data = TSNE(random_state=42, perplexity=perp).fit_transform(cluster.data)
fig, ax = plt.subplots()
sns.scatterplot(x=projected_data[:,0], y=projected_data[:,1], hue=clustered_data, ax=ax)
st.pyplot(fig)