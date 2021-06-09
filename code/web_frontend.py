import streamlit as st
import pandas as pd
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
from results import Results

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Group 42", page_icon=":koala:")
st.title('Datascience: Group 42')

seeded = st.checkbox('Use precalculated results (with random seed for reproduction).', value=True)
seed = None

# result handler for set seed clusters
results = Results("./code/results")

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
cluster_dist = col1.selectbox('Choose an awesome distance measure',list(cluster_dist_desc.keys()))
col1.latex(cluster_dist_desc[cluster_dist])

cluster_algo_class = {'kmeans': kmeansClustering, 'kmedians': kmediansClustering, 'kmedoids': kmedoidsClustering, 'DBSCAN': DBSCANClustering}
cluster_algo = col2.selectbox('Choose a lovely clustering algorithm',list(cluster_algo_class.keys()))

cluster = cluster_algo_class[cluster_algo](cluster_dist, dataset, seed)
cluster.load_data()


# algorithm specific parameter selection and clustering
if cluster_algo == 'DBSCAN':
    epsilon = col2.slider("Choose a nice value for epsilon", min_value=0.1, max_value=20.0, step=0.1)
    minpts = col2.slider("Choose a minimal number of nearest points", min_value=1, max_value=20, step=1, value=5)
    
    if seeded and results.set_exists(dataset, cluster_algo, cluster_dist, minpts=minpts, eps=epsilon):
        clusters, stuff = results.load_set(dataset, cluster_algo, cluster_dist, minpts=minpts, eps=epsilon)
        print(f"loaded {dataset}, {cluster_algo}, {cluster_dist}, minpts={minpts}, eps={epsilon}")
    
    else:
        clusters, stuff = cluster.cluster(epsilon, minpts)
        
        if seeded:
            results.save_set(dataset, cluster_algo, cluster_dist, clusters, stuff, minpts=minpts, eps=epsilon)
            print(f"saved {dataset}, {cluster_algo}, {cluster_dist}, minpts={minpts}, eps={epsilon}")

else:
    k_value = col2.slider("Choose a nice value for k (number of clusters)", min_value=1, max_value=10, step=1, value=3)
    
    if seeded and results.set_exists(dataset, cluster_algo, cluster_dist, k=k_value):
        clusters, stuff = results.load_set(dataset, cluster_algo, cluster_dist, k=k_value)
        print(f"loaded {dataset}, {cluster_algo}, {cluster_dist}, k={k_value}")
    
    else:
        clusters, stuff = cluster.cluster(k_value)
        
        if seeded:
            results.save_set(dataset, cluster_algo, cluster_dist, clusters, stuff, k=k_value)
            print(f"saved {dataset}, {cluster_algo}, {cluster_dist}, k={k_value}")

clustered_data = np.zeros(len(cluster.data))
for ic,c in enumerate(clusters):
    clustered_data[c] = ic+1

if cluster_algo == 'DBSCAN':
    color_palette = ['black'] + sns.color_palette("husl", len(set(clustered_data))-1)
else:
    color_palette = sns.color_palette("husl", len(set(clustered_data)))

st.success('Great choice! Here are the results!!!!')
st.balloons()

# Projections
col1, col2 = st.beta_columns(2)
col1.header("Projection with t-SNE")
perp = col1.slider("Perplexity for t-SNE", 5, 50, 25)
col1.write("t-SNE is a nonlinear dimension reduction. The outcome will depend on the perplexity you have chosen. ")


col2.header("Projection with PCA")
col2.markdown("#")
col2.write("PCA is a linear dimension reduction. The data will be projected on the first 2 principal components, "
           "which capture the most variance in the data. ")

if cluster_algo == 'kmedoids':
    marking_centroids = np.ones(cluster.data.shape[0])
    for centroid in stuff:
        marking_centroids[cluster.data.tolist().index(centroid)] = 25
    print(marking_centroids)


# actual projecting and plot generating
col1, col2 = st.beta_columns(2)
fig, ax = plt.subplots()
with st.spinner('Please wait a second. Some colorful plots are generated...'):
    projected_data_tsne = TSNE(random_state=42, perplexity=perp).fit_transform(cluster.data)
    if cluster_algo == 'kmedoids':
        sns.scatterplot(x=projected_data_tsne[:, 0], y=projected_data_tsne[:, 1], hue=clustered_data, ax=ax,
                        palette=color_palette, legend=False, style=marking_centroids, size=marking_centroids*30, markers=["o", "P"])
    else:
        sns.scatterplot(x=projected_data_tsne[:,0], y=projected_data_tsne[:,1], hue=clustered_data, ax=ax, palette=color_palette, legend=False)
col1.pyplot(fig)


fig, ax = plt.subplots()
with st.spinner('Please wait a second. Some colorful plots are generated...'):
    projected_data_pca = PCA(random_state=42, n_components=3).fit_transform(cluster.data)
    if cluster_algo == 'kmedoids':
        sns.scatterplot(x=projected_data_pca[:, 0], y=projected_data_pca[:, 1], hue=clustered_data, ax=ax,
                        palette=color_palette, legend=False, style=marking_centroids, size=marking_centroids*30, markers=["o", "P"])
    else:
        sns.scatterplot(x=projected_data_pca[:, 0], y=projected_data_pca[:, 1], hue=clustered_data, ax=ax,
                        palette=color_palette, legend=False)
col2.pyplot(fig)

if cluster_algo == 'DBSCAN':
    st.write("*Please notice for the DBSCAN clustering algorithm: Data points classified as noise are plotted as black points*")


# Clustering evaluation
st.header("Clustering evaluation")
st.write("Clustering results can be stored in a cluster-table and used for comparative evaluation.")

# generates button to add or reset the calculated clustering
col1, col2 = st.beta_columns(2)
add_result = col1.button('Add')
reset_tmp = col2.button('Reset')

# write empty CSV to clear CSV
if reset_tmp:
    with open("tmp.csv", "w") as f:
        f.write('')
    st.write("Cluster-table cleared succesfully!")

# add clustering result to CSV with new column and characteristics as header
if add_result:
    # epsilon or k, depends on clustering algorithm
    if cluster_algo == "DBSCAN":
        val="epsilon="+str(epsilon)
    else:
        val="k="+str(k_value)
    try:
        df = pd.read_csv("tmp.csv", delimiter=",")
        if str((cluster_algo, cluster_dist, val, dataset)) in df.columns:
            st.write("Cluster", (cluster_algo, cluster_dist, val, dataset), "already in cluster-table!")
        else:
            labels = cluster.labels.tolist()
            predicted = clustered_data.tolist()
            precalc = []
            index_eval = ["ARI", "NMI", "Completeness Score", "Homogeneity Score"]
            for i in range(0,4):
                I1 = Indices(predicted, labels)
                score = I1.index_external(index_eval[i])
                precalc.append(score)
            df[(cluster_algo, cluster_dist, val, dataset)] = pd.Series(precalc)
            df.to_csv("tmp.csv", sep=",", index=False)
            st.write("Cluster", (cluster_algo, cluster_dist, val, dataset), "added succesfully!")
    # if csv is empty
    except:
        labels = cluster.labels.tolist()
        predicted = clustered_data.tolist()
        precalc = []
        index_eval = ["ARI", "NMI", "Completeness Score", "Homogeneity Score"]
        for i in range(0, 4):
            I1 = Indices(predicted, labels)
            score = I1.index_external(index_eval[i])
            precalc.append(score)
        df = pd.DataFrame(precalc, columns=[(cluster_algo, cluster_dist, val, dataset)])
        df.to_csv("tmp.csv", sep=",", index=False)
        st.write("Cluster", (cluster_algo, cluster_dist, val, dataset), "added succesfully!")
try:
    df = pd.read_csv("tmp.csv", delimiter=",")
    st.write("The cluster-table contains the following cluster:", df.columns)
except:
    st.write("Cluster-table is empty!")

index_eval = st.selectbox('Choose an adorable index',["ARI", "NMI", "Completeness Score", "Homogeneity Score"])


# iterate over cluster results and calculate score with chosen index
try:
    results = [[1, "maximum reference value"]]
    df = pd.read_csv("tmp.csv", delimiter=",")
    if index_eval in ["ARI", "NMI", "Completeness Score", "Homogeneity Score"]:
        for i in range(0, len(df.columns)):
            if index_eval == "ARI":
                results.append([df.iloc[:,i].values[0], df.columns[i]])
            elif index_eval == "NMI":
                results.append([df.iloc[:, i].values[1], df.columns[i]])
            elif index_eval == "Completeness Score":
                results.append([df.iloc[:,i].values[2], df.columns[i]])
            else:
                results.append([df.iloc[:, i].values[3], df.columns[i]])
    for i in range(0, len(results)):
        st.write(results[i][0], results[i][1])
    datasets = []
    for i in range(1, len(results)):
        results[i][1] = results[i][1].replace("(", "").replace(")", "").replace("'", "").split(",")
        if results[i][1][3] not in datasets:
            datasets.append(results[i][1][3])

# if list is empty or two diff. datasets were chosen
except:
    st.write("Cluster-table is empty.")


try:
    # preprocess radar plot
    desc_list = []
    if len(datasets) == 1:
        for j in range(0, len(results)):
            if j != 0:
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

    # if length of results smaller than 3, barplot instead of radar chart
    if len(results) <= 2:
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
            ax.set_title("Index:"+" "+index_eval+","+" "+"Dataset:"+" "+dataset)
        else:
            ax.set_title("Index:" + " " + index_eval)
        ax.grid(True)
        st.pyplot(fig)

# if list is empty
except Exception as er:
    print(er)
    st.write("Plot not possible.")
