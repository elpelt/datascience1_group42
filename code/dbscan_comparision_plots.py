from dbscan_heuristic import DBSCANHeuristic
import matplotlib.pyplot as plt


distances = ["euclidean", "manhattan", "chebyshev", "cosine"]
datasets = ["iris", "wine", "diabetes", "housevotes"]

def plot_kdist(kdists, dataset):
    """
    plots the sorted kdist graph using matplotlib
    @param k-dist list containing the k-distances for every point of the dataset
    """
    p = [i for i in range(len(kdists[0]))]

    fig, ax = plt.subplots()
    ax.set_title(f"DBSCAN Heuristic for {dataset} dataset, k = 4")

    distcount = 0
    for kdist in kdists:
        kdist.sort(reverse=True)
        ax.plot(p, kdist, label=distances[distcount])
        distcount += 1

    ax.grid(True)
    ax.set_xlabel("Points")
    ax.set_ylabel("k-dist")
    ax.legend(loc="best")
    fig.savefig(dataset)
    return fig


heu = DBSCANHeuristic()
for s in datasets:
    kdists = []
    heu.set_dataset(s)
    for d in distances:
        heu.set_metric(d)
        kdists.append(heu.kdist(4))
    
    plot_kdist(kdists, s)



