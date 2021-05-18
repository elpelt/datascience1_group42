import numpy as np
from sklearn.metrics import jaccard_score, silhouette_score

class Indices():
    def __init__(self, cluster_calc, cluster_label):
        self.cluster_calc = cluster_calc
        self.cluster_label = cluster_label

    def index_external(self, index):
        if index == "jaccard":
            jacc = jaccard_score(self.cluster_calc, self.cluster_label, average='micro')
            return jacc

        elif index == "franzi":
            pass

        else:
            print("wrong index given")
            return None

    def index_internal(self, index):
        if index == "silhouette":
            return silhouette_score(self.cluster_calc, self.cluster_label)

        else:
            print("wrong index given")
            return None

