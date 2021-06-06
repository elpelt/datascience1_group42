import numpy as np
from sklearn.metrics import jaccard_score, silhouette_score, adjusted_rand_score, adjusted_mutual_info_score,homogeneity_score, completeness_score

class Indices():
    """
    calculates Indices for computed cluster labels
    uses the scikit library
    """
    def __init__(self, cluster_calc, cluster_label):
        self.cluster_calc = cluster_calc
        self.cluster_label = cluster_label

    def index_external(self, index):
        """
        Function to calculate external index scores
        ARI, AMI, Homogeneity Score and Completeness Score
        """
        if index == "ARI":
            ari = adjusted_rand_score(self.cluster_calc, self.cluster_label)
            return ari

        elif index == "NMI":
            nmi = adjusted_mutual_info_score(self.cluster_calc, self.cluster_label)
            return nmi

        elif index == "Completeness Score":
            cs = completeness_score(self.cluster_calc, self.cluster_label)
            return cs

        elif index == "Homogeneity Score":
            hs = homogeneity_score(self.cluster_calc, self.cluster_label)
            return hs

        else:
            print("wrong index given")
            return None

    def index_internal(self, index):
        if index == "silhouette":
            return silhouette_score(self.cluster_calc, self.cluster_label)

        else:
            print("wrong index given")
            return None

