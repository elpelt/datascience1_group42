"""
@file indices.py
Evaluation Modul to compare clustering results
"""

import numpy as np
from sklearn.metrics import jaccard_score, silhouette_score, adjusted_rand_score, adjusted_mutual_info_score,homogeneity_score, completeness_score

class Indices():
    """
    calculates Indices for computed cluster labels
    uses the scikit library
    """
    def __init__(self, cluster_calc, cluster_label):
        """
        constructor
        @param cluster_calc calculated clustering results
        @param cluster_label expected cluster results
        """
        
        ## calculated clustering results
        self.cluster_calc = cluster_calc

        ## expected cluster results
        self.cluster_label = cluster_label

    def index_external(self, index):
        """
        Function to calculate external index scores<br>
        ARI, AMI, Homogeneity Score and Completeness Score
        @params index string with name of index used ("ARI", "AMI", "Homogeneity Score", "Completeness Score")
        """
        if index == "ARI":
            ari = adjusted_rand_score(self.cluster_label, self.cluster_calc)
            return ari

        elif index == "NMI":
            nmi = adjusted_mutual_info_score(self.cluster_label, self.cluster_calc)
            return nmi

        elif index == "Completeness Score":
            cs = completeness_score(self.cluster_label, self.cluster_calc)
            return cs

        elif index == "Homogeneity Score":
            hs = homogeneity_score(self.cluster_label, self.cluster_calc)
            return hs

        else:
            print("wrong index given")
            return None

    def index_internal(self, index, points, metric):
        """
        Function to calculate internal index scores<br>
        @param index 
        @param points
        @param metric
        """
        if index == "Silhouette Score":
            return silhouette_score(points, self.cluster_calc, metric=metric)

        else:
            print("wrong index given")
            return None

