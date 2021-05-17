import numpy as np
from sklearn.metrics import jaccard_score

class Indices():
    def __init__(self, cluster_calc, cluster_label):
        self.cluster_calc = cluster_calc
        self.cluster_label = cluster_label

    def index_external(self, index):
        if index == "jaccard":
            jacc = jaccard_score(self.cluster_calc, self.cluster_label, average="samples")
            print(jacc)
            return jacc

        elif index == "franzi":
            pass

        else:
            print("wrong index given")
            return None

    def index_internal(self, index):
        if index == "jonas":
            pass

        else:
            print("wrong index given")
            return None

if __name__ == "__main__":
    #TEST
    #Example arrays
    x = np.array([[0, 1, 1],[1, 1, 0]])
    y = np.array([[1, 1, 1],[1, 0, 0]])

    #jaccard score for x and y
    I1 = Indices(x, y)
    I1.index_external("jaccard")
