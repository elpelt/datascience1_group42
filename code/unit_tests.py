import unittest
import numpy as np

from kmeans import kmeansClustering
from kmedians import kmediansClustering
from kmedoids import kmedoidsClustering
from dbscan import DBSCANClustering

from clustering import Clustering

class ClusterTests(unittest.TestCase):
    """ these tests are only used for guaranteeing basic functionality of the library in terms of combining all distance measures with
    all clustering algorithms"""
    def setUp(self):
        self.testarray1 = np.zeros([10, 6])
        self.testarray2 = np.ones([3, 2])

    def test_kmeans(self):
        for distance in ["euclidean", "cosine", "manhattan", "chebyshev"]:
            c = kmeansClustering(distance, "")
            c.data = self.testarray1
            results, centers = c.cluster(1)
            
            self.assertEqual(results, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], 'kmeans failed')
            
            self.assertEqual(centers, [[0, 0, 0, 0, 0, 0]], 'kmeans failed')
        
            c.data = self.testarray2
            results, centers = c.cluster(1)
            
            self.assertEqual(results, [[0, 1, 2]], 'kmeans failed')
            
            self.assertEqual(centers, [[1, 1]], 'kmeans failed')
        
    
    def test_kmedians(self):
        for distance in ["euclidean", "cosine", "manhattan", "chebyshev"]:
            c = kmediansClustering(distance, "")
            c.data = self.testarray1
            results, centers = c.cluster(1)
            
            self.assertEqual(results, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], 'kmedians failed')
            
            self.assertEqual(centers, [[0, 0, 0, 0, 0, 0]], 'kmedians failed')
        
            c.data = self.testarray2
            results, centers = c.cluster(1)
            
            self.assertEqual(results, [[0, 1, 2]], 'kmedians failed')
            
            self.assertEqual(centers, [[1, 1]], 'kmedians failed')
    
    def test_kmedoids(self):
        for distance in ["euclidean", "cosine", "manhattan", "chebyshev"]:
            c = kmedoidsClustering(distance, "")
            c.data = self.testarray1
            results, centers = c.cluster(1)
            
            self.assertEqual(results, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], 'kmedoids failed')
            
            self.assertEqual(centers, [0], 'kmedoids failed')
        
            c.data = self.testarray2
            results, centers = c.cluster(1)
            
            self.assertEqual(results, [[0, 1, 2]], 'kmedoids failed')
            
            self.assertEqual(centers, [0], 'kmedoids failed')
    
    def test_dbscan(self):
        for distance in ["euclidean", "cosine", "manhattan", "chebyshev"]:
            c = DBSCANClustering(distance, "")
            c.data = self.testarray1
            results, noise = c.cluster(1, 2)
            
            self.assertEqual(results, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], 'kmedians failed')
            
            self.assertEqual(noise, [], 'kmedians failed')
        
            c.data = self.testarray2
            results, noise = c.cluster(1, 2)
            
            self.assertEqual(results, [[0, 1, 2]], 'kmedians failed')
            
            self.assertEqual(noise, [], 'kmedians failed')

if __name__ == '__main__':
    unittest.main()
