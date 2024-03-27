"""
Author: Nitish Sanghi
Date: 2024-03-27

Description:
This code contains a series of unit tests for validating the functionality of various 
classes related to clustering algorithms and point cloud processing. The tests are organized 
into test classes such as `TestCluster`, `TestAgglomerativeClustering`, `TestDBSCANClustering`, 
and `TestConcentricZoneModel`, each focusing on different aspects of the implemented algorithms.

Dependencies:
- unittest
- numpy
- agglomerative_clustering (imported classes: Cluster, AgglomerativeClustering)
- dbscan (imported classes: Category, DBSCANClustering)
- concentric_zone_model (imported classes: ConcentricZoneModel, Bin)
"""


import unittest
import numpy as np
from  agglomerative_clustering import Cluster, AgglomerativeClustering
from dbscan import Category, DBSCANClustering
from concentric_zone_model import ConcentricZoneModel, Bin

import unittest

class TestCluster(unittest.TestCase):
    def test_calculate_centroid(self):
        points = np.array([[1, 2, 3, 1], [4, 5, 6, 2], [7, 8, 9, 3]])
        cluster = Cluster(points, 1)
        self.assertTrue(np.array_equal(cluster.calculate_centroid(), np.array([4, 5, 6])))

class TestAgglomerativeClustering(unittest.TestCase):
    def test_reset_state(self):
        agglomerative_clustering = AgglomerativeClustering()
        agglomerative_clustering.reset_state()
        self.assertEqual(agglomerative_clustering.clusters, [])
        self.assertEqual(agglomerative_clustering.centroids, None)
        self.assertEqual(agglomerative_clustering.similarity_matrix, None)

    def test_create_clusters(self):
        points = np.array([[1, 2, 3, 1], [4, 5, 6, 2], [7, 8, 9, 3]])
        segment_ids = np.array([1, 1, 2])
        agglomerative_clustering = AgglomerativeClustering()
        agglomerative_clustering.points = points
        agglomerative_clustering.segment_ids = segment_ids
        agglomerative_clustering.create_clusters()
        self.assertEqual(len(agglomerative_clustering.clusters), 2)

    def test_perform_clustering(self):
        points = np.array([[1, 2, 3, 1], [4, 5, 6, 2], [7, 8, 9, 3]])
        segment_ids = np.array([1, 1, 2])
        agglomerative_clustering = AgglomerativeClustering()
        clustered_points, cluster_ids = agglomerative_clustering.perform_clustering(points, segment_ids, 5)
        self.assertEqual(len(clustered_points), len(points))
        self.assertEqual(len(cluster_ids), len(segment_ids))

class TestDBSCANClustering(unittest.TestCase):
    def test_init(self):
        dbscan = DBSCANClustering(0.5, 5)
        self.assertEqual(dbscan.epsilon, 0.5)
        self.assertEqual(dbscan.min_samples, 5)
        self.assertEqual(dbscan.classifications, None)

    def test_reset_parameters(self):
        dbscan = DBSCANClustering(0.5, 5)
        dbscan.reset_parameters(0.3, 3)
        self.assertEqual(dbscan.epsilon, 0.3)
        self.assertEqual(dbscan.min_samples, 3)
        self.assertEqual(dbscan.classifications, None)

    def test_dbscan(self):
        pointcloud = np.array([[1, 2, 3, 1], [4, 5, 6, 2], [7, 8, 9, 3], [10, 11, 12, 4], [13, 14, 15, 5]])
        dbscan = DBSCANClustering(10, 2)
        clustered_points, noise_unclassified_points = dbscan.dbscan(pointcloud)
        self.assertEqual(len(clustered_points) + len(noise_unclassified_points), len(pointcloud))

    def test_dbscan_points_to_cluster(self):
        clustered_points = np.array([[1, 2, 3, 1], [4, 5, 6, 2]])
        unclustered_points = np.array([[7, 8, 9, 3], [10, 11, 12, 4], [13, 14, 15, 5]])
        dbscan = DBSCANClustering(10, 2)
        all_points = dbscan.dbscan_points_to_cluster(clustered_points, unclustered_points)
        self.assertEqual(len(all_points), len(clustered_points) + len(unclustered_points))

class TestConcentricZoneModel(unittest.TestCase):
    def setUp(self):
        self.model = ConcentricZoneModel()
        self.pointcloud = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_pointcloud_augmentation(self):
        augmented_pointcloud = self.model.pointcloud_augmentation(self.pointcloud)
        self.assertEqual(augmented_pointcloud.shape, (3, 6))

    def test_czm_bins(self):
        self.model.czm_bins(self.pointcloud)
        self.assertEqual(len(self.model.czm), self.model.Nz)

    def test_init_ground_czm_bins(self):
        self.model.czm_bins(self.pointcloud)
        self.model.init_ground_czm_bins()
        for ring in self.model.czm:
            for bin in ring:
                if bin.points.size != 0:
                    self.assertIsNotNone(bin.ground_points)
                    self.assertIsNotNone(bin.non_ground_points)


    def test_compute_covariance_matrix(self):
        covariance_matrix = self.model.compute_covariance_matrix(self.pointcloud)
        self.assertEqual(covariance_matrix.shape, (3, 3))

    def test_normal_vector(self):
        normal_vector = self.model.normal_vector(self.pointcloud)
        self.assertEqual(normal_vector.shape, (3,))

    def test_perform_ground_detection(self):
        ground, non_ground = self.model.perform_ground_detection(self.pointcloud)
        self.assertIsInstance(ground, np.ndarray)
        self.assertIsInstance(non_ground, np.ndarray)

    def setUp(self):
        self.model = ConcentricZoneModel()
        self.pointcloud = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_compute_and_reset(self):
        self.model.czm_bins(self.pointcloud)
        bin = self.model.czm[0][0]
        self.model.compute_and_reset(bin)
        self.assertIsNone(bin.ground_points)
        self.assertIsNone(bin.non_ground_points)

    def test_compute_and_reset(self):
        self.model.czm_bins(self.pointcloud)
        bin = self.model.czm[0][0]
        if bin.points.size != 0:
            self.model.compute_and_reset(bin)
            self.assertIsNone(bin.ground_points)
            self.assertIsNone(bin.non_ground_points)

    def test_uprightness(self):
        self.model.czm_bins(self.pointcloud)
        self.model.init_ground_czm_bins()
        self.model.ground_czm_bins()
        self.model.uprightness()
        for ring in self.model.czm:
            for bin in ring:
                if bin.ground_points is not None and bin.ground_points.shape[0] >= 3:
                    self.assertTrue(bin.uprightness)

    def test_get_points(self):
        self.model.czm_bins(self.pointcloud)
        self.model.init_ground_czm_bins()
        self.model.ground_czm_bins()
        self.model.uprightness()
        ground, non_ground = self.model.get_points()
        self.assertIsInstance(ground, np.ndarray)
        self.assertIsInstance(non_ground, np.ndarray)

if __name__ == '__main__':
    unittest.main()