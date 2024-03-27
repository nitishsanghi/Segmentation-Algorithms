"""
Author: Nitish Sanghi
Date: 2024-03-27

Description:
This code implements the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 
algorithm for clustering 3D point clouds. It defines a class DBSCANClustering with methods to 
initialize the parameters, perform clustering, grow clusters, and convert classifications. 
The algorithm iteratively grows clusters based on the density of points within a specified 
epsilon neighborhood. Points are classified as noise, unclassified, or assigned to a particular 
cluster. The implementation allows for resetting parameters, finding nearest neighbors, and converting 
classifications. Additionally, it provides methods to perform DBSCAN clustering on the given 
point cloud data and to map unclustered points to existing clusters.

Dependencies:
- numpy
- enum
- scipy
"""

import numpy as np
from enum import IntEnum, Enum
from scipy.spatial import KDTree

# Define indices for different attributes of a point
class Index(IntEnum):
    X = 0
    Y = 1
    Z = 2
    R = 3
    A = 4
    ID = 5

# Define categories for point classification
class Category(Enum):
    NOISE = -2
    UNCLASSIFIED = -1
    CLASSIFIED = 1


class DBSCANClustering:
    def __init__(self, epsilon, min_samples):
        """
        Initialize DBSCANClustering with epsilon and min_samples parameters.
        """
        if epsilon <= 0 or min_samples < 1:
            raise ValueError("epsilon must be positive and min_samples must be at least 1")
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.classifications = None  # Initialize classifications as None

    def reset_parameters(self, epsilon, min_samples):
        """
        Update epsilon and min_samples parameters.
        """
        if epsilon <= 0 or min_samples < 1:
            raise ValueError("epsilon must be positive and min_samples must be at least 1")
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.classifications = None  # Reset classifications

    def kdtree(self, pointcloud):
        """
        Construct a KDTree from the point cloud.
        """
        return KDTree(pointcloud, copy_data=False)  # Create a KDTree

    def find_nearest_neighbors(self, pointcloud, index, kd_tree):
        """
        Find indices of points within epsilon distance from the given point.
        """
        # Calculate density of points within epsilon distance
        density = kd_tree.query_ball_point(pointcloud[index, :-1], self.epsilon)
        density = np.round(4/3*np.pi*self.epsilon**3/len(density)*1000,2)

        # Adjust epsilon based on density
        epsilon = self.epsilon + self.epsilon/2*1/density

        # Return indices of points within adjusted epsilon distance
        return kd_tree.query_ball_point(pointcloud[index, :-1], epsilon)

    def grow_cluster(self, pointcloud, kd_tree, point_id, cluster_id):
        """
        Grow a cluster starting from the given point_id.
        """
        # Find nearest neighbors of the point
        neighbors = self.find_nearest_neighbors(pointcloud, point_id, kd_tree)
        if len(neighbors) < self.min_samples:  # If not enough neighbors, return False
            return False
   
        # Assign cluster ID to the point and its neighbors
        self.classifications[point_id] = cluster_id
        for neighbor_id in neighbors:
            self.classifications[neighbor_id] = cluster_id

        # Explore neighbors of neighbors
        while len(neighbors) > 0:
            current_point = neighbors[0]
            results = self.find_nearest_neighbors(pointcloud, current_point, kd_tree)
            if len(results) >= self.min_samples:  # If enough neighbors, add to cluster
                for result_point in results:
                    if self.classifications[result_point] in (Category.UNCLASSIFIED, Category.NOISE):
                        if self.classifications[result_point] == Category.UNCLASSIFIED:
                            neighbors.append(result_point)
                        self.classifications[result_point] = cluster_id
            neighbors = neighbors[1:]  # Remove the current point from neighbors list
        return True

    def convert_classification(self, s):
        # Convert classification from Category to integer for easier processing
        if s == Category.NOISE:
            return -2
        elif s == Category.UNCLASSIFIED:
            return -1
        else:
            return s

    def dbscan(self, pointcloud):
        """
        Perform DBSCAN clustering on the given point cloud.
        """
        cluster_id = 1  # Initialize cluster ID
        n_points = pointcloud.shape[0]  # Get number of points
        self.classifications = [Category.UNCLASSIFIED] * n_points  # Initialize all points as unclassified
        kd_tree = self.kdtree(pointcloud[:,:3])  # Create a KDTree
        for point_id in range(n_points):  # For each point, try to grow a cluster
            if self.classifications[point_id] == Category.UNCLASSIFIED and self.grow_cluster(pointcloud, kd_tree, point_id, cluster_id):
                cluster_id += 1  # If a cluster is grown, increment cluster ID

        # Convert noise points to -2, rest to their respective cluster IDs
        self.classifications = np.array([self.convert_classification(s) for s in self.classifications])
        cluster_ids = self.classifications[self.classifications > 0]
        noise_unclassified_ids = self.classifications[self.classifications < 0]
        cluster_points = pointcloud[self.classifications > 0]
        noise_unclassified_points = pointcloud[self.classifications < 0]
        return np.concatenate((cluster_points, cluster_ids[:,np.newaxis]), axis=1), np.concatenate((noise_unclassified_points, noise_unclassified_ids[:,np.newaxis]),axis=1)

    def is_within_epsilon(self, kd_tree, point):
        # Check if the second nearest neighbor is within epsilon distance
        knn = kd_tree.query(point, k=2)
        return knn[0][1] < self.epsilon

    def dbscan_points_to_cluster(self, clustered_points, unclustered_points):
        # Get the maximum cluster ID
        max_cluster_id = np.max(clustered_points[:,-1])
        for id in range(0, int(max_cluster_id)+1):
            # Get points in the current cluster
            cluster_points = clustered_points[clustered_points[:,-1] == id]
            kd_tree = self.kdtree(cluster_points[:,:-2])  # Create a KDTree for the cluster
            # Find unclustered points within epsilon distance to the cluster
            points_to_delete_add = [point_id for point_id in range(unclustered_points.shape[0]) if self.is_within_epsilon(kd_tree, unclustered_points[point_id,:-2])]
            unclustered_points[points_to_delete_add,-1] = id  # Assign cluster ID to these points
            clustered_points = np.vstack((clustered_points, unclustered_points[points_to_delete_add]))  # Add these points to the cluster
            unclustered_points = np.delete(unclustered_points, np.array(points_to_delete_add).astype(int), axis=0)  # Remove these points from unclustered_points
        return np.concatenate((clustered_points, unclustered_points), axis=0)  # Return all points