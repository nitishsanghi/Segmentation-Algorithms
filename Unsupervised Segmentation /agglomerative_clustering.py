"""
Author: Nitish Sanghi
Date: 2024-03-27

Description:
This code implements agglomerative clustering for point clouds, allowing the 
grouping of points into clusters based on their proximity to each other. 
It includes classes such as `Cluster` and `AgglomerativeClustering`. `Cluster` 
represents a cluster of points with attributes for points, centroid, and cluster 
identifier. `AgglomerativeClustering` represents the clustering algorithm with 
methods for initializing clusters, calculating centroids, computing similarity 
between clusters, merging clusters, and performing the clustering process. 
The clustering process involves iteratively merging the two closest clusters 
until the distance between them exceeds a specified threshold. The code also 
provides methods for plotting the similarity matrix between clusters and visualizing 
the centroids of the resulting clusters in 3D space.

Dependencies:
- numpy
- matplotlib.pyplot
- scipy.spatial.distance
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform

class Cluster:
    def __init__(self, points, cluster_id):
        """
        Initialize a Cluster object.

        Args:
        - points (numpy.ndarray): Array of points belonging to the cluster.
        - cluster_id (int): Identifier for the cluster.
        """
        self.points = points
        self.centroid = self.calculate_centroid()  # Calculate centroid of the cluster
        self.cluster_id = cluster_id  # Assign cluster identifier

    def calculate_centroid(self):
        """
        Calculate the centroid of the cluster.

        Returns:
        - numpy.ndarray: Centroid coordinates.
        """
        return np.mean(self.points[:,:-1], axis=0) if self.points.shape[0] > 1 else self.points[0,:-1]

class AgglomerativeClustering:
    def __init__(self):
        """
        Initialize an AgglomerativeClustering object.

        The object is initialized without any points, segment identifiers, or distance threshold.
        These values should be set later using appropriate methods.
        """
        # The distance threshold for merging clusters
        self.distance_threshold = None
        
        # The segment identifiers for each point
        self.segment_ids = None
        
        # The data points to be clustered
        self.points = None
        
        # A list to store the clusters
        self.clusters = []
        
        # The centroids of the clusters, to be calculated later
        self.centroids = None
        
        # The similarity matrix between clusters, to be calculated later
        self.similarity_matrix = None

    def reset_state(self):
        """
        Reset the state of the AgglomerativeClustering object.

        This method clears the list of clusters and sets the centroids and similarity matrix to None.
        It is useful when you want to reuse the same AgglomerativeClustering object for multiple clustering tasks.
        """
        self.clusters = []
        self.centroids = None
        self.similarity_matrix = None

    def create_clusters(self):
        """
        Create initial clusters based on segment identifiers.
        """
        # Get unique segment identifiers
        unique_segment_ids = np.unique(self.segment_ids)
        

        for i in unique_segment_ids:
            # Get points that belong to the current segment
            segment_points = self.points[self.segment_ids == i]
            
            if len(segment_points) > 0:
                # If the segment identifier is less than 0 i.e. noise or unclassified, create a new cluster for each point
                if i < 0:
                    ids = np.arange(self.segment_ids.max() + 1, self.segment_ids.max() + 1 + len(segment_points))
                    self.clusters.extend(Cluster(point[np.newaxis, :], id) for id, point in zip(ids, segment_points))
                
                # If the segment identifier is not less than 0, create a single cluster for all points
                else:
                    self.clusters.append(Cluster(segment_points, i))


    def collect_centroids(self):
        """
        Calculate centroids for all clusters and compute similarity matrix.
        """
        self.centroids = np.array([cluster.calculate_centroid() for cluster in self.clusters])
        self.compute_similarity_matrix()
        

    def compute_similarity_matrix(self):
        """
        Compute similarity matrix between clusters.
        """
        # Compute the pairwise distances between the centroids using squared Euclidean distance and construct the similarity matrix
        distances = pdist(self.centroids, metric='sqeuclidean')
        self.similarity_matrix = squareform(distances)
        
        # Fill the diagonal of the similarity matrix with infinity (since the distance of a point to itself is not meaningful in this context)
        np.fill_diagonal(self.similarity_matrix, np.inf)

    def closest_clusters(self):
        """
        Find indices of the closest clusters based on similarity matrix.
        """
        return np.unravel_index(np.argmin(self.similarity_matrix), self.similarity_matrix.shape)

    def merge_clusters(self, closest_clusters):
        """
        Merge the two closest clusters based on similarity.
        """
        # Get the two closest clusters based on the similarity matrix
        cluster_1, cluster_2 = self.clusters[closest_clusters[0]], self.clusters[closest_clusters[1]]
        
        # Merge the points of the two clusters into cluster_1
        cluster_1.points = np.concatenate((cluster_1.points, cluster_2.points), axis=0)
        
        # Remove cluster_2 from the list of clusters
        self.clusters.remove(cluster_2)

    def generate_pointcloud_segment_ids(self):
        """
        Generate point cloud and segment IDs based on merged clusters.
        """
        points = [cluster.points for cluster in self.clusters]
        cluster_ids = [np.ones(cluster.points.shape[0]) * cluster.cluster_id for cluster in self.clusters]

        return np.concatenate(points, axis=0), np.concatenate(cluster_ids, axis=0)
    
    def update_cluster_ids(self):
        """
        Update cluster IDs after merging clusters to be consecutive integers.
        """
        for i, cluster in enumerate(self.clusters):
            cluster.cluster_id = i
            cluster.points[:,-1] = i

    def perform_clustering(self, points, segment_ids, distance_threshold):
        """
        Perform agglomerative clustering on the given point cloud.

        Args:
        - points (numpy.ndarray): Array of data points.
        - segment_ids (numpy.ndarray): Array of segment identifiers for each point.
        - distance_threshold (float): Threshold for merging clusters based on distance.

        This method first resets the state of the AgglomerativeClustering object, then sets the distance threshold,
        the points, and the segment identifiers. It then performs the clustering process.
        """
        # Reset the state of the AgglomerativeClustering object
        self.reset_state()
        # Set the distance threshold for merging clusters
        self.distance_threshold = distance_threshold
        # Set the data points to be clustered
        self.points = points
        # Set the segment identifiers for each point
        self.segment_ids = segment_ids

        # Create initial clusters based on segment identifiers
        self.create_clusters()
        
        # Start an infinite loop for the clustering process
        while True:
            # Calculate centroids for all clusters and compute similarity matrix
            self.collect_centroids()
            
            # Compute similarity matrix between clusters
            self.compute_similarity_matrix()
            
            # If the minimum value in the similarity matrix is greater than the distance threshold, break the loop
            if np.min(self.similarity_matrix) > self.distance_threshold:
                break
            
            # Merge the two closest clusters based on similarity
            self.merge_clusters(self.closest_clusters())
        
        # Generate point cloud and segment IDs based on merged clusters and return them
        return self.generate_pointcloud_segment_ids()

    def plot_similarity_matrix(self):
        """
        Plot the similarity matrix between clusters.
        """
        plt.imshow(self.similarity_matrix/20, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Similarity Matrix')
        plt.xlabel('Cluster Index')
        plt.ylabel('Cluster Index')
        plt.show()

    def plot_cluster_centroids(self):
        """
        Plot the centroids of all clusters in 3D space.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        self.update_cluster_ids()
        for cluster in self.clusters:
            centroid = cluster.calculate_centroid()
            ax.scatter(centroid[0], centroid[1], centroid[2], s=1.5)
            ax.text(centroid[0], centroid[1], centroid[2],str(cluster.cluster_id),fontsize=6)
        ax.set_xlabel("x (m)", fontsize=12)
        ax.set_ylabel("y (m)", fontsize=12)
        ax.set_zlabel("z (m)", fontsize=12)
        ax.set_title("Cluster Centroids (3D)", fontsize=14)
        plt.show()