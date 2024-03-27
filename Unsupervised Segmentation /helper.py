"""
Author: Nitish Sanghi
Date: 2024-03-27

Description:
This code provides functionalities for clustering and evaluating clusters in a point cloud dataset. 
It includes functions for clustering points based on specified criteria, separating clusters for 
reprocessing, and evaluating the quality of clustering using various metrics such as Calinski-Harabasz score, 
Davies-Bouldin score, and Silhouette score. Additionally, it implements the Local Distance-based Outlier Factor (LDoF) 
algorithm to compute outlier scores for each point in the point cloud based on local distances.

Dependencies:
- numpy
- scipy
- enum
- collections
- matplotlib
- sklearn
"""

import numpy as np
from scipy import spatial
from enum import IntEnum
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples 

class Index(IntEnum):
    X = 0  # Index for X coordinate
    Y = 1  # Index for Y coordinate
    Z = 2  # Index for Z coordinate
    ID = 3  # Index for point unique identifier
    R = 4  # Index for radial distance
    A = 5  # Index for azimuth angle

def cluster_id_thresholder(segment_ids, min_samples = 20):
    """
    Thresholds cluster IDs based on the minimum number of samples.

    Parameters:
    - segment_ids (numpy.ndarray): Array containing segment IDs.
    - min_samples (int): Minimum number of samples for a cluster to be considered valid.

    Returns:
    - id_to_cluster (numpy.ndarray): Array of cluster IDs that do not meet the minimum sample threshold.
    """
    id_count = np.array(list(Counter(segment_ids).items())).astype(int)
    id_count_sorted = id_count[np.argsort(id_count[:, 1])]
    id_to_cluster = id_count_sorted[id_count_sorted[:, 1] < min_samples][:, 0]
    return id_to_cluster

def concatenate_cluster_with_ids(clustered_points, segment_ids):
    """
    Concatenates cluster IDs with corresponding clustered points.

    Parameters:
    - clustered_points (numpy.ndarray): Array containing clustered points.
    - segment_ids (numpy.ndarray): Array containing segment IDs for clustered points.

    Returns:
    - concatenated_array (numpy.ndarray): Concatenated array of clustered points with segment IDs.
    """
    return np.concatenate((clustered_points, segment_ids[:,np.newaxis]), axis=1)

def separate_cluster_recluster(pointscloud, segment_ids, recluster_point_ids):
    """
    Separates clusters based on specified cluster IDs for reprocessing.

    Parameters:
    - pointscloud (numpy.ndarray): Array containing points of the point cloud.
    - segment_ids (numpy.ndarray): Array containing segment IDs for points.
    - recluster_point_ids (numpy.ndarray): Array containing cluster IDs to be reclustered.

    Returns:
    - reclustered_clusters (numpy.ndarray): Array containing reclustered clusters.
    - non_reclustered_clusters (numpy.ndarray): Array containing non-reclustered clusters.
    """
    indices = np.isin(segment_ids, list(recluster_point_ids))
    clusters, non_clusters = pointscloud[~indices], pointscloud[indices]
    cluster_ids, non_cluster_ids = segment_ids[~indices], segment_ids[indices]
    return concatenate_cluster_with_ids(clusters, cluster_ids), concatenate_cluster_with_ids(non_clusters, non_cluster_ids)
    
def generate_clusters_for_reprocessing(pointcloud, segment_ids, min_samples = 20):
    """
    Generates clusters for reprocessing based on specified minimum samples.

    Parameters:
    - pointcloud (numpy.ndarray): Array containing points of the point cloud.
    - segment_ids (numpy.ndarray): Array containing segment IDs for points.
    - min_samples (int): Minimum number of samples for a cluster to be considered valid.

    Returns:
    - reclustered_clusters (numpy.ndarray): Array containing reclustered clusters.
    - non_reclustered_clusters (numpy.ndarray): Array containing non-reclustered clusters.
    """
    recluster_point_ids = cluster_id_thresholder(segment_ids, min_samples)
    return separate_cluster_recluster(pointcloud, segment_ids, recluster_point_ids)

def evaluation(pointcloud, segment_ids):
    """
    Evaluates clustering quality using various metrics and visualizes Silhouette scores.

    Parameters:
    - pointcloud (numpy.ndarray): Array containing points of the point cloud.
    - segment_ids (numpy.ndarray): Array containing segment IDs for points.
    """
    vrc_score = calinski_harabasz_score(pointcloud, segment_ids)
    print(vrc_score)
    
    dbi_score = davies_bouldin_score(pointcloud, segment_ids)
    print(dbi_score)
    
    score = silhouette_score(pointcloud, segment_ids, metric='euclidean')
    print(score)
    
    score = silhouette_samples(pointcloud, segment_ids, metric='euclidean')
    plt.hist(score, bins=20, alpha=0.5, color='b', label='Silhouette Score')
    plt.title('Samples Silhouette Score Histogram')
    plt.xlabel('Silhouette Score')
    plt.ylabel('Frequency')
    plt.show()

def local_distance_based_outlier_factor(point_cloud, k):
    """
    Compute the Local Distance-based Outlier Factor (LDoF) for each point in the point cloud.

    Parameters:
    - point_cloud (numpy.ndarray): The point cloud data with shape (n_points, n_dimensions).
    - k (int): The number of neighbors to consider for computing LDoF.

    Returns:
    - ldof_scores (numpy.ndarray): The LDoF scores for each point in the point cloud.
    """

    # Initialize a nearest neighbors model
    nn_model = NearestNeighbors(n_neighbors=k+1)  # k+1 to include the point itself
    nn_model.fit(point_cloud)

    # Find k nearest neighbors for each point
    distances, indices = nn_model.kneighbors(point_cloud)

    # Compute LDoF for each point
    ldof_scores = np.zeros(point_cloud.shape[0])
    for i in range(len(point_cloud)):
        neighbors = point_cloud[indices[i, 1:]]  # Exclude the point itself
        center = point_cloud[i]
        mean_distance = np.mean(np.linalg.norm(neighbors - center, axis=1))
        local_distance = np.linalg.norm(center - np.mean(neighbors, axis=0))
        ldof_scores[i] = local_distance / mean_distance

    return ldof_scores
