"""
Author: Nitish Sanghi
Date: 2024-03-27

Description:
This code segment implements a pipeline for segmenting a point cloud into different clusters. 
It utilizes various clustering techniques such as Concentric Zone Model, DBSCAN, and Agglomerative Clustering 
for segmenting the point cloud. Additionally, it provides visualization functions to display the segmented 
point cloud in both 3D and 2D views. 

Dependencies:
- numpy
- matplotlib
- enum
- mpl_toolkits
- scipy
- collections
- concentric_zone_model.py
- dbscan.py
- agglomerative_clustering.py
- helper.py
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum, Enum
from mpl_toolkits import mplot3d
from scipy.spatial import KDTree
from collections import Counter


#Custom imports
from concentric_zone_model import ConcentricZoneModel, Bin
from dbscan import DBSCANClustering, Category
from agglomerative_clustering import AgglomerativeClustering, Cluster
import helper
#%% types
class Index(IntEnum):
    X = 0  # Index for X coordinate
    Y = 1  # Index for Y coordinate
    Z = 2  # Index for Z coordinate
    ID = 3  # Index for point unique identifier
    R = 4  # Index for radial distance
    A = 5  # Index for azimuth angle

#%% helper functions
def visualize_pointcloud_downsampled(pc:np.ndarray, downsample_factor:int=10) -> None:
    fig = plt.figure(figsize=(50, 50))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pc[::downsample_factor, Index.X],
               pc[::downsample_factor, Index.Y],
               pc[::downsample_factor, Index.Z],
               color="red", s=0.1)
    ax.set_xlabel("x (m)", fontsize=14)
    ax.set_ylabel("y (m)", fontsize=14)
    ax.set_zlabel("z (m)", fontsize=14)
    ax.set_xlim(-17, 65)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-6, 6)
    ax.set_title("Pointcloud (3D)", fontsize=14)
    plt.show()

    # make this plot occupy 30% of the figure's width and 100% of its height
    plt.figure(figsize=(25, 25))
    plt.plot(pc[:, Index.X], pc[:, Index.Y], "rx", markersize=1, alpha=0.2)
    plt.xlabel("x (m)", fontsize=14)
    plt.ylabel("y (m)", fontsize=14)
    plt.grid(True)

    plt.gca().set_aspect('auto', adjustable='box')
    plt.title("Pointcloud (Top View)", fontsize=14)
    plt.show()

    # make this plot occupy 30% of the figure's width and 100% of its height
    plt.figure(figsize=(25, 25))
    plt.plot(pc[:, Index.X], pc[:, Index.Z], "rx", markersize=1, alpha=0.2)
    plt.xlabel("x (m)", fontsize=14)
    plt.ylabel("z (m)", fontsize=14)
    plt.grid(True)

    plt.gca().set_aspect('auto', adjustable='box')
    plt.title("Pointcloud (Front View)", fontsize=14)
    plt.show()

     # make this plot occupy 30% of the figure's width and 100% of its height
    plt.figure(figsize=(25, 25))
    plt.plot(pc[:, Index.Y], pc[:, Index.Z], "rx", markersize=1, alpha=0.2)
    plt.xlabel("y (m)", fontsize=14)
    plt.ylabel("z (m)", fontsize=14)
    plt.grid(True)

    plt.gca().set_aspect('auto', adjustable='box')
    plt.title("Pointcloud (Side View)", fontsize=14)
    plt.show()

def visualize_pointcloud_downsampled_with_segment_ids(pc: np.ndarray, segment_ids: np.ndarray,
                                                      downsample_factor:int=10) -> None:
    fig = plt.figure(figsize=(25, 25))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(pc[::downsample_factor, Index.X],
               pc[::downsample_factor, Index.Y],
               pc[::downsample_factor, Index.Z],
               c=segment_ids[::downsample_factor],
               cmap="prism",
               s=0.2,label=segment_ids[::downsample_factor])
    ax.set_xlabel("x (m)", fontsize=14)
    ax.set_ylabel("y (m)", fontsize=14)
    ax.set_zlabel("z (m)", fontsize=14)
    ax.set_xlim(-20, 60)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-5, 5)
    ax.set_title("Pointcloud (3D)", fontsize=14)
    
    plt.show()

    # make this plot occupy 30% of the figure's width and 100% of its height
    plt.figure(figsize=(25, 25))
    plt.scatter(pc[:, Index.X], pc[:, Index.Y], c=segment_ids, cmap="prism", s=1, alpha=0.5)
    plt.xlabel("x (m)", fontsize=14)
    plt.ylabel("y (m)", fontsize=14)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Pointcloud (Top View)", fontsize=14)
    plt.show()

#%%
pointcloud = np.load("data/cyngn_interview_question_pointcloud_data.npy")
#visualize_pointcloud_downsampled(pointcloud, downsample_factor=1) # use 'downsample_factor=1' for no downsampling during visualization

##### TODO: REQUIRES IMPLEMENTATION ##############################
##################################################################
def segment_pointcloud(pointcloud:np.ndarray) -> np.ndarray:
    """
    Segment a pointcloud into different clusters.

    Args:
    - pointcloud (numpy.ndarray): Input pointcloud of shape (N, 3)

    Returns:
    - numpy.ndarray: A segmentation mask of shape (N,), where each element is an integer representing the segment id
    """
    # Create a ConcentricZoneModel object
    czm = ConcentricZoneModel()
    # Perform ground detection on the pointcloud
    ground_points, non_ground_points = czm.perform_ground_detection(pointcloud)
    # Initialize segment ids for ground points
    segment_ids_ground = np.zeros(ground_points.shape[0])
    # Concatenate ground points with their segment ids
    ground_points = helper.concatenate_cluster_with_ids(ground_points, segment_ids_ground)

    # Create a DBSCANClustering object
    dbscan_clusterer = DBSCANClustering(.5, 10)
    # Perform DBSCAN clustering on non-ground points
    clusters, noise_unclassified = dbscan_clusterer.dbscan(non_ground_points)

    # Create an AgglomerativeClustering object
    agglomerative_clusterer = AgglomerativeClustering()
    # Perform agglomerative clustering on the clusters
    non_ground_points_cloud_clustered, segment_ids = agglomerative_clusterer.perform_clustering(clusters[:,:-1], clusters[:,-1].astype(int), 1.)


    # Generate clusters for reprocessing
    non_ground_clusters, points_to_recluster = helper.generate_clusters_for_reprocessing(non_ground_points_cloud_clustered, segment_ids, min_samples = 40)
    # Concatenate points to recluster with unclassified noise
    recluster_points_and_unclassified = np.concatenate((points_to_recluster, noise_unclassified), axis=0)
    # Concatenate non-ground clusters with ground points
    clustered_points = np.concatenate((non_ground_clusters, ground_points), axis=0)

    # Reset parameters of DBSCAN clusterer
    dbscan_clusterer.reset_parameters(2.0,10)
    # Perform DBSCAN clustering on the clustered points and points to recluster
    semi_final_cluster = dbscan_clusterer.dbscan_points_to_cluster(clustered_points, recluster_points_and_unclassified)

    # Generate clusters for reprocessing
    semi_final_cluster, points_to_recluster = helper.generate_clusters_for_reprocessing(semi_final_cluster[:,:-1], semi_final_cluster[:,-1], min_samples = 20)

    # Reset parameters of DBSCAN clusterer
    dbscan_clusterer.reset_parameters(3.0,10)
    # Perform DBSCAN clustering on the final cluster and points to recluster
    final_cluster = dbscan_clusterer.dbscan_points_to_cluster(semi_final_cluster, points_to_recluster)

    # Perform agglomerative clustering on the final cluster
    pointcloud, segment_ids = agglomerative_clusterer.perform_clustering(final_cluster [:,:-1], final_cluster[:,-1].astype(int), 2)
    #agglomerative_clusterer.plot_cluster_centroids() #Uncomment to see cluster centroids
    
    # Concatenate the pointcloud with the segment ids
    pointcloud = np.concatenate((pointcloud, segment_ids[:, np.newaxis]), axis=1)
    # Sort the pointcloud by the unique identifier
    pointcloud = pointcloud[np.argsort(pointcloud[:, Index.ID])]

    # Return the segment ids
    return pointcloud[:,-1]


if __name__ == "__main__":
    pointcloud = np.load("data/cyngn_interview_question_pointcloud_data.npy")
    segment_ids = segment_pointcloud(pointcloud) 
    visualize_pointcloud_downsampled_with_segment_ids(pointcloud, segment_ids, downsample_factor=1)

    #helper.evaluation(pointcloud, segment_ids) #Uncomment to see evaluation metrics takes a while to computer