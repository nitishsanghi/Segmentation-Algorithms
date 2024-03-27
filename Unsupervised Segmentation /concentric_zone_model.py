"""
Author: Nitish Sanghi
Date: 2024-03-27

Description:
This code implements a concentric zone model for point cloud processing, 
particularly focusing on ground detection. It includes classes such as `Bin` 
and `ConcentricZoneModel`. `Bin` represents a bin in the point cloud with attributes 
for points, normal vectors, centroid, and ground/non-ground classification. `ConcentricZoneModel` 
represents the model itself with methods for point cloud augmentation, bin creation, ground detection, 
uprightness determination, and retrieval of ground and non-ground points. The ground detection process 
involves computing the normal vector and distance coefficient of the ground plane equation for each bin, 
classifying points as ground or non-ground based on their distance from the plane, and determining the 
uprightness of bins. The code also provides a method for performing ground detection on a given 
point cloud for a specified number of iterations.

Dependencies:
- numpy
- enum
- helper module (containing Index enumeration)
"""


import numpy as np
from enum import IntEnum

from helper import Index

class Bin:
    """
    Represents a bin in the point cloud.

    Attributes:
        points: The points contained in the bin.
        normal: Normal vector of the ground plane fitted to the bin points.
        centroid: Centroid of the bin points.
        d: Distance coefficient of the ground plane equation.
        ground_points: Points classified as ground points.
        non_ground_points: Points classified as non-ground points.
        uprightness: Flag indicating if the bin is upright.
    """
    def __init__(self, points):
        self.points = points  
        self.normal = None  # The normal vector of the bin, to be calculated later
        # The centroid of the bin, calculated as the mean of the points if there are any, else None
        self.centroid = np.mean(points[:, :3], axis=0) if points.size > 0 else None
        self.d = None  # The perpendicular distance from the origin to the bin, to be calculated later
        self.ground_points = None  
        self.non_ground_points = None 
        self.uprightness = True  # A flag indicating whether the bin is upright, default is True
        self.vertical_points = None  # The points classified as being on vertical plane, to be determined later

class ConcentricZoneModel:
    """
    Represents the concentric zone model for point cloud processing.

    Attributes:
        pi: Value of pi.
        Nz: Number of concentric zones.
        L_max: Maximum value of zone size.
        L_min: Minimum value of zone size.
        upright_threshold: Threshold for determining uprightness.
        z_normal: Normal vector in the Z direction.
        z_margin: Margin for ground plane threshold.
        d_coeff_threshold: Threshold for distance coefficient.
        Nrt: Number of rings and theta divisions for each zone.
        L: Array containing minimum and maximum values of zone sizes.
        czm: List of concentric zone model bins.
    """
    def __init__(self):
        self.pi = np.pi 
        self.Nz = 4  # Number of zones
        self.L_max = 63.0  # Maximum length
        self.L_min = 0.5  # Minimum length
        
        self.upright_threshold = np.cos(self.pi/4)
        self.z_normal = np.array([[0],[0],[1]])  # Normal vector in z direction
        self.z_margin = 0.3  # Margin for z direction
        self.d_coeff_threshold = 0.1  # Threshold for d coefficient
        
        self.Nrt = np.array([[4, 32], [8, 64], [8, 108], [8, 64]])  # Array of radial and tangential divisions for each zone inner to outer
        # Array of minimum radial values for each zone
        self.L_min_values = np.array([self.L_min, (7 * self.L_min + self.L_max) / 8, (3 * self.L_min + self.L_max) / 4, (self.L_min + self.L_max) / 2])
        # Array of maximum radial values for each zone
        self.L_max_values = np.array([(7 * self.L_min + self.L_max) / 8, (3 * self.L_min + self.L_max) / 4, (self.L_min + self.L_max) / 2, self.L_max])

        self.L = np.column_stack((self.L_min_values, self.L_max_values))
        self.czm = []  # List to store the concentric zone model bins
        

    def pointcloud_augmentation(self, pointcloud):
        """
        Augments point cloud with radial distance and azimuth angle.

        Args:
            pointcloud: Input point cloud.

        Returns:
            Augmented point cloud.
        """
        radial = np.sqrt(pointcloud[:, Index.X] ** 2 + pointcloud[:, Index.Y] ** 2)
        azimuth = np.arctan2(pointcloud[:, Index.Y], pointcloud[:, Index.X])
        row_ids = np.arange(pointcloud.shape[0])
        return np.column_stack((pointcloud, row_ids, radial, azimuth))

    def czm_bins(self, pointcloud):
        """
        Creates bins for the concentric zone model.

        Args:
            pointcloud: Input point cloud.
        """
        self.pointcloud = self.pointcloud_augmentation(pointcloud)
        for k in range(self.Nz):
            ring = []
            dL = self.L[k, 1] - self.L[k, 0]
            Nr, Nt = self.Nrt[k]
            for i in range(Nr):
                # Defining radial condition
                condition_1 = (i * dL / Nr <= self.pointcloud[:, Index.R] - self.L[k, 0]) & (self.pointcloud[:, Index.R] - self.L[k, 0] < (i + 1) * dL / Nr)
                for j in range(Nt):
                    # Defining azimuth condition
                    condition_2 =  (j * 2 * self.pi / Nt - self.pi <= self.pointcloud[:, Index.A]) & (self.pointcloud[:, Index.A] < (j + 1) * 2 * self.pi / Nt - self.pi) & condition_1
                    points = self.pointcloud[condition_2, :Index.R]
                    ring.append(Bin(points))
            self.czm.append(ring)

    def init_ground_czm_bins(self):
        """
        Initializes ground and non-ground points for bins.
        """
        for ring in self.czm:
            for bin in ring:
                if bin.points.size != 0:
                    # Calculate threshold for ground points
                    z_threshold = np.min(bin.points[:, Index.Z]) + self.z_margin
                    
                    # Assign points as ground or non-ground based on threshold
                    bin.ground_points = bin.points[bin.points[:, Index.Z] < z_threshold]
                    bin.non_ground_points = bin.points[bin.points[:, Index.Z] >= z_threshold]

    def compute_covariance_matrix(self, cloud_points): 
        """
        Computes the covariance matrix of cloud points.

        Args:
            cloud_points: Points for computing covariance matrix.

        Returns:
            Covariance matrix.
        """
        return np.cov(cloud_points[:,:3], rowvar=False)

    def normal_vector(self, cloud_points):
        """
        Computes the normal vector of the ground plane.

        Args:
            cloud_points: Points used for normal vector computation.

        Returns:
            Normal vector.
        """
        # Compute the covariance matrix of the cloud points
        covariance = self.compute_covariance_matrix(cloud_points)
        
        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigen_values, eigen_vectors = np.linalg.eigh(covariance)
        
        # Return the eigenvector corresponding to the smallest eigenvalue
        return eigen_vectors[:, np.argmin(eigen_values)]

    def plane_coeff_d(self, normal_vector, p):
        """
        Computes the distance coefficient of the ground plane equation.

        Args:
            normal_vector: Normal vector of the ground plane.
            p: Point on the ground plane.

        Returns:
            Distance coefficient.
        """
        return -np.dot(normal_vector, p[:-1])

    def compute_and_reset(self, bin):
        """
        Computes the normal vector and distance coefficient of the ground plane and resets the bin points.

        Args:
            bin: Bin for computation and reset.
        """
        covariance = self.compute_covariance_matrix(bin.ground_points)

        bin.normal = self.normal_vector(covariance)
        
        mean = np.mean(bin.ground_points, axis=0)

        bin.d = self.plane_coeff_d(bin.normal, mean)
        
        # Clear the ground points and non-ground points in the bin to save memory
        bin.ground_points = None
        bin.non_ground_points = None

    def assign_points(self, bin, ground_points, non_ground_points):
        """
        Assigns ground and non-ground points to a bin.

        Args:
            bin: Bin to assign points.
            ground_points: Ground points to assign.
            non_ground_points: Non-ground points to assign.
        """
        bin.ground_points = np.array(ground_points) if ground_points is not None else None
        bin.non_ground_points = np.array(non_ground_points) if non_ground_points is not None else None
   
    def iterate_bins(self, func):
        """
        Iterates over all bins in all rings and applies a function.

        Args:
            func: Function to apply to each bin.
        """
        for ring in self.czm:
            for bin in ring:
                func(bin)

    def ground_czm_bins(self):
        """
        Classifies ground and non-ground points in each bin based on the ground plane equation.
        """
        def classify_points(bin):
            # If bin or its ground points are None, return
            if bin is None or bin.ground_points is None:
                return

            # If there are less than 3 ground points, assign all points as non-ground
            if bin.ground_points.shape[0] < 3:
                self.assign_points(bin, None, bin.points)
                return

            # Compute normal and reset bin
            self.compute_and_reset(bin)

            ground = []  # List to store ground points
            non_ground = []  # List to store non-ground points

            # Classify points as ground or non-ground based on distance from plane
            for p in bin.points:
                d_k = self.plane_coeff_d(bin.normal, p)
                if abs(bin.d - d_k) < self.d_coeff_threshold:
                    ground.append(p)
                else:
                    non_ground.append(p)

            # If there are ground points, assign them to bin
            if ground:
                bin.ground_points = np.array(ground)
            # If there are no ground points, assign all points as non-ground
            else:
                self.assign_points(bin, None, bin.points)

            # If there are non-ground points, assign them to bin
            if non_ground:
                bin.non_ground_points = np.array(non_ground)

        # Iterate over all bins and classify points
        self.iterate_bins(classify_points)

    def upright_norm_dot_product(self, bin):
        """
        Computes the dot product of the normal vector and Z normal vector to determine uprightness.

        Args:
            bin: Bin for computing uprightness.

        Returns:
            Dot product value.
        """
        return (np.transpose(bin.normal).dot(self.z_normal)/(np.linalg.norm(bin.normal)*np.linalg.norm(self.z_normal)))

    def uprightness(self):
        """
        Determines the uprightness of each bin based on the dot product threshold.
        """
        def check_uprightness(bin):
            if bin is None or bin.ground_points is None or bin.ground_points.shape[0] < 3:
                return

            bin.uprightness = (self.upright_norm_dot_product(bin) > self.upright_threshold)

            if not bin.uprightness:
                self.assign_points(bin, None, bin.points)

        self.iterate_bins(check_uprightness)
    
    def get_points(self):
        """
        Retrieves all ground and non-ground points from the bins.

        Returns:
            Ground and non-ground points concatenated from all bins.
        """
        # Gather all ground points from each bin in the concentric zone model having at least 3 ground points, and are upright
        ground = [bin.ground_points for ring in self.czm for bin in ring if bin is not None and bin.ground_points is not None and bin.ground_points.shape[0] >= 3 and bin.uprightness]
        
        # Gather all non-ground points from each bin in the concentric zone model
        non_ground = [bin.non_ground_points for ring in self.czm for bin in ring if bin is not None and bin.non_ground_points is not None]

        ground = np.concatenate(ground, axis=0) if ground else np.array([])
        non_ground = np.concatenate(non_ground, axis=0) if non_ground else np.array([])

        # If the total number of points in the pointcloud is not equal to the sum of the number
        # of ground and non-ground points then adding those points to non-ground points
        if self.pointcloud.shape[0] != ground.shape[0] + non_ground.shape[0]:
            pointcloud_ids = self.pointcloud[:, Index.ID]
            ground_non_ground_ids = np.concatenate((ground[:, Index.ID], non_ground[:, Index.ID]), axis=0) if ground.size and non_ground.size else np.array([])
            values = np.setdiff1d(pointcloud_ids, ground_non_ground_ids)
            non_ground = np.concatenate((non_ground, self.pointcloud[np.array(values).astype(int),:Index.R]), axis=0) if values.size else non_ground

        return ground, non_ground

    def perform_ground_detection(self, pointcloud, iterations=3):
        """
        Perform ground detection on the given pointcloud.

        Args:
            pointcloud: The pointcloud to perform ground detection on.
            iterations: The number of iterations to perform ground detection. Default is 3.

        Returns:
            The points after performing ground detection.
        """
        # Create bins for the concentric zone model
        self.czm_bins(pointcloud)

        # Initialize the ground bins in the concentric zone model
        self.init_ground_czm_bins()

        # Perform ground detection for the specified number of iterations
        for _ in range(iterations):
            self.ground_czm_bins()

        # Calculate the uprightness of the pointcloud
        self.uprightness()

        # Return the points after performing ground detection
        return self.get_points()