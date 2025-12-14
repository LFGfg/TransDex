# The datasets of this paper are in the following format: 
# 2-10 is a noise sample, and 1 is a label sample. 
# 12-20 are noise samples, and 11 are label samples.  
# 22-30 are noise samples, and 21 are label samples.  
# 32-40 are noise samples, and 31 are label samples.
# It is recommended to assign different color attributes to the point clouds of the dexterous hand and objects for differentiation.
import torch
from torch.utils.data import Dataset, DataLoader
import os
import open3d as o3d
import numpy as np
import math
import copy

def farthest_point_sample(data, npoints):
    """
    Args:
        data: Input tensor with shape N,D (N: number of points, D: dimensions)
        npoints: Number of points to sample

    Returns: Tensor composed of sampled points, where each row represents a sampled point
    """
    N, D = data.shape  # N is number of points, D is dimensions
    xyz = data[:, :3]  # Only need coordinates
    centroids = torch.zeros(size=(npoints,))  # Final indices of sampled points
    distance = torch.ones(size=(N,)) * 1e10  # Distance list, initialized to a sufficiently large value to ensure update in first round
    farthest = torch.randint(low=0, high=N, size=(1,))  # Randomly select initial sampled point index
    for i in range(npoints):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = ((xyz - centroid)**2).sum(dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, dim=-1)
    data = data[centroids.type(torch.long)]
    return data

def mirror_point_cloud(point_cloud, normal, point_on_plane):
    """
    Perform mirror reflection on a point cloud

    Parameters:
    point_cloud: Open3D PointCloud object
    normal: Normal vector of the mirror plane [a, b, c]
    point_on_plane: A point on the mirror plane [x0, y0, z0]

    Returns:
    Reflected point cloud object
    """
    # Normalize the normal vector
    normal = np.array(normal) / np.linalg.norm(normal)

    # Construct reflection transformation matrix
    # Reflection matrix formula: I - 2 * n * n^T, where n is the unit normal vector
    n = normal.reshape(3, 1)
    reflection_matrix = np.eye(3) - 2 * np.dot(n, n.T)

    # Translation vector (from origin to the point on the plane)
    translation = np.array(point_on_plane)

    # Create a deep copy of the point cloud using deepcopy
    mirrored_cloud = copy.deepcopy(point_cloud)
    points = np.asarray(mirrored_cloud.points)

    # First translate to origin, apply reflection, then translate back
    for i in range(len(points)):
        # Translate to origin
        point = points[i] - translation
        # Apply reflection
        point = np.dot(reflection_matrix, point)
        # Translate back
        points[i] = point + translation

    return mirrored_cloud

def pc_normalize(pc, pc2=None):
    """
    Normalize a point cloud, and optionally apply the same normalization parameters to a second point cloud

    Parameters:
    pc: First point cloud (Open3D PointCloud object or its point set)
    pc2: Second point cloud (Open3D PointCloud object or its point set), optional

    Returns:
    If only pc is provided: Normalized pc point cloud
    If both pc and pc2 are provided: Normalized pc point cloud, normalized pc2 point cloud
    """
    # Process the first point cloud
    pc_np = np.asarray(pc)
    centroid = np.mean(pc_np, axis=0)
    pc_np = pc_np - centroid
    m = np.max(np.sqrt(np.sum(pc_np ** 2, axis=1)))
    pc_np = pc_np / m
    pc_normalized = o3d.utility.Vector3dVector(pc_np)

    # If no second point cloud is provided, return only the normalized first point cloud
    if pc2 is None:
        return pc_normalized

    # Process the second point cloud using the same normalization parameters as the first
    pc2_np = np.asarray(pc2)
    pc2_np = pc2_np - centroid  # Use the same centroid
    pc2_np = pc2_np / m  # Use the same scale factor
    pc2_normalized = o3d.utility.Vector3dVector(pc2_np)

    return pc_normalized, pc2_normalized

def process(pcd, pcd2):
    normal = [0, 1, 0]  # Normal vector along X-axis
    point_on_plane = [0, 0, 0]  # Origin point
    # Perform mirror reflection
    pcd = mirror_point_cloud(pcd, normal, point_on_plane)
    pcd2 = mirror_point_cloud(pcd2, normal, point_on_plane)
    
    # Create rotation matrix: rotate 90 degrees around y-axis
    R_y = pcd.get_rotation_matrix_from_xyz((0, math.pi / 2, 0))

    # Create rotation matrix: rotate 90 degrees around x-axis
    R_x = pcd.get_rotation_matrix_from_xyz((-math.pi / 2, 0, 0))

    # Combine rotation matrices (first around y-axis, then around x-axis)
    R = np.dot(R_x, R_y)

    # Apply rotation
    pcd.rotate(R, center=(0, 0, 0))
    pcd2.rotate(R, center=(0, 0, 0))
    
    pcd.points, pcd2.points = pc_normalize(pcd.points, pcd2.points)
    return pcd, pcd2

# Custom dataset class
class PCDDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = []
        self.label_files = []
        for file in os.listdir(data_dir):
            if file.endswith('.pcd'):
                parts = file.split('_')
                index_str = parts[-1].replace('.pcd', '')
                index = int(index_str)
                if index in [1, 11, 21, 31]:
                    self.label_files.append(os.path.join(data_dir, file))
                    self.data_files.append(os.path.join(data_dir, file))
                else:
                    self.data_files.append(os.path.join(data_dir, file))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        # Extract object_name, scale and index
        parts = data_file.split('_')
        object_name = "_".join(parts[:-2])
        scale = parts[-2].replace('scale', '')
        index_str = parts[-1].replace('.pcd', '')
        index = int(index_str)

        if 1 <= index <= 10:
            label_index = 1
        elif 11 <= index <= 20:
            label_index = 11
        elif 21 <= index <= 30:
            label_index = 21
        else:
            label_index = 31

        # Filter label files matching object_name, scale and label_index
        label_file = [f for f in self.label_files if object_name in f and f'scale{scale}' in f and f'_{label_index}.pcd' in f][0]

        data_pcd = o3d.io.read_point_cloud(data_file)
        label_pcd = o3d.io.read_point_cloud(label_file)
        data_pcd, label_pcd = process(data_pcd, label_pcd)
        
        # Extract point coordinates and color information
        data_points = torch.tensor(data_pcd.points, dtype=torch.float32)
        data_colors = torch.tensor(data_pcd.colors, dtype=torch.float32)
        label_points = torch.tensor(label_pcd.points, dtype=torch.float32)
        label_colors = torch.tensor(label_pcd.colors, dtype=torch.float32)

        # Merge point coordinates and color information
        # data = torch.cat((data_points, data_colors), dim=1)
        # label = torch.cat((label_points, label_colors), dim=1)
        data = data_points
        
        # Filter points in label with color [1, 1, 0]
        target_color = torch.tensor([1, 1, 0], dtype=torch.float32)
        mask = torch.all(label_colors == target_color, dim=1)
        label = label_points[mask]
        
        # Use farthest point sampling to downsample/upsample data to 1600 points
        if data.shape[0] > 1600:
            data = farthest_point_sample(data, 1600)
        elif data.shape[0] < 1600:
            padding = torch.zeros((1600 - data.shape[0], data.shape[1]), dtype=torch.float32)
            data = torch.cat([data, padding], dim=0)

        return data, label