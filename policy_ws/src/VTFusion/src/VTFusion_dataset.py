import random
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import configargparse
from torch.utils.data import DataLoader

import random
import torchvision.transforms as transforms
import copy
import pickle
import os
import imageio.v2 as imageio
import open3d as o3d
from pin_forward import get_left_arm_end_effector_pose


def farthest_point_sample(data, npoints):
    """
    Args:
        data: Input tensor with shape N,D (N: number of points, D: dimensions)
        npoints: Number of points to sample

    Returns: Tensor composed of sampled points, each row is a sampled point
    """
    N, D = data.shape  # N is number of points, D is dimensions
    xyz = data[:, :3]  # Only need coordinates
    centroids = torch.zeros(size=(npoints,))  # Final sampled point indices
    distance = torch.ones(size=(N,)) * 1e10  # Distance list, initialized to a sufficiently large value to ensure update in first round
    farthest = torch.randint(low=0, high=N, size=(1,))  # Randomly select initial sampled point index
    for i in range(npoints):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, dim=-1)
    data = data[centroids.type(torch.long)]
    return data


def pc_normalize(pc):
    """
    Normalize the point cloud while preserving color attributes (if present)

    Parameters:
    pc: Open3D PointCloud object

    Returns:
    Normalized Open3D PointCloud object (color attributes remain unchanged)
    """
    # Ensure input is an Open3D PointCloud object
    if not isinstance(pc, o3d.geometry.PointCloud):
        raise TypeError("Input must be an Open3D PointCloud object")
    
    # Extract original point cloud coordinates and colors
    points = np.asarray(pc.points)
    has_colors = pc.has_colors()
    
    if has_colors:
        colors = np.asarray(pc.colors)
    
    # Normalization processing (only for coordinates)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / m
    
    # Create new point cloud object
    normalized_pc = o3d.geometry.PointCloud()
    normalized_pc.points = o3d.utility.Vector3dVector(points)
    
    # Restore color attributes (if present)
    if has_colors:
        normalized_pc.colors = o3d.utility.Vector3dVector(colors)
    
    return normalized_pc

def pc_normalize_max_min(pc, config):
    """
    Normalize the point cloud using max/min values from configuration file, 
    while preserving color attributes (if present)

    Parameters:
    pc: Open3D PointCloud object
    config: Configuration dictionary containing max/min settings for point cloud, format should be:
           {
               'x_min': float,
               'x_max': float,
               'y_min': float,
               'y_max': float,
               'z_min': float,
               'z_max': float
           }

    Returns:
    Normalized Open3D PointCloud object (color attributes remain unchanged)
    """
    # Ensure input is an Open3D PointCloud object
    if not isinstance(pc, o3d.geometry.PointCloud):
        raise TypeError("Input must be an Open3D PointCloud object")
    
    # Verify that configuration contains necessary keys
    required_keys = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Configuration missing required key: {key}")
    
    # Extract original point cloud coordinates and colors
    points = np.asarray(pc.points)
    has_colors = pc.has_colors()
    
    if has_colors:
        colors = np.asarray(pc.colors)
    
    # Normalize coordinates according to configured max/min values
    # Get boundary values from configuration
    x_min, x_max = config['x_min'], config['x_max']
    y_min, y_max = config['y_min'], config['y_max']
    z_min, z_max = config['z_min'], config['z_max']
    
    # Calculate range for each axis
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    # Prevent division by zero
    if x_range == 0 or y_range == 0 or z_range == 0:
        raise ValueError("Max and min values in configuration cannot be equal, will cause division by zero error")
    
    # Normalize to [0, 1] range
    points[:, 0] = (points[:, 0] - x_min) / x_range
    points[:, 1] = (points[:, 1] - y_min) / y_range
    points[:, 2] = (points[:, 2] - z_min) / z_range
    
    # Create new point cloud object
    normalized_pc = o3d.geometry.PointCloud()
    normalized_pc.points = o3d.utility.Vector3dVector(points)
    
    # Restore color attributes (if present)
    if has_colors:
        normalized_pc.colors = o3d.utility.Vector3dVector(colors)
    
    return normalized_pc

def pc_normalize_max_min_hand(pc, config):
    """
    Normalize the hand point cloud using max/min values from configuration file, 
    while preserving color attributes (if present)

    Parameters:
    pc: Open3D PointCloud object
    config: Configuration dictionary containing max/min settings for hand point cloud, format should be:
           {
               'x_min_hand': float,
               'x_max_hand': float,
               'y_min_hand': float,
               'y_max_hand': float,
               'z_min_hand': float,
               'z_max_hand': float
           }

    Returns:
    Normalized Open3D PointCloud object (color attributes remain unchanged)
    """
    # Ensure input is an Open3D PointCloud object
    if not isinstance(pc, o3d.geometry.PointCloud):
        raise TypeError("Input must be an Open3D PointCloud object")
    
    # Verify that configuration contains necessary keys
    required_keys = ['x_min_hand', 'x_max_hand', 'y_min_hand', 'y_max_hand', 'z_min_hand', 'z_max_hand']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Configuration missing required key: {key}")
    
    # Extract original point cloud coordinates and colors
    points = np.asarray(pc.points)
    has_colors = pc.has_colors()
    
    if has_colors:
        colors = np.asarray(pc.colors)
    
    # Normalize coordinates according to configured max/min values
    # Get boundary values from configuration
    x_min, x_max = config['x_min_hand'], config['x_max_hand']
    y_min, y_max = config['y_min_hand'], config['y_max_hand']
    z_min, z_max = config['z_min_hand'], config['z_max_hand']
    
    # Calculate range for each axis
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    # Prevent division by zero
    if x_range == 0 or y_range == 0 or z_range == 0:
        raise ValueError("Max and min values in configuration cannot be equal, will cause division by zero error")
    
    # Normalize to [0, 1] range
    points[:, 0] = (points[:, 0] - x_min) / x_range
    points[:, 1] = (points[:, 1] - y_min) / y_range
    points[:, 2] = (points[:, 2] - z_min) / z_range
    
    # Create new point cloud object
    normalized_pc = o3d.geometry.PointCloud()
    normalized_pc.points = o3d.utility.Vector3dVector(points)
    
    # Restore color attributes (if present)
    if has_colors:
        normalized_pc.colors = o3d.utility.Vector3dVector(colors)
    
    return normalized_pc

def process_joint_angles(joint_angles):
    """Process joint angle data:
    1. Keep the 9th to 16th joint angles (indices 8-15)
    2. Sum remaining joint angles in pairs
    3. Sum the 4th and 5th elements of the processed data into one value
    4. Remove the 15th, 19th, 23rd, and 27th values
    """
    # 1. Keep the 9th to 16th joint angles (indices 8~15)
    retained_joints = joint_angles[:, 8:16]  # (number of frames, 8)
    
    # 2. Sum remaining joint angles in pairs (starting from index 16)
    remaining_joints = joint_angles[:, 16:]  # (number of frames, 40)
    
    # Ensure remaining joint angles count is even
    if remaining_joints.shape[1] % 2 != 0:
        print(f"Warning: Remaining joint angles count is odd, discarding last one: {remaining_joints.shape[1]}")
        remaining_joints = remaining_joints[:, :-1]  # Discard last one to ensure even count
    
    # Reshape to (number of frames, 20, 2) and sum along last dimension
    summed_joints = remaining_joints.reshape(
        remaining_joints.shape[0], 
        remaining_joints.shape[1] // 2, 
        2
    ).sum(axis=2)  # (number of frames, 20)
    
    # 3. Merge results
    processed_joints = np.concatenate([retained_joints, summed_joints], axis=1)  # (number of frames, 28)
    
    # 4. Sum the 4th and 5th elements into one value
    new_joint_angles = np.zeros((processed_joints.shape[0], processed_joints.shape[1] - 1))  # (number of frames, 27)
    
    # Copy first 3 elements
    new_joint_angles[:, :3] = processed_joints[:, :3]
    
    # Sum 4th and 5th elements, put into new 4th position
    new_joint_angles[:, 3] = processed_joints[:, 3] + processed_joints[:, 4]
    
    # Copy remaining elements (starting from index 5)
    new_joint_angles[:, 4:] = processed_joints[:, 5:]
    # 5. Remove the 15th, 19th, 23rd, and 27th values (indices 14, 18, 22, 26)
    indices_to_remove = [14, 18, 22, 26]
    new_joint_angles = np.delete(new_joint_angles, indices_to_remove, axis=1)
    return new_joint_angles


class VT_dataset(Dataset):
    """Dataset class for loading robot manipulation data, containing all episodes directly"""

    def __init__(self, dataset_name, config, task_name) -> None:
        super().__init__()
        
        self.pred_horizon = config['fusion_net']['diffusion']['pred_time']
        self.obs_horizon = config['obs_t']
        self.tactile_point_num = config.get('tactile_point_num', 50)  
        self.hand_object_point_num = config.get('hand_object_point_num', 1000) 
        self.overall_point_num = config.get('overall_point_num', 1000)  
        self.episodes = []  # Store information of all episodes
        self.samples_lookup = []  # Store sample index information
        self.episode_initial_joints = {}  # Store initial joint angles for each episode
        self.prefix_length = config['prefix_length']  # Prefix length
        self.config = config
        # Example: Check sample count for each episode
        # print(dataset.get_task_samples_count())
        # Output: {'episode_ep000': 100, 'episode_ep001': 85, ...}
        self.task_samples_count = {}  # Sample count for each task

        self.task_name = task_name
        self.data_dir = dataset_name + self.task_name
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset and generate samples, including all episodes"""
        # Get all episode directories
        episode_dirs = sorted([d for d in os.listdir(self.data_dir) 
                              if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith("ep")])
        
        for ep_dir in episode_dirs:
            ep_path = os.path.join(self.data_dir, ep_dir)
            meta_path = os.path.join(ep_path, "meta.npz")
            
            if not os.path.exists(meta_path):
                print(f"Warning: Episode {ep_dir} is missing metadata file, skipping")
                continue
                
            # Load metadata
            meta_data = np.load(meta_path)
            timestamps = meta_data['timestamp'].flatten()
            joint_angles = meta_data['robot_arm_joint']
            end_state_poses = meta_data['end_state_pose']
            end_target_poses = meta_data['end_target_pose']
            
            # Determine task type (simplified to episode number here, can get from annotation file in practical application)
            episode_name = f"episode_{ep_dir}"
            # Record initial joint angles
            self.episode_initial_joints[episode_name] = joint_angles[0].copy()

            # Update task count
            if episode_name not in self.task_samples_count:
                self.task_samples_count[episode_name] = 0
                          
            # Calculate available sample count in episode
            ep_length = len(timestamps) + self.prefix_length  # Add prefix length
            if ep_length < self.obs_horizon + self.pred_horizon:
                print(f"Warning: Episode {ep_dir} is too short, skipping")
                continue
                
            # Generate samples
            for pred_start_idx in range(0, ep_length - self.pred_horizon + 1):
                pred_end_idx = pred_start_idx + self.pred_horizon - 1
                obs_start_idx = pred_start_idx
                obs_end_idx = obs_start_idx + self.obs_horizon - 1

                # Skip samples exceeding episode length
                if obs_end_idx >= ep_length or pred_end_idx > ep_length:
                    continue

                sample_info = {
                    "task_name": self.task_name,
                    "episode_name": episode_name,
                    "episode_dir": ep_dir,
                    "pred_start_idx": pred_start_idx,
                    "pred_end_idx": pred_end_idx,
                    "obs_start_idx": obs_start_idx,
                    "obs_end_idx": obs_end_idx,
                    "timestamp_start":  timestamps[max(0, pred_start_idx - self.prefix_length)],
                    "timestamp_end": timestamps[min(len(timestamps) - 1, pred_end_idx - self.prefix_length)]
                }
                
                self.samples_lookup.append(sample_info)
                self.task_samples_count[episode_name] += 1
                
            # Record episode information
            self.episodes.append({
                "task_name":  self.task_name,
                "dir": ep_dir,
                "episode_name": episode_name,
                "length": ep_length,
                "timestamp_start": timestamps[0],
                "timestamp_end": timestamps[-1]
            })
    
    def __len__(self):
        """Return total number of samples"""
        return len(self.samples_lookup)
    
    def __getitem__(self, idx):
        """Get sample at specified index"""
        sample = self.samples_lookup[idx]
        episode_dir = sample["episode_dir"]
        pred_start = sample["pred_start_idx"]
        pred_end = sample["pred_end_idx"]
        obs_start = sample["obs_start_idx"]
        obs_end = sample["obs_end_idx"]
        # Load observation and prediction data
        observations = self._load_observation_data(episode_dir, obs_start, obs_end)
        predictions = self._load_prediction_data(episode_dir, pred_start, pred_end)

                
        # -------------------------- 1. Point Cloud Data Conversion --------------------------
        # mini_pointnet_input: Tactile point clouds (tactile_pointclouds, coordinates only)
        mini_pointnet_input = []
        for pcd in observations["tactile_pointclouds"]:
            if pcd is not None:
                # Extract tactile point cloud coordinates and convert to numpy array
                tactile_pointcloud = np.asarray(pcd.points)
                
                # Convert to torch tensor for processing
                tactile_tensor = torch.from_numpy(tactile_pointcloud).float()
                
                # Ensure point cloud count meets requirements
                if tactile_tensor.shape[0] > self.tactile_point_num:
                    # Too many points: sample to specified count
                    tactile_tensor = farthest_point_sample(tactile_tensor, self.tactile_point_num)
                elif tactile_tensor.shape[0] < self.tactile_point_num:
                    # Too few points: pad to specified count
                    pad_num = self.tactile_point_num - tactile_tensor.shape[0]
                    tactile_tensor = torch.nn.functional.pad(
                        tactile_tensor, (0, 0, 0, pad_num), mode='constant', value=0.0
                    )
                
                # Convert back to numpy array and add to list
                tactile_np = tactile_tensor.numpy()
                mini_pointnet_input.append(tactile_np)
            else:
                # Point cloud does not exist: create zero array with specified shape
                zero_pointcloud = np.zeros((self.tactile_point_num, 3), dtype=np.float32)
                mini_pointnet_input.append(zero_pointcloud)

        # Verify all elements have consistent shape
        for i, item in enumerate(mini_pointnet_input):
            if item.shape != (self.tactile_point_num, 3):
                raise ValueError(f"Tactile point cloud element {i} has wrong shape: expected ({self.tactile_point_num}, 3), got {item.shape}")

        # Convert to numpy array and torch tensor
        mini_pointnet_input = np.array(mini_pointnet_input, dtype=np.float32)
        mini_pointnet_input = torch.from_numpy(mini_pointnet_input)
            
        # point_encoder_input: pointclouds2 (coordinates only)
        point_encoder_input = []
        for pcd in observations["pointclouds2"]:
            if pcd is not None:
                # Extract point cloud coordinates and convert to numpy array
                hand_object_pointcloud = np.asarray(pcd.points)  # Ensure it's numpy array
                if hand_object_pointcloud.shape[0] > self.hand_object_point_num:
                    # Convert to torch tensor for sampling
                    hand_tensor = torch.from_numpy(hand_object_pointcloud).float()
                    sampled_tensor = farthest_point_sample(hand_tensor, self.hand_object_point_num)
                    hand_object_pointcloud = sampled_tensor.numpy()  # Convert back to numpy
                elif hand_object_pointcloud.shape[0] < self.hand_object_point_num:
                    # Pad with numpy
                    pad_num = self.hand_object_point_num - hand_object_pointcloud.shape[0]
                    hand_object_pointcloud = np.pad(
                        hand_object_pointcloud, 
                        ((0, pad_num), (0, 0)),  # Pad only in point count dimension
                        mode='constant'
                    )
                # Verify shape
                assert hand_object_pointcloud.shape == (self.hand_object_point_num, 3), \
                    f"Point cloud shape error: expected ({self.hand_object_point_num}, 3), got {hand_object_pointcloud.shape}"
                point_encoder_input.append(hand_object_pointcloud)
            else:
                # Create numpy array directly when point cloud does not exist
                zero_pointcloud = np.zeros((self.hand_object_point_num, 3), dtype=np.float32)
                point_encoder_input.append(zero_pointcloud)
        
        # Now safely convert to numpy array and torch tensor
        point_encoder_input = np.array(point_encoder_input, dtype=np.float32)  # First convert to numpy array
        point_encoder_input = torch.from_numpy(point_encoder_input)  # Then convert to torch tensor
        # overall_point_encoder_input: pointclouds1 (coordinates + colors, 6-dimensional points)
        overall_point_encoder_input = []
        for pcd in observations["pointclouds1"]:
            if pcd is not None and pcd.has_colors():
                # Extract coordinates (n_points, 3) and colors (n_points, 3)
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                # Concatenate to 6-dimensional points (n_points, 6)
                combined = np.concatenate([points, colors], axis=1)
                
                # Convert to torch tensor for sampling
                combined_tensor = torch.from_numpy(combined).float()
                if combined_tensor.shape[0] > self.overall_point_num:
                    # Sample to specified point count
                    combined_tensor = farthest_point_sample(combined_tensor, self.overall_point_num)
                elif combined_tensor.shape[0] < self.overall_point_num:
                    # Pad to specified point count
                    pad_num = self.overall_point_num - combined_tensor.shape[0]
                    combined_tensor = torch.nn.functional.pad(
                        combined_tensor, (0, 0, 0, pad_num), mode='constant', value=0.0
                    )
                
                # Convert back to numpy array and add to list
                combined_np = combined_tensor.numpy()
                overall_point_encoder_input.append(combined_np)
                
            elif pcd is not None:
                # If coordinates exist but no colors, pad color channel with 0
                points = np.asarray(pcd.points)
                colors = np.zeros_like(points)  # (n_points, 3) all zeros
                combined = np.concatenate([points, colors], axis=1)
                
                # Convert to torch tensor for processing
                combined_tensor = torch.from_numpy(combined).float()
                if combined_tensor.shape[0] > self.overall_point_num:
                    combined_tensor = farthest_point_sample(combined_tensor, self.overall_point_num)
                elif combined_tensor.shape[0] < self.overall_point_num:
                    pad_num = self.overall_point_num - combined_tensor.shape[0]
                    combined_tensor = torch.nn.functional.pad(
                        combined_tensor, (0, 0, 0, pad_num), mode='constant', value=0.0
                    )
                
                # Convert back to numpy array and add to list
                combined_np = combined_tensor.numpy()
                overall_point_encoder_input.append(combined_np)
                
            else:
                # Point cloud does not exist, pad with empty array (ensure correct shape)
                zero_pointcloud = np.zeros((self.overall_point_num, 6), dtype=np.float32)
                overall_point_encoder_input.append(zero_pointcloud)

        # Verify all elements have consistent shape
        for i, item in enumerate(overall_point_encoder_input):
            if item.shape != (self.overall_point_num, 6):
                raise ValueError(f"Element {i} has wrong shape: expected ({self.overall_point_num}, 6), got {item.shape}")

        # Convert to numpy array and torch tensor
        overall_point_encoder_input = np.array(overall_point_encoder_input, dtype=np.float32)
        overall_point_encoder_input = torch.from_numpy(overall_point_encoder_input)


        # -------------------------- 2. Tactile Images (force_encoder_input) --------------------------
        force_encoder_input = np.array(observations["tactile_images"])
        force_encoder_input = torch.from_numpy(force_encoder_input).float()
        
        # -------------------------- 3. State Input (state_input): end_poses concatenated with joint_angles --------------------------
        end_poses = observations["end_poses"]  # (obs_horizon, 6)
        joint_angles = observations["joint_angles"]  # (obs_horizon, N)
        state_input = np.concatenate([end_poses, joint_angles], axis=1)  # (obs_horizon, 6+N)
        state_input = torch.from_numpy(state_input).float()
        # -------------------------- 4. Relative Actions (rel_actions): target_poses concatenated with target_joint_angles --------------------------
        target_poses = predictions["target_poses"]  # (pred_horizon, 6)
        target_joint_angles = predictions["target_joint_angles"]  # (pred_horizon, M)

        rel_actions = np.concatenate([target_poses, target_joint_angles], axis=1)  # (pred_horizon, 6+M)
        rel_actions = torch.from_numpy(rel_actions).float()
        # Organize into nbatch tuple
        nbatch = (
            mini_pointnet_input.float(),
            point_encoder_input.float(),
            overall_point_encoder_input.float(),
            force_encoder_input.float(),
            state_input.float(),
            rel_actions.float()
        )
        # Construct sample data
        sample_data = {
            "task_name": sample["task_name"],
            "episode_name": sample["episode_name"],
            "episode_dir": episode_dir,
            "timestamps": np.load(os.path.join(self.data_dir, episode_dir, "meta.npz"))["timestamp"].flatten(),
            "observations": observations,
            "predictions": predictions
        }

        # return sample_data,nbatch
        return nbatch
    
    def _generate_joint_prefix(self, initial_joints):
        """Generate smooth transition sequence from 0 to initial joint angles, keep first 16 joints unchanged"""
        prefix = np.zeros((self.prefix_length, initial_joints.shape[0]))
        
        # Keep first 16 joints unchanged
        prefix[:, :16] = initial_joints[:16]
        
        # Smooth transition for remaining joints
        for i in range(self.prefix_length):
            ratio = (i + 1) / self.prefix_length
            prefix[i, 16:] = initial_joints[16:] * ratio
        
        return prefix
    
    def _load_observation_data(self, episode_dir: str, start_idx: int, end_idx: int):
        """Load data within observation window"""
        ep_path = os.path.join(self.data_dir, episode_dir)
        meta_data = np.load(os.path.join(ep_path, "meta.npz"))

        episode_name = f"episode_{episode_dir}"
        
        # Get initial joint angles
        initial_joints = self.episode_initial_joints[episode_name]
        
        # Calculate actual indices in original data (subtract prefix length)
        actual_start = max(0, start_idx - self.prefix_length)
        actual_end = max(0, end_idx - self.prefix_length)

        # Load joint angles
        joint_angles = meta_data["robot_arm_joint"][actual_start:actual_end+1]  # +1 to include end_idx

        # Add prefix if needed
        if start_idx < self.prefix_length:
            prefix = self._generate_joint_prefix(initial_joints)
            # Calculate number of frames needed from prefix
            prefix_frames_needed = self.obs_horizon - len(joint_angles)
            joint_angles = np.vstack([prefix[-prefix_frames_needed:], joint_angles])
        
        # Process joint angles
        joint_angles = process_joint_angles(joint_angles)
        end_poses = get_left_arm_end_effector_pose(joint_angles)  # Get end effector pose
        # 5. Remove first 7 values
        indices_to_remove = list(range(7))
        new_joint_angles = np.delete(joint_angles, indices_to_remove, axis=1)

        # Load tactile images
        tactile_images = []
        for i in range(start_idx, end_idx+1):
            # Map to original data index
            original_idx = i - self.prefix_length
            if original_idx < 0:
                # Use first frame image for prefix part
                img_file = os.path.join(ep_path, "tactile_image", f"{0:04d}.png")
            else:
                img_file = os.path.join(ep_path, "tactile_image", f"{original_idx:04d}.png")
            
            if os.path.exists(img_file):
                tactile_images.append(imageio.imread(img_file))
            else:
                tactile_images.append(None)  # Fill with None if image does not exist
        
        # Load point clouds
        pointclouds1 = []  # First point cloud
        pointclouds2 = []  # Second point cloud
        tactile_pointclouds = []  # Tactile point cloud
        for i in range(start_idx, end_idx+1):
            # Map to original data index
            original_idx = i - self.prefix_length
            
            # Load first point cloud
            if original_idx < 0:
                # Use first frame point cloud for prefix part
                pcd1_file = os.path.join(ep_path, "point_cloud", f"{0:04d}.pcd")
            else:
                pcd1_file = os.path.join(ep_path, "point_cloud", f"{original_idx:04d}.pcd")
            
            if os.path.exists(pcd1_file):
                pcd1 = o3d.io.read_point_cloud(pcd1_file)
                pcd1 = pc_normalize_max_min(pcd1, self.config)
                pointclouds1.append(pcd1)
                # Extract red points as tactile point cloud
                if pcd1.has_colors():
                    # Convert point cloud data to numpy array
                    points = np.asarray(pcd1.points)
                    colors = np.asarray(pcd1.colors)
                    
                    # Filter red points (custom threshold: R > G+B and R > 0.3)
                    # Threshold can be adjusted according to actual scenario
                    red_mask = (
                            (colors[:, 0] > 0.99) &
                            (colors[:, 1] < 0.01) &  
                            (colors[:, 2] < 0.01) 
                        )
                    
                    # Create new tactile point cloud
                    tactile_pcd = o3d.geometry.PointCloud()
                    tactile_pcd.points = o3d.utility.Vector3dVector(points[red_mask])
                    tactile_pcd.colors = o3d.utility.Vector3dVector(colors[red_mask])
                    
                    tactile_pointclouds.append(tactile_pcd)
                    # Remove red points if needed, then update original point cloud pcd1
                    if self.config['no_tactile']:
                        non_red_mask = ~red_mask  # Invert mask to keep non-red points
                        pcd1.points = o3d.utility.Vector3dVector(points[non_red_mask])
                        pcd1.colors = o3d.utility.Vector3dVector(colors[non_red_mask])
                        # Remove last added original point cloud, add filtered one
                        pointclouds1.pop()  # Remove original
                        pointclouds1.append(pcd1)  # Re-add filtered point cloud
                else:
                    # Point cloud without color information
                    tactile_pointclouds.append(None)
            else:
                pointclouds1.append(None)
                tactile_pointclouds.append(None)
                
  # -------------------------- Load second point cloud (added red point removal logic) --------------------------
            if original_idx < 0:
                # Use first frame point cloud for prefix part
                pcd2_file = os.path.join(ep_path, "point_cloud2", f"{0:04d}.pcd")
            else:
                pcd2_file = os.path.join(ep_path, "point_cloud2", f"{original_idx:04d}.pcd")

            if os.path.exists(pcd2_file):
                pcd2 = o3d.io.read_point_cloud(pcd2_file)
                pcd2 = pc_normalize_max_min_hand(pcd2, self.config)
                
                # -------------------------- Added: Red point removal logic for second point cloud --------------------------
                # 1. First check if point cloud has color information (no filtering if no color)
                if pcd2.has_colors() and self.config['no_tactile']:  # Reuse no_tactile config to uniformly control red point removal
                    # 2. Convert point cloud and colors to numpy arrays (for easy filtering)
                    pcd2_points = np.asarray(pcd2.points)
                    pcd2_colors = np.asarray(pcd2.colors)
                    
                    # 3. Filter red points with same threshold as first point cloud (ensure logic consistency)
                    pcd2_red_mask = (
                            (pcd2_colors[:, 0] > 0.99) &  # R channel close to 1 (red)
                            (pcd2_colors[:, 1] < 0.01) &  # G channel close to 0 (no green)
                            (pcd2_colors[:, 2] < 0.01)    # B channel close to 0 (no blue)
                        )
                    
                    # 4. Invert mask to keep non-red points
                    pcd2_non_red_mask = ~pcd2_red_mask
                    
                    # 5. Update point cloud data (keep only non-red points)
                    pcd2.points = o3d.utility.Vector3dVector(pcd2_points[pcd2_non_red_mask])
                    pcd2.colors = o3d.utility.Vector3dVector(pcd2_colors[pcd2_non_red_mask])
                # -------------------------- End of added logic --------------------------
                
                # Add filtered second point cloud to list
                pointclouds2.append(pcd2)
            else:
                pointclouds2.append(None)
        
        return {
            "joint_angles": new_joint_angles,
            "end_poses": end_poses,
            "tactile_images": tactile_images,
            "pointclouds1": pointclouds1,
            "pointclouds2": pointclouds2,
            "tactile_pointclouds": tactile_pointclouds,
            "start_idx": start_idx,
            "end_idx": end_idx
        }
    
    def _load_prediction_data(self, episode_dir: str, start_idx: int, end_idx: int):
        """Load data within prediction window, handle prefix logic"""
        ep_path = os.path.join(self.data_dir, episode_dir)
        meta_data = np.load(os.path.join(ep_path, "meta.npz"))
        episode_name = f"episode_{episode_dir}"
        
        # Get initial joint angles for generating prefix
        initial_joints = self.episode_initial_joints[episode_name]
        initial_pose = meta_data["end_state_pose"][0, :6].reshape(1, 6)  # First 6 values of first frame
        # Calculate actual index range in original data
        actual_start = max(0, start_idx - self.prefix_length)
        actual_end = max(0, end_idx - self.prefix_length)
        
        # Get target poses from meta file
        end_target_poses = meta_data["end_target_pose"]
        
        # Extract target poses within prediction window
        target_poses = end_target_poses[actual_start:actual_end+1]  # +1 to include end_idx
        target_poses = target_poses[:, :6]  # Keep first 6 values of each frame
        
        # Generate complete joint angle sequence (including prefix)
        original_joints = meta_data["robot_arm_joint"]
        prefix = self._generate_joint_prefix(initial_joints)
        full_joint_sequence = np.vstack([prefix, original_joints])
        
        if end_idx + 2 <= len(full_joint_sequence):
            # Extract target joint angles for prediction window (start from next frame of current frame)
            target_joint_angles = full_joint_sequence[start_idx+1:end_idx+2]
        else:
            target_joint_angles = full_joint_sequence[start_idx+1:len(full_joint_sequence)]
            last_frame = meta_data["robot_arm_joint"][-1].reshape(1, -1)  # Ensure shape matches
            need_frame = self.pred_horizon - len(target_joint_angles)
            if need_frame > 0:
                # Pad with last frame joint angles if needed
                last_frame = np.repeat(last_frame, need_frame, axis=0)
                target_joint_angles = np.vstack([target_joint_angles, last_frame])

        # Process joint angles
        target_joint_angles = process_joint_angles(target_joint_angles)
        target_poses = get_left_arm_end_effector_pose(target_joint_angles)  # Get end effector pose
        # 5. Remove first 7 values
        indices_to_remove = list(range(7))
        new_joint_angles = np.delete(target_joint_angles, indices_to_remove, axis=1)
        # Load corresponding timestamps (only return timestamps of actual data, no real timestamps for prefix area)
        timestamps = meta_data["timestamp"].flatten()[actual_start:actual_end+1]
        
        return {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "timestamps": timestamps,
            "target_poses": target_poses,
            "target_joint_angles": new_joint_angles
        }