import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from easydict import EasyDict as edict
import random
from transformer import TransformerEncoder, TransformerDecoder, Group, DummyGroup, Encoder
from torch.utils.data import Dataset, DataLoader
import os
import open3d as o3d
from torchmetrics import Accuracy
from Dataset_process_nor import PCDDataset
import sys
import os

import argparse  # Add argparse for command line arguments
sys.path.append("..")

from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from extensions.pointops.functions import pointops
import numpy as np
from torch.utils.data import DataLoader, random_split
import copy
import math

from models.PretrainPoint import PretrainPoint


def compute_chamfer_distance(pred_pc, labels_pc):
    """（Chamfer Distance）"""

    if not isinstance(pred_pc, o3d.geometry.PointCloud):
        pred_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_pc))
    if not isinstance(labels_pc, o3d.geometry.PointCloud):
        labels_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(labels_pc))
    

    labels_tree = o3d.geometry.KDTreeFlann(labels_pc)
    pred_tree = o3d.geometry.KDTreeFlann(pred_pc)
    
    pred_points = np.asarray(pred_pc.points)
    labels_points = np.asarray(labels_pc.points)
    
 
    if len(pred_points) == 0 or len(labels_points) == 0:
        return float('inf')

    pred_to_labels = 0.0
    for point in pred_points:
        [k, idx, _] = labels_tree.search_knn_vector_3d(point, 1)
        pred_to_labels += np.sum((point - labels_points[idx[0]]) ** 2)
    pred_to_labels /= len(pred_points)

    labels_to_pred = 0.0
    for point in labels_points:
        [k, idx, _] = pred_tree.search_knn_vector_3d(point, 1)
        labels_to_pred += np.sum((point - pred_points[idx[0]]) ** 2)
    labels_to_pred /= len(labels_points)
    
    return pred_to_labels + labels_to_pred


class Config:
    def __init__(self):
        # Set configuration parameters (mask_ratio will be overridden by cmd args)
        self.transformer_config = edict({
            'mask_ratio': 0.70,  # Default value (will be replaced by command line argument)
            'trans_dim': 384,
            'depth': 12,
            'dec_depth': 1,
            'use_sigmoid': True,
            'use_moco_loss': True,
            'query_loss_weight': 1.0,
            'moco_loss_weight': 0.01,
            'use_focal_loss': False,
            'focal_loss_alpha': 0.25,
            'focal_loss_gamma': 2,
            'ambiguous_threshold': -1,
            'ambiguous_dynamic_threshold': 256,
            'dec_query_mode': 'points',     # [center, points]
            'dec_query_real_num': 1,
            'dec_query_fake_num': 100000,
            'drop_path_rate': 0.1,
            'cls_dim': 512,
            'num_heads': 6,
            'group_size': 32, 
            'num_group': 64, 
            'encoder_dims': 256
        })
        self.m = 0.999
        self.T = 0.07
        self.K = 16384
        self.pos_embed_hidden_dim = 128

def farthest_point_sample(data, npoints):
    """
    Farthest Point Sampling (FPS) algorithm
    
    Args:
        data: Input tensor with shape [N, D] (N=number of points, D=dimension)
        npoints: Number of points to sample
    
    Returns:
        Sampled point tensor with shape [npoints, D]
    """
    
    N, D = data.shape  # N: number of points, D: dimension
    if N == 0:
        return torch.empty_like(data) 
    if npoints > N:
        npoints = N 
    xyz = data[:, :3]  # Only need coordinate information
    centroids = torch.zeros(size=(npoints,))  # Final sampled point indices
    distance = torch.ones(size=(N,)) * 1e10  # Distance list (initialized to large value)
    farthest = torch.randint(low=0, high=N, size=(1,))  # Random initial sample index
    
    for i in range(npoints):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = ((xyz - centroid)**2).sum(dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, dim=-1)
    
    data = data[centroids.type(torch.long)]
    return data


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluation with Command Line Args')
    parser.add_argument('--mask_ratio', type=float, default=0.70, 
                        help='Mask ratio for point cloud masking (default: 0.70)')
    parser.add_argument('--ckpt_path', type=str, required=True, 
                        help='Path to the pretrained model checkpoint file (e.g., /path/to/ckpt-best.pth)')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing the PCD dataset (e.g., /path/to/touch_dataset_4)')
    args = parser.parse_args()

    # Instantiate config and override mask_ratio with command line argument
    config = Config()
    config.transformer_config.mask_ratio = args.mask_ratio

    # Instantiate transformer_q model
    transformer_q =  PretrainPoint(config).transformer_q

    # Load model weights from checkpoint (path from command line)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")

    # Extract transformer_q weights from checkpoint
    transformer_q_weights = {}
    base_model = ckpt['base_model']

    for key, value in base_model.items():
        if key.startswith('module.transformer_q.'):
            # Remove prefix to match model weight names
            new_key = key[len('module.transformer_q.'):]
            transformer_q_weights[new_key] = value

    # Load weights into model
    transformer_q.load_state_dict(transformer_q_weights)
    transformer_q = transformer_q.cuda()
    # Set model to evaluation mode
    transformer_q.eval()

    # Load dataset (directory from command line)
    dataset = PCDDataset(args.data_dir)
    
    # Split dataset (0% train, 100% validation)
    train_size = int(0 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Store Chamfer Distance values for all samples
    all_cd_values = []
    test_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    for idx, (data, label) in enumerate(test_dataloader):
        with torch.no_grad():
            data = data.cuda()
            label = label.cuda()

            # Generate point cloud groups
            neighborhood, center = Group(64, 32)(data)
            neighborhood = neighborhood.cuda()
            center = center.cuda()

            # Forward pass through model
            q_cls_feature, query_preds, query_labels, query_points = transformer_q(
                neighborhood, center, data, label,return_cd=True
            )

            # Process predictions
            query_preds = query_preds.transpose(1, 2).contiguous().to(data.device)
            query_preds = torch.sigmoid(query_preds)
            
            # Get indices with prediction probability > threshold for class 1
            threshold = 0.45
            pos_indices = (query_preds[..., 1] > threshold).nonzero(as_tuple=True)

            # Extract corresponding query points
            selected_points = query_points[pos_indices].cpu().numpy()
            selected_points = torch.tensor(np.array(selected_points), dtype=torch.float32)

            # Resample to 1000 points using FPS
            selected_points = farthest_point_sample(selected_points, 1000)

            # Create Open3D point cloud for predictions
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(selected_points)
            
            # Create Open3D point cloud for ground truth
            label_numpy = label[0].detach().cpu().numpy().astype(np.float64)
            label_pcd = o3d.geometry.PointCloud()
            label_pcd.points = o3d.utility.Vector3dVector(label_numpy)

            # Calculate Chamfer Distance
            cd = compute_chamfer_distance(pred_pcd, label_pcd)
            print(f'Sample {idx+1}/{len(test_dataloader)} CD: {cd:.6f}')
        
            # Save CD value for current sample
            all_cd_values.append(cd)
    
     # Calculate average CD value
    if all_cd_values:
        valid_cd = [v for v in all_cd_values if not np.isinf(v)]
        if valid_cd:
            avg_cd = np.mean(valid_cd)
            print(f'\nAverage Chamfer Distance: {avg_cd:.6f}')
    else:
        print('No valid CD values were collected.')