import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from easydict import EasyDict as edict
import random
from transformer import TransformerEncoder, TransformerDecoder, Group, DummyGroup, Encoder
from detr.build import build_encoder as build_encoder_3detr, build_preencoder as build_preencoder_3detr
from torch.utils.data import Dataset, DataLoader
import os
import open3d as o3d
from torchmetrics import Accuracy
from Dataset_process_nor import PCDDataset
import sys
import argparse  # Add argparse for command line arguments
sys.path.append("..")
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from extensions.pointops.functions import pointops
import numpy as np
from torch.utils.data import DataLoader, random_split
import copy
import math
from matplotlib import cm
import matplotlib.pyplot as plt  # Add matplotlib import


class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate 
        self.cls_dim = config.cls_dim 
        self.num_heads = config.num_heads 

        self.group_size = config.group_size
        self.num_group = config.num_group
        # Grouper module
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # Define encoder
        self.encoder_dims =  config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # Bridge encoder and transformer
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_arch = config.get('cls_head_arch', '1x')
        if self.cls_head_arch == '2x':
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )
        else:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()
        
    def build_loss_func(self):
        # Build cross entropy loss function
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, ret, gt):
        # Calculate loss and accuracy
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        # Load model weights from checkpoint
        ckpt = torch.load(bert_ckpt_path, map_location="cpu")
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger = 'Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger = 'Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger = 'Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger = 'Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger = 'Transformer')


    def forward(self, pts, return_feature=False):
        # Divide the point cloud in uniform format (critical step)
        neighborhood, center = self.group_divider(pts)
        # Encode input point cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # Prepare classification token
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # Add position embedding
        pos = self.pos_embed(center)
        # Final input tensor
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # Transformer encoder forward pass
        x = self.blocks(x, pos)
        x = self.norm(x)
        if return_feature: return x
        concat_f = torch.cat([x[:,0], x[:, 1:].max(1)[0]], dim = -1)
        ret = self.cls_head_finetune(concat_f)
        return ret


class MaskPointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # Define encoder parameters
        self.num_group = config.transformer_config.num_group
        self.group_size = config.transformer_config.group_size
        self.encoder_dims = config.transformer_config.encoder_dims
        # Define transformer parameters
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.dec_depth = config.transformer_config.dec_depth
        self.dec_query_mode = config.transformer_config.dec_query_mode
        self.dec_query_real_num = config.transformer_config.dec_query_real_num
        self.dec_query_fake_num = config.transformer_config.dec_query_fake_num
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.cls_dim = config.transformer_config.cls_dim
        self.use_sigmoid = config.transformer_config.use_sigmoid
        self.num_heads = config.transformer_config.num_heads
        self.ambiguous_threshold = config.transformer_config.ambiguous_threshold
        self.ambiguous_dynamic_threshold = config.transformer_config.ambiguous_dynamic_threshold
        print_log(f'[Transformer args] {config.transformer_config}', logger = 'MaskPoint')
        # Define encoder architecture
        self.enc_arch = config.transformer_config.get('enc_arch', 'PointViT')
        if self.enc_arch == '3detr':
            self.encoder = build_preencoder_3detr(num_group=self.num_group, group_size=self.group_size, dim=self.encoder_dims)
        else:
            self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # Bridge encoder and transformer
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        # Define learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        # Position embedding for each patch
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        # Define transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        if self.enc_arch == '3detr':
            self.blocks = build_encoder_3detr(
                ndim=self.trans_dim,
                nhead=self.num_heads,
                nlayers=self.depth
            )
        else:
            self.blocks = TransformerEncoder(
                embed_dim = self.trans_dim,
                depth = self.depth,
                drop_path_rate = dpr,
                num_heads = self.num_heads
            )
        self.decoder = TransformerDecoder(
            embed_dim = self.trans_dim,
            depth = self.dec_depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )
        self.cls_head = nn.Sequential(
            nn.Linear(self.trans_dim, self.cls_dim),
            nn.GELU(),
            nn.Linear(self.cls_dim, self.cls_dim)
        )
        self.bin_cls_head = nn.Sequential(
            nn.Linear(self.trans_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        # Layer normalization
        self.norm = nn.LayerNorm(self.trans_dim)
        # Initialize learnable tokens
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
        self.access_count = 0

    def _init_weights(self, m):
        # Initialize layer weights
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _generate_fake_query(self, target):
        # Generate fake query points for contrastive learning
        B = target.shape[0]
        min_coords, max_coords = torch.min(target, dim=1, keepdim=True)[0], torch.max(target, dim=1, keepdim=True)[0]
        fake_target = torch.rand(B, self.dec_query_fake_num, 3, dtype=target.dtype, device=target.device) * (max_coords - min_coords) + min_coords
        return fake_target

    def _generate_query_xyz(self, points, center, lable_points, mode='center'):
        # Generate query point coordinates (xyz) for decoder
        if mode == 'center':
            target = pointops.fps(lable_points, self.dec_query_real_num)
        elif mode == 'points':
            if self.dec_query_real_num == -1:
                target = lable_points
            else:
                target = pointops.fps(lable_points, self.dec_query_real_num)

        bs, npoints, _ = target.shape
        q, fake_q = target, self._generate_fake_query(points)

        nn_dist = pointops.knn(fake_q, lable_points, 1)[1].squeeze()
        if self.ambiguous_dynamic_threshold > 0:
            assert self.ambiguous_threshold == -1
            if self.ambiguous_dynamic_threshold == self.dec_query_real_num:
                thres_q = q
            else:
                thres_q = pointops.fps(lable_points, self.ambiguous_dynamic_threshold)
            dist_thres = pointops.knn(thres_q, thres_q, 2)[1][..., -1].mean(-1, keepdims=True)
        else:
            assert self.ambiguous_dynamic_threshold == -1
            dist_thres = self.ambiguous_threshold
        queries = torch.cat((q, fake_q), dim=1)
        labels = torch.zeros(bs, queries.shape[1], dtype=torch.long, device=target.device)
        labels[:, :npoints] = 1
        labels[:, npoints:][nn_dist < dist_thres] = -1

        return queries, labels

    def preencoder(self, neighborhood):
        # Pre-encoder for point cloud features
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        return group_input_tokens

    def forward(self, neighborhood, center, points_orig, lables_points, only_cls_tokens = False, noaug = False):
        if self.enc_arch == '3detr':
            pre_enc_xyz, group_input_tokens, pre_enc_inds = self.preencoder(center)
            group_input_tokens = group_input_tokens.permute(0, 2, 1)
            center = pre_enc_xyz
        else:
            group_input_tokens = self.preencoder(neighborhood)
        B, G, _ = center.shape
        mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
        if not noaug:
            if type(self.mask_ratio) is list:
                assert len(self.mask_ratio) == 2
                mask_ratio = random.uniform(*self.mask_ratio)
                n_mask = int(mask_ratio * G)
            elif self.mask_ratio > 0:
                n_mask = int(self.mask_ratio * G)
            perm = torch.randperm(G)[:n_mask]
            mask[:, perm] = True
        else:
            n_mask = 0
        n_unmask = G - n_mask

        masked_input_tokens = group_input_tokens[~mask].view(B, n_unmask, -1)
        masked_centers = center[~mask].view(B, n_unmask, -1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)

        pos = self.pos_embed(masked_centers)

        if self.enc_arch == '3detr':
            x = self.blocks(masked_input_tokens.transpose(0, 1), pos=pos.transpose(0, 1))[1].transpose(0, 1)

            if only_cls_tokens:
                return self.cls_head(torch.mean(x, dim=1))
        else:
            x = torch.cat((cls_tokens, masked_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)

            x = self.blocks(x, pos)
            x = self.norm(x)

            if only_cls_tokens:
                return self.cls_head(x[:, 0])

        query_points, query_labels = self._generate_query_xyz(points_orig, center, lables_points, mode=self.dec_query_mode)

        query_pos = self.pos_embed(query_points)
        query_tensor = torch.zeros_like(query_pos)
        dec_outputs = self.decoder(query_tensor, query_pos, x, pos) # q = query_tensor + query_pos, k = x + pos, v = x
        query_preds = self.bin_cls_head(dec_outputs).transpose(1, 2)
        
        # For 3DETR encoder, neighborhood is relative coordinates - restore absolute coordinates with pre_enc_xyz
        if self.enc_arch == '3detr':
            masked_centers = pre_enc_xyz[~mask].view(B, n_unmask, 3)  # [B, n_unmask, 3]
        else:
            masked_centers = center[~mask].view(B, n_unmask, 3)       # [B, n_unmask, 3]

        # Get unmasked neighborhood (relative coordinates) and restore absolute coordinates
        masked_neighborhood = neighborhood[~mask].view(B, n_unmask, self.group_size, 3)  # [B, n_unmask, group_size, 3]
        masked_points = masked_neighborhood + masked_centers.unsqueeze(2)  # Broadcast addition [B, n_unmask, group_size, 3]
        masked_points = masked_points.view(B, -1, 3)  # Reshape to [B, n_unmask*group_size, 3]
        
        return self.cls_head(x[:, 0]), query_preds, query_labels, query_points, masked_points


class MaskPoint(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[MaskPoint] build MaskPoint...', logger ='MaskPoint')
        self.config = config
        self.m = config.m
        self.T = config.T
        self.K = config.K
        
        self.transformer_q = MaskPointTransformer(config)
        self.transformer_k = MaskPointTransformer(config)
        for param_q, param_k in zip(self.transformer_q.parameters(), self.transformer_k.parameters()):
            param_k.data.copy_(param_q.data)  # Initialize key encoder with query encoder weights
            param_k.requires_grad = False  # Disable gradient update for key encoder
        self.use_moco_loss = config.transformer_config.use_moco_loss
        self.moco_loss_weight = config.transformer_config.moco_loss_weight
        self.query_loss_weight = config.transformer_config.query_loss_weight
        self.use_sigmoid = config.transformer_config.use_sigmoid
        self.use_focal_loss = config.transformer_config.use_focal_loss
        if self.use_focal_loss:
            self.focal_loss_alpha = config.transformer_config.focal_loss_alpha
            self.focal_loss_gamma = config.transformer_config.focal_loss_gamma

        self.group_size = config.transformer_config.group_size
        self.num_group = config.transformer_config.num_group

        print_log(f'[MaskPoint Group] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='MaskPoint')
        self.enc_arch = config.transformer_config.get('enc_arch', 'PointViT')
        self.group_divider = (DummyGroup if self.enc_arch == '3detr' else Group)(num_group = self.num_group, group_size = self.group_size)

        # Create MoCo queue
        self.register_buffer("queue", torch.randn(self.transformer_q.cls_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_batch = nn.CrossEntropyLoss(reduction='none')

        # Build loss functions
        self.build_loss_func()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.transformer_q.parameters(), self.transformer_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Dequeue old keys and enqueue new keys into MoCo queue
        """
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def build_loss_func(self):
        # Build loss functions based on configuration
        if self.use_sigmoid:
            self.loss_bce_batch = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_ce = nn.CrossEntropyLoss(ignore_index=-1)
            self.loss_ce_batch = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    def forward_eval(self, pts, lables_points):
        # Forward pass for evaluation phase
        with torch.no_grad():
            neighborhood, center = self.group_divider(pts)
            cls_feature, query_preds, query_labels = self.transformer_q(neighborhood, center, pts, lables_points, only_cls_tokens = False, noaug = True)
            # Calculate accuracy
            query_preds = query_preds.transpose(1, 2).contiguous().to(pts.device)
            query_labels = query_labels.to(pts.device)

            # Convert -1 labels to 1
            query_labels[query_labels == -1] = 1

            # Get predicted classes
            predicted_classes = torch.argmax(query_preds, dim=-1)

            # Calculate number of correct predictions
            correct_predictions = (predicted_classes == query_labels).sum().item()

            # Calculate accuracy
            total_samples = query_labels.numel()
            acc = correct_predictions / total_samples if total_samples > 0 else 0
            acc = torch.tensor(acc).cuda()
            
            nonzero_count = torch.count_nonzero(predicted_classes)
            total_count = predicted_classes.numel()
            zero_count = total_count - nonzero_count
            print('0 of predicted_classes:', zero_count)
            
            return cls_feature , acc

    def loss_focal_bce(self, pred, target):
        # Calculate focal BCE loss
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.focal_loss_alpha * target + (1 - self.focal_loss_alpha) * (1 - target)) * pt.pow(self.focal_loss_gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        return loss

    def loss_bce(self, preds, labels, reduction='mean'):
        # Calculate BCE loss (with focal loss support)
        loss_labels = labels.clone()
        loss_labels[labels == -1] = 1
        loss_labels_one_hot = F.one_hot(loss_labels, num_classes=2)
        preds = preds.transpose(1, 2).contiguous()

        if self.use_focal_loss:
            loss = self.loss_focal_bce(preds, loss_labels_one_hot)
        else:
            loss = self.loss_bce_batch(preds, loss_labels_one_hot.float())
            
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
            
        return loss

    def forward(self, pts, lables_points, noaug = False, return_acc_=False,**kwargs):
        if noaug:
            return self.forward_eval(pts, lables_points)
        else:
            self._momentum_update_key_encoder()

            neighborhood, center = self.group_divider(pts)
            q_cls_feature, query_preds, query_labels = self.transformer_q(neighborhood, center, pts, lables_points)
            q_cls_feature = F.normalize(q_cls_feature, dim=1)

            if self.use_moco_loss:
                with torch.no_grad():
                    k_cls_feature = self.transformer_k(neighborhood, center, pts, lables_points, only_cls_tokens = True)
                    k_cls_feature = F.normalize(k_cls_feature, dim=1)
                l_pos = torch.einsum('nc, nc->n', [q_cls_feature, k_cls_feature]).unsqueeze(-1)
                l_neg = torch.einsum('nc, ck->nk', [q_cls_feature, self.queue.clone().detach()])
                ce_logits = torch.cat([l_pos, l_neg], dim=1) / self.T
                labels = torch.zeros(l_pos.shape[0], dtype=torch.long).to(pts.device)
                moco_loss = self.loss_ce(ce_logits, labels)
                moco_loss = self.moco_loss_weight * moco_loss
            else:
                moco_loss = torch.tensor(0.).to(pts.device)
                
            if self.use_moco_loss:
                self._dequeue_and_enqueue(k_cls_feature)

            if self.use_sigmoid:
                recon_loss = self.loss_bce(query_preds, query_labels)
            else:
                recon_loss = self.loss_ce(query_preds, query_labels)
            recon_loss = self.query_loss_weight * recon_loss

            return recon_loss, moco_loss

def compute_chamfer_distance(pred_pc, labels_pc):
    """
    Calculate Chamfer Distance between two point clouds
    
    Parameters:
    pred_pc: Predicted point cloud
    labels_pc: Ground truth point cloud
    
    Returns:
    chamfer_distance: Chamfer Distance value
    """
    # Ensure inputs are Open3D point cloud objects
    if not isinstance(pred_pc, o3d.geometry.PointCloud):
        pred_pc = o3d.geometry.PointCloud(pred_pc)
    
    if not isinstance(labels_pc, o3d.geometry.PointCloud):
        labels_pc = o3d.geometry.PointCloud(labels_pc)
    
    # Build KD-Tree for nearest neighbor search
    labels_tree = o3d.geometry.KDTreeFlann(labels_pc)
    pred_tree = o3d.geometry.KDTreeFlann(pred_pc)
    
    # Get point cloud points
    pred_points = np.asarray(pred_pc.points)
    labels_points = np.asarray(labels_pc.points)
    
    # Calculate distance from predicted to ground truth point cloud
    pred_to_labels = 0.0
    for point in pred_points:
        [k, idx, _] = labels_tree.search_knn_vector_3d(point, 1)
        pred_to_labels += np.sum((point - labels_points[idx[0]]) ** 2)
    pred_to_labels /= len(pred_points)
    
    # Calculate distance from ground truth to predicted point cloud
    labels_to_pred = 0.0
    for point in labels_points:
        [k, idx, _] = pred_tree.search_knn_vector_3d(point, 1)
        labels_to_pred += np.sum((point - pred_points[idx[0]]) ** 2)
    labels_to_pred /= len(labels_points)
    
    # Chamfer Distance is the average of two directional distances
    chamfer_distance = pred_to_labels + labels_to_pred
    
    return chamfer_distance

def mirror_point_cloud(point_cloud, normal, point_on_plane):
    """
    Perform mirror reflection on point cloud
    
    Parameters:
    point_cloud: Open3D point cloud object
    normal: Normal vector of mirror plane [a, b, c]
    point_on_plane: A point on the mirror plane [x0, y0, z0]
    
    Returns:
    Mirrored point cloud object
    """
    # Normalize normal vector
    normal = np.array(normal) / np.linalg.norm(normal)

    # Build reflection transformation matrix
    # Reflection matrix formula: I - 2 * n * n^T (n is unit normal vector)
    n = normal.reshape(3, 1)
    reflection_matrix = np.eye(3) - 2 * np.dot(n, n.T)

    # Translation vector (from origin to point on plane)
    translation = np.array(point_on_plane)

    # Create deep copy of point cloud
    mirrored_cloud = copy.deepcopy(point_cloud)
    points = np.asarray(mirrored_cloud.points)

    # Translate to origin → apply reflection → translate back
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
    Normalize point cloud, and optionally apply the same normalization to a second point cloud
    
    Parameters:
    pc: First point cloud (Open3D point cloud object or point array)
    pc2: Second point cloud (optional)
    
    Returns:
    If only pc is provided: Normalized pc point cloud
    If both pc and pc2 are provided: Normalized pc and pc2 point clouds
    """
    # Process first point cloud
    pc_np = np.asarray(pc)
    centroid = np.mean(pc_np, axis=0)
    pc_np = pc_np - centroid
    m = np.max(np.sqrt(np.sum(pc_np ** 2, axis=1)))
    pc_np = pc_np / m
    pc_normalized = o3d.utility.Vector3dVector(pc_np)

    # Return only normalized first point cloud if no second point cloud
    if pc2 is None:
        return pc_normalized

    # Process second point cloud with same normalization parameters
    pc2_np = np.asarray(pc2)
    pc2_np = pc2_np - centroid  # Use same centroid
    pc2_np = pc2_np / m        # Use same scale factor
    pc2_normalized = o3d.utility.Vector3dVector(pc2_np)

    return pc_normalized, pc2_normalized

def process(pcd, pcd2):
    """
    Process point cloud (mirror reflection + rotation + normalization)
    """
    normal = [0, 1, 0]  # Normal vector along X-axis
    point_on_plane = [0, 0, 0]  # Origin point on plane
    # Perform mirror reflection
    pcd = mirror_point_cloud(pcd, normal, point_on_plane)
    pcd2 = mirror_point_cloud(pcd2, normal, point_on_plane)
    
    # Create rotation matrix: rotate 90 degrees around y-axis
    R_y = pcd.get_rotation_matrix_from_xyz((0, math.pi / 2, 0))
    # Create rotation matrix: rotate 90 degrees around x-axis
    R_x = pcd.get_rotation_matrix_from_xyz((-math.pi / 2, 0, 0))
    # Combine rotation matrices (y-axis first, then x-axis)
    R = np.dot(R_x, R_y)

    # Apply rotation
    pcd.rotate(R, center=(0, 0, 0))
    pcd2.rotate(R, center=(0, 0, 0))
    
    # Normalize point clouds
    pcd.points, pcd2.points = pc_normalize(pcd.points, pcd2.points)
    
    return pcd, pcd2

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

# Custom Dataset class for PCD files
class PCDDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = []
        self.label_files = []
        
        # Collect data and label files
        for file in os.listdir(data_dir):
            if file.endswith('.pcd'):
                parts = file.split('_')
                index_str = parts[-1].replace('.pcd', '')
                index = int(index_str)
                # Label files are those with index 1, 11, 21, 31
                if index in [1, 11, 21, 31]:
                    self.label_files.append(os.path.join(data_dir, file))
                    self.data_files.append(os.path.join(data_dir, file))
                else:
                    self.data_files.append(os.path.join(data_dir, file))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        # Extract object name, scale and index from filename
        parts = data_file.split('_')
        object_name = "_".join(parts[:-2])
        scale = parts[-2].replace('scale', '')
        index_str = parts[-1].replace('.pcd', '')
        index = int(index_str)

        # Determine label index based on data index
        if 1 <= index <= 10:
            label_index = 1
        elif 11 <= index <= 20:
            label_index = 11
        elif 21 <= index <= 30:
            label_index = 21
        else:
            label_index = 31

        # Filter label file matching object name, scale and label index
        label_file = [f for f in self.label_files if object_name in f and f'scale{scale}' in f and f'_{label_index}.pcd' in f][0]

        # Read point clouds
        data_pcd = o3d.io.read_point_cloud(data_file)
        label_pcd = o3d.io.read_point_cloud(label_file)
        
        # Process point clouds
        data_pcd, label_pcd = process(data_pcd, label_pcd)
        print(data_file)

        # Extract point coordinates
        data_points = torch.tensor(np.array(data_pcd.points), dtype=torch.float32)
        label_points = torch.tensor(np.array(label_pcd.points), dtype=torch.float32)
        
        # Extract color information for label filtering
        label_colors = torch.tensor(np.array(label_pcd.colors), dtype=torch.float32)

        # Filter label points with color [1, 1, 0]
        target_color = torch.tensor([1, 1, 0], dtype=torch.float32)
        mask = torch.all(label_colors == target_color, dim=1)
        label = label_points[mask]
        
        # Resample data to 1600 points using FPS
        if data_points.shape[0] > 1600:
            data = farthest_point_sample(data_points, 1600)
        elif data_points.shape[0] < 1600:
            padding = torch.zeros((1600 - data_points.shape[0], data_points.shape[1]), dtype=torch.float32)
            data = torch.cat([data_points, padding], dim=0)
        else:
            data = data_points

        return data, label


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MaskPointTransformer Evaluation with Command Line Args')
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
    transformer_q = MaskPoint(config).transformer_q

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
            q_cls_feature, query_preds, query_labels, query_points, masked_points = transformer_q(
                neighborhood, center, data, label
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
        avg_cd = sum(all_cd_values) / len(all_cd_values)
        print(f'\nAverage Chamfer Distance: {avg_cd:.6f}')
    else:
        print('No valid CD values were collected.')