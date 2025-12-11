import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from utils.logger import *
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from torchvision.models import resnet18
from timm.models.vision_transformer import PatchEmbed, Block
import torchvision
import Diffusion_modules
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformer import TransformerEncoder, TransformerDecoder, Group, DummyGroup, Encoder
from termcolor import cprint
from typing import Optional, Dict, Tuple, Union, List, Type
import yaml  
from pathlib import Path 

def min_max_normalize_robot_state(state, config):
    """
    Normalize the robot state vector using min-max normalization, and clip the normalized values to the range [0, 1].
    The quaternion part is not normalized.

    """
    batch_size, obs_t, _ = state.shape
    normalized_state = torch.zeros_like(state)
    
    # Get parameters from config
    pos_mins = torch.tensor(config['pos_mins'], device=state.device)
    pos_maxs = torch.tensor(config['pos_maxs'], device=state.device)
    rpy_mins = torch.tensor(config['rpy_mins'], device=state.device)
    rpy_maxs = torch.tensor(config['rpy_maxs'], device=state.device)
    joint_min = torch.tensor(config['joint_min'], device=state.device)
    joint_max = torch.tensor(config['joint_max'], device=state.device)

    
    # Normalize end-effector position (first 3 values) and clip to [0, 1]
    pos_normalized = (state[:, :, :3] - pos_mins) / (pos_maxs - pos_mins)
    normalized_state[:, :, :3] = torch.clamp(pos_normalized, min=0, max=1)
    
    # Normalize Euler angles
    rpy_normalized = (state[:, :, 3:6] - rpy_mins) / (rpy_maxs - rpy_mins)
    normalized_state[:, :, 3:6] = torch.clamp(rpy_normalized, min=0, max=1)

    # Normalize joint positions (last 16 values) and clip to [0, 1]
    joint_normalized = (state[:, :, 6:] - joint_min) / (joint_max - joint_min)
    normalized_state[:, :, 6:] = torch.clamp(joint_normalized, min=0, max=1)
    
    return normalized_state

def min_max_denormalize_robot_state(normalized_state, config):
    """
    Denormalize the robot state vector from min-max normalized values to original scale,
    supports multi-dimensional input. The quaternion part is not denormalized.
    
    Args:
        normalized_state: Normalized state tensor with shape [batch_size, obs_t, 22]
        config: Configuration dictionary containing:
            - pos_mins: Minimum values of position, shape [3]
            - pos_maxs: Maximum values of position, shape [3]
            - rpy_mins: Minimum values of roll-pitch-yaw, shape [3]
            - rpy_maxs: Maximum values of roll-pitch-yaw, shape [3]
            - joint_min: Minimum values of joint positions, shape [16]
            - joint_max: Maximum values of joint positions, shape [16]
    
    Returns:
        Denormalized original state tensor with the same shape as input
    """
    batch_size, obs_t, _ = normalized_state.shape
    denormalized_state = torch.zeros_like(normalized_state)
    
    # Get parameters from config
    pos_mins = torch.tensor(config['pos_mins'], device=normalized_state.device)
    pos_maxs = torch.tensor(config['pos_maxs'], device=normalized_state.device)
    rpy_mins = torch.tensor(config['rpy_mins'], device=normalized_state.device)
    rpy_maxs = torch.tensor(config['rpy_maxs'], device=normalized_state.device)
    joint_min = torch.tensor(config['joint_min'], device=normalized_state.device)
    joint_max = torch.tensor(config['joint_max'], device=normalized_state.device)
    
    # Denormalize end-effector position
    denormalized_state[:, :, :3] = normalized_state[:, :, :3] * (pos_maxs - pos_mins) + pos_mins
    
    # Denormalize Euler angles
    denormalized_state[:, :, 3:6] = normalized_state[:, :, 3:6] * (rpy_maxs - rpy_mins) + rpy_mins

    # Denormalize joint positions
    denormalized_state[:, :, 6:] = normalized_state[:, :, 6:] * (joint_max - joint_min) + joint_min

    return denormalized_state



def farthest_point_sample(data, npoints):
    """
    Farthest Point Sampling (FPS) algorithm for point cloud downsampling

    """
    N, D = data.shape  # N: number of points, D: dimension
    xyz = data[:, :3]  # Only use 3D coordinates
    centroids = torch.zeros(size=(npoints,))  # Indices of final sampled points
    distance = torch.ones(size=(N,)) * 1e10  # Distance list, initialized to large value
    farthest = torch.randint(low=0, high=N, size=(1,))  # Randomly select initial point index
    for i in range(npoints):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = ((xyz - centroid)**2).sum(dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, dim=-1)
    data = data[centroids.type(torch.long)]
    return data


# New: Config loading function
def load_config(config_path):
    """Load network configuration parameters from yaml file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



# Mini PointNet Module
class MiniPointNet(nn.Module):
    def __init__(self, Trans_dim, target_points):
        super().__init__()
        self.encoder_channel = Trans_dim
        self.target_points = target_points  # Target number of points
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    
    def forward(self, point_groups):
        '''
            point_groups : Input point groups with shape [B, N, 3] (B: batch size, N: number of points)
            -----------------
            output : Global feature with shape [B, C] (C: feature dimension)
        '''
        bs, n, _ = point_groups.shape
        
        # Handle point count
        if n < self.target_points:
            # Pad with zeros if insufficient points
            padding = torch.zeros(bs, self.target_points - n, 3, device=point_groups.device)
            point_groups = torch.cat([point_groups, padding], dim=1)
        elif n > self.target_points:
            # Use FPS sampling if too many points
            point_groups = self.fps(point_groups, self.target_points)
        
        # Encoder forward pass
        feature = self.first_conv(point_groups.transpose(2, 1))  # [B, 256, n]
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # [B, 256, 1]
        feature = torch.cat([feature_global.expand(-1, -1, self.target_points), feature], dim=1)  # [B, 512, n]
        feature = self.second_conv(feature)  # [B, Trans_dim, n]
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # [B, Trans_dim]
        return feature_global.reshape(bs, self.encoder_channel).unsqueeze(dim=1)


# Point Encoder Module
class PointEncoder(nn.Module):
    def __init__(self, Trans_dim, Depth, Num_heads, Group_size, Num_group, Drop_path_rate=0.1):
        super().__init__()

        self.trans_dim = Trans_dim
        self.depth = Depth
        self.drop_path_rate = Drop_path_rate
        self.num_heads = Num_heads

        self.group_size = Group_size
        self.num_group = Num_group
        self.encoder_dims = Trans_dim

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successfully loaded checkpoint from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_tokens = self.encoder(neighborhood)  # [B, G, D]

        pos = self.pos_embed(center)

        # Transformer forward pass
        x = self.blocks(group_tokens, pos)
        x = self.norm(x)
        return x

# Point Encoder Module (variant with cls token)
class PointEncoder2(nn.Module):
    def __init__(self, Trans_dim, Depth, Num_heads, Group_size, Num_group, Drop_path_rate=0.1):
        super().__init__()

        self.trans_dim = Trans_dim
        self.depth = Depth
        self.drop_path_rate = Drop_path_rate
        self.num_heads = Num_heads

        self.group_size = Group_size
        self.num_group = Num_group
        self.encoder_dims = Trans_dim

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.build_loss_func()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successfully loaded checkpoint from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, pts):
        """
        Forward pass for PointEncoder2 (with cls token)
        
        Args:
            pts: Input point cloud with shape [B, N, 3]
        
        Returns:
            Encoded features with shape [B, G+1, D] (including cls token)
        """
        neighborhood, center = self.group_divider(pts)
        group_tokens = self.encoder(neighborhood)  # [B, G, D]
        
        # Prepare cls token
        cls_tokens = self.cls_token.expand(group_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_tokens.size(0), -1, -1)
        
        pos = self.pos_embed(center)
        
        # Final input (cls token + group tokens)
        x = torch.cat((cls_tokens, group_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        
        # Transformer forward pass
        x = self.blocks(x, pos)
        x = self.norm(x)
        return x
        
class MaskPointTransformer(nn.Module):
    def __init__(self, encoder_dims, Trans_dim, Depth, Num_heads, Group_size, Num_group, cls_dim=512, drop_path_rate=0.1):
        super().__init__()
        # Define encoder parameters
        self.num_group = Num_group
        self.group_size = Group_size
        self.encoder_dims = encoder_dims
        
        # Define transformer parameters
        self.trans_dim = Trans_dim
        self.depth = Depth
        self.drop_path_rate = drop_path_rate
        self.cls_dim = cls_dim
        self.num_heads = Num_heads
        
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        
        # Bridge encoder and transformer (dimension reduction if needed)
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        # Define learnable cls token and position embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        # Position embedding for each patch
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        # Define transformer blocks with linear drop path rate
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(self.trans_dim, self.cls_dim),
            nn.GELU(),
            nn.Linear(self.cls_dim, self.cls_dim)
        )

        # Layer normalization
        self.norm = nn.LayerNorm(self.trans_dim)
        
        # Initialize learnable tokens
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

        self.apply(self._init_weights)
        self.access_count = 0
        self.fc = nn.Linear(self.cls_dim, self.trans_dim)

    def _init_weights(self, m):
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

    def preencoder(self, neighborhood):
        """Pre-encoder for point groups"""
        group_input_tokens = self.encoder(neighborhood)  # [B, G, D]
        group_input_tokens = self.reduce_dim(group_input_tokens)
        return group_input_tokens

    def forward(self, points):
       neighborhood, center = self.group_divider(points)
        group_input_tokens = self.preencoder(neighborhood)
        B, G, _ = center.shape

        # Expand cls token to batch size
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)

        # Position embedding for group centers
        pos = self.pos_embed(center)

        # Concatenate cls token with group tokens
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        # Transformer forward pass
        x = self.blocks(x, pos)
        x = self.norm(x)
        
        # Extract cls token output
        cls_output = self.cls_head(x[:, 0])
        cls_output = self.fc(cls_output)
        
        return cls_output, x

# MLP for robot state encoding (DP3 style)
def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a Multi-Layer Perceptron (MLP) consisting of fully-connected layers
    followed by activation functions (except for the final layer).
    
    :param input_dim: Dimension of the input vector
    :param output_dim: Dimension of the output vector (0 for no final layer)
    :param net_arch: Architecture of the neural network - list of integers
        representing the number of units per hidden layer
    :param activation_fn: Activation function to use after each hidden layer
    :param squash_output: Whether to apply Tanh activation to the final output
    :return: List of nn.Module layers composing the MLP
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

class PointEncoderXYZRGB(nn.Module):
    """Encoder for Point Cloud (XYZ + RGB features)"""
    def __init__(self,
                 in_channels: int = 6,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs
                 ):
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
        # Final projection layer with normalization
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):

        x = self.mlp(x)
        x = torch.max(x, 1)[0]  # Global max pooling
        x = self.final_projection(x)
        return x


# Force Encoder Module (tactile force sensing)
class ForceEncoder1(nn.Module):
    def __init__(self, Trans_dim, kernel_size=3, stride=1, padding=1):
        super(ForceEncoder1, self).__init__()

        # Convolutional layers for feature extraction per channel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=kernel_size, stride=stride, padding=padding)
        self.BN1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=kernel_size, stride=stride, padding=padding)
        self.BN2 = nn.BatchNorm2d(12)
        
        # Global average pooling (reduce to fixed size)
        self.APooling = nn.AdaptiveAvgPool2d((3, 3))
        
        # Fully connected layers for dimension reduction
        self.fc1 = nn.Linear(108, 64)
        self.fc2 = nn.Linear(64, Trans_dim)

    def block(self, x):
        """Feature extraction block for single force channel"""
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.BN1(x)
        x = self.conv2(x)
        
        # Global average pooling
        x = self.APooling(x)
        
        # Reshape for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def forward(self, x):

        channel_features = []
        # Process each force channel independently
        for i in range(x.size(1)):
            # Extract single channel
            single_channel = x[:, i:i + 1, :, :]
            # Extract features for this channel
            feature = self.block(single_channel)
            channel_features.append(feature)
        
        # Concatenate features from all channels
        final_features = torch.stack(channel_features, dim=1)
        return final_features


# Multi-Head Attention Module
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.MultiHead_Attention = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)

    def forward(self, q, k, v):

        output, attn_output_weights = self.MultiHead_Attention(q, k, v)
        return output, attn_output_weights


# Feed Forward Network Layer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Self-Attention Block (with Multi-Head Attention, FFN, and LayerNorm)
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim):
        super(SelfAttentionBlock, self).__init__()
        self.multihead_attn = MultiHeadAttention(dim, num_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):

        # Multi-head self-attention with residual connection
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed forward network with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x


# Cross-Attention Block (with Multi-Head Attention, FFN, and LayerNorm)
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim):
        super(CrossAttentionBlock, self).__init__()
        self.multihead_attn = MultiHeadAttention(dim, num_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.norm1 = nn.LayerNorm(dim)

    def forward(self, q, k, v):

        # Multi-head cross-attention with residual connection
        attn_output, _ = self.multihead_attn(q, k, v)
        q = self.norm1(q + attn_output)

        # Feed forward network with residual connection
        ff_output = self.feed_forward(q)
        q = self.norm1(q + ff_output)
        return q


# Action Prediction Head (MLP)
class ActionHead1(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(ActionHead1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Complete Fusion Network Architecture
class FusionNet1(nn.Module):
    def __init__(self, config):
        super(FusionNet1, self).__init__()

        # Network configuration parameters
        self.Encoder_dims = config['fusion_net']['Encoder_dims']
        self.Trans_dim = config['fusion_net']['Trans_dim']
        self.target_point_num = config['fusion_net']['target_point_num']
        self.point_encoder_Depth = config['fusion_net']['point_encoder_Depth']
        self.PointEncoder_Num_heads = config['fusion_net']['PointEncoder_Num_heads']
        self.cls_dim = config['fusion_net']['cls_dim']
        self.drop_path_rate = config['fusion_net']['drop_path_rate']
        self.Group_size = config['fusion_net']['Group_size']
        self.Num_group = config['fusion_net']['Num_group']
        self.state_mlp_size = config['fusion_net']['state_mlp_size']
        self.state_mlp_activation_fn = nn.ReLU
        self.state_shape = config['fusion_net']['state_shape']
        self.num_self_blocks = config['fusion_net']['num_self_blocks']
        self.num_cross_blocks = config['fusion_net']['num_cross_blocks']
        self.atten_num_heads = config['fusion_net']['atten_num_heads']
        self.atten_ff_hidden_dim = config['fusion_net']['atten_ff_hidden_dim']
        self.arm_degree = config['fusion_net']['arm_degree']
        self.hand_degree = config['fusion_net']['hand_degree']
        self.pred_time = config['fusion_net']['diffusion']['pred_time']
        self.num_diffusion_iters = config['fusion_net']['diffusion']['num_train_timesteps']
        self.obs_t = config['obs_t']  # Observation time steps
        self.num_inference_steps = config['num_inference_steps']  # Inference steps for diffusion
        self.config = config
        
        # Encoder modules
        self.mini_pointnet = MiniPointNet(self.Trans_dim, self.target_point_num)
        self.point_encoder = MaskPointTransformer(self.Encoder_dims, self.Trans_dim, self.point_encoder_Depth,
                                                self.PointEncoder_Num_heads, self.Group_size, self.Num_group, self.cls_dim)
        self.overall_point_encoder = PointEncoderXYZRGB(in_channels=6, out_channels=self.Trans_dim, use_layernorm=True,
                                                        final_norm='layernorm', use_projection=True)
        self.force_encoder = ForceEncoder1(self.Trans_dim)

        # Self-Attention Blocks
        self.self_attention_blocks = nn.ModuleList([
            SelfAttentionBlock(self.Trans_dim, self.atten_num_heads, self.atten_ff_hidden_dim) 
            for _ in range(self.num_self_blocks)
        ])

        # Cross-Attention Blocks
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(self.Trans_dim, self.atten_num_heads, self.atten_ff_hidden_dim) 
            for _ in range(self.num_cross_blocks)
        ])

        # Robot state MLP
        if len(self.state_mlp_size) == 0:
            raise RuntimeError(f"State MLP size is empty")
        elif len(self.state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = self.state_mlp_size[:-1]
        state_output_dim = self.state_mlp_size[-1]
        self.state_mlp = nn.Sequential(*create_mlp(
            self.state_shape * self.obs_t, 
            state_output_dim, 
            net_arch, 
            self.state_mlp_activation_fn
        ))

        # Diffusion modules (single or dual diffusion for arm/hand)
        if config['use_one_diffusion']:
            # Single diffusion for combined arm+hand action
            self.action_diffusion = Diffusion_modules.ConditionalUnet1D(
                input_dim=self.arm_degree + self.hand_degree,
                global_cond_dim=self.Trans_dim * 3 * self.obs_t + state_output_dim
            )
            self.noise_scheduler_action = DDIMScheduler(
                num_train_timesteps=config['fusion_net']['diffusion']['num_train_timesteps'],
                beta_schedule=config['fusion_net']['diffusion']['beta_schedule'],
                clip_sample=True,
                prediction_type=config['fusion_net']['diffusion']['pred_type']
            )
        else:
            if config['use_two_diffusion']:
                # Dual diffusion (separate for arm and hand)
                self.arm_diffusion = Diffusion_modules.ConditionalUnet1D(
                    input_dim=self.arm_degree,
                    global_cond_dim=self.Trans_dim * (1+1+1) * self.obs_t + state_output_dim
                )
                self.hand_diffusion = Diffusion_modules.ConditionalUnet1D(
                    input_dim=self.hand_degree,
                    global_cond_dim=self.Trans_dim * 3 * self.obs_t + state_output_dim
                )

                self.noise_scheduler_arm = DDIMScheduler(
                    num_train_timesteps=config['fusion_net']['diffusion']['num_train_timesteps'],
                    beta_schedule=config['fusion_net']['diffusion']['beta_schedule'],
                    clip_sample=True,
                    prediction_type=config['fusion_net']['diffusion']['pred_type']
                )
                self.noise_scheduler_hand = DDIMScheduler(
                    num_train_timesteps=config['fusion_net']['diffusion']['num_train_timesteps'],
                    beta_schedule=config['fusion_net']['diffusion']['beta_schedule'],
                    clip_sample=True,
                    prediction_type=config['fusion_net']['diffusion']['pred_type']
                )
            else:
                # Single diffusion + MLP head for the other part
                if config['use_arm_diffusion'] and not config['use_hand_diffusion']:
                    self.arm_diffusion = Diffusion_modules.ConditionalUnet1D(
                        input_dim=self.arm_degree,
                        global_cond_dim=self.Trans_dim * (1+1+1) * self.obs_t + state_output_dim
                    )
                    self.noise_scheduler_arm = DDIMScheduler(
                        num_train_timesteps=config['fusion_net']['diffusion']['num_train_timesteps'],
                        beta_schedule=config['fusion_net']['diffusion']['beta_schedule'],
                        clip_sample=True,
                        prediction_type=config['fusion_net']['diffusion']['pred_type']
                    )
                    self.Hand_head = ActionHead1(
                        input_dim=3 * self.Trans_dim * self.obs_t + state_output_dim,
                        output_dim=self.hand_degree * self.pred_time
                    )
                
                if config['use_hand_diffusion'] and not config['use_arm_diffusion']:
                    self.hand_diffusion = Diffusion_modules.ConditionalUnet1D(
                        input_dim=self.hand_degree,
                        global_cond_dim=self.Trans_dim * 3 * self.obs_t + state_output_dim
                    )
                    self.noise_scheduler_hand = DDIMScheduler(
                        num_train_timesteps=config['fusion_net']['diffusion']['num_train_timesteps'],
                        beta_schedule=config['fusion_net']['diffusion']['beta_schedule'],
                        clip_sample=True,
                        prediction_type=config['fusion_net']['diffusion']['pred_type']
                    )
                    self.Arm_head = ActionHead1(
                        input_dim=(1+1+1) * self.Trans_dim * self.obs_t + state_output_dim,
                        output_dim=self.arm_degree * self.pred_time
                    )

    def forward(self, mini_pointnet_input, point_encoder_input, overall_point_encoder_input, 
                force_encoder_input, state_input, rel_actions):

        Batch_size = mini_pointnet_input.shape[0]
        
        # Reshape inputs for time sequence processing
        mini_pointnet_input = mini_pointnet_input.view(Batch_size * self.obs_t, -1, 3)
        point_encoder_input = point_encoder_input.view(Batch_size * self.obs_t, -1, 3)
        overall_point_encoder_input = overall_point_encoder_input.view(Batch_size * self.obs_t, -1, 6)
        B, T, H, W, C = force_encoder_input.shape
        force_encoder_input = force_encoder_input.view(Batch_size * self.obs_t, 3, H, W)
        
        # Normalize state and action inputs
        state_input = min_max_normalize_robot_state(state_input, self.config)
        rel_actions = min_max_normalize_robot_state(rel_actions, self.config)
        state_input = state_input.view(Batch_size, -1)
        
        # Zero out tactile inputs if disabled in config
        if self.config['no_tactile']:
            mini_pointnet_input = torch.zeros_like(mini_pointnet_input)
            force_encoder_input = torch.zeros_like(force_encoder_input)

        # Encode different input modalities
        mini_pointnet_output = self.mini_pointnet(mini_pointnet_input)
        point_encoder_cls_output, point_encoder_output = self.point_encoder(point_encoder_input)
        overall_point_encoder_output = self.overall_point_encoder(overall_point_encoder_input)
        force_encoder_output = self.force_encoder(force_encoder_input)
        state_output = self.state_mlp(state_input)

        # Reshape outputs back to batch-time format
        mini_pointnet_output = mini_pointnet_output.view(Batch_size, self.obs_t, -1, self.Trans_dim)
        point_encoder_cls_output = point_encoder_cls_output.view(Batch_size, self.obs_t, -1)
        overall_point_encoder_output = overall_point_encoder_output.view(Batch_size, self.obs_t, -1)
        force_encoder_output = force_encoder_output.view(Batch_size, self.obs_t, -1, self.Trans_dim)

        # Prepare attention inputs
        q = force_encoder_output
        q = q.view(Batch_size, self.obs_t * q.size(2), self.Trans_dim)
        
        # Build key tensor (point encoder + global point encoder for each time step)
        k_list = []
        for t in range(self.obs_t):
            k_t = torch.cat((
                point_encoder_cls_output[:, t].unsqueeze(1), 
                overall_point_encoder_output[:, t].unsqueeze(1)
            ), dim=1)  # [B, 2, Trans_dim]
            k_list.append(k_t)
        k = torch.cat(k_list, dim=1)  # [B, obs_t*2, Trans_dim]

        # Value tensor from mini pointnet
        v = mini_pointnet_output.view(Batch_size, self.obs_t * mini_pointnet_output.size(2), self.Trans_dim)
        concatenated_kv = torch.cat((k, v), dim=1)  # [B, obs_t*3, Trans_dim]

        # Attention blocks
        num_blocks = min(len(self.self_attention_blocks), len(self.cross_attention_blocks))
        for i in range(num_blocks):
            concatenated_kv = self.self_attention_blocks[i](concatenated_kv)
            q = self.cross_attention_blocks[i](q, concatenated_kv, concatenated_kv)

        # Flatten for diffusion input
        q = torch.flatten(q, start_dim=1)
        concatenated_kv = torch.flatten(concatenated_kv, start_dim=1)

        # Concatenate with state features
        q = torch.cat([q, state_output], dim=1)
        concatenated_kv = torch.cat([concatenated_kv, state_output], dim=1)

        # Diffusion forward pass
        if self.config['use_one_diffusion']:
            # Single diffusion for combined action
            noise = torch.randn(rel_actions.shape).cuda()
            timesteps = torch.randint(
                0, self.noise_scheduler_action.config.num_train_timesteps,
                (Batch_size,)).long().cuda()
            noisy_actions = self.noise_scheduler_action.add_noise(rel_actions, noise, timesteps)
            noise_pred = self.action_diffusion(noisy_actions, timesteps, global_cond=q)
            
            # Set target based on prediction type
            if self.noise_scheduler_action.config.prediction_type == 'epsilon':
                target = noise
            elif self.noise_scheduler_action.config.prediction_type == 'sample':
                target = rel_actions

            return noise_pred, target

        else:
            if self.config['use_two_diffusion']:
                # Split actions into arm and hand components
                rel_arm = rel_actions[:, :, :self.arm_degree]  # [B, T, arm_degree]
                rel_hand = rel_actions[:, :, self.arm_degree:]  # [B, T, hand_degree]
                
                # Sample noise
                noise = torch.randn(rel_actions.shape).cuda()
                noise_arm = noise[:, :, :self.arm_degree]
                noise_hand = noise[:, :, self.arm_degree:]
                
                # Sample timesteps
                timesteps = torch.randint(
                    0, self.noise_scheduler_arm.config.num_train_timesteps,
                    (Batch_size,)).long().cuda()
                timesteps_arm = timesteps
                timesteps_hand = timesteps

                # Add noise (forward diffusion)
                noisy_arm_actions = self.noise_scheduler_arm.add_noise(
                    rel_arm, noise_arm, timesteps_arm)
                noisy_hand_actions = self.noise_scheduler_hand.add_noise(
                    rel_hand, noise_hand, timesteps_hand)

                # Predict noise
                arm_noise_pred = self.arm_diffusion(noisy_arm_actions, timesteps_arm, global_cond=concatenated_kv)
                hand_noise_pred = self.hand_diffusion(noisy_hand_actions, timesteps_hand, global_cond=q)

                # Set targets based on prediction type
                if self.noise_scheduler_arm.config.prediction_type == 'epsilon':
                    target_arm = noise_arm
                elif self.noise_scheduler_arm.config.prediction_type == 'sample':
                    target_arm = rel_arm

                if self.noise_scheduler_hand.config.prediction_type == 'epsilon':
                    target_hand = noise_hand
                elif self.noise_scheduler_hand.config.prediction_type == 'sample':
                    target_hand = rel_hand
                
                return arm_noise_pred, target_arm, hand_noise_pred, target_hand
            
            else:
                if self.config['use_arm_diffusion'] and not self.config['use_hand_diffusion']:
                    # Arm diffusion + hand MLP prediction
                    rel_arm = rel_actions[:, :, :self.arm_degree]
                    rel_hand = rel_actions[:, :, self.arm_degree:]
                    
                    # Sample noise and timesteps
                    noise_arm = torch.randn(rel_arm.shape).cuda()
                    timesteps = torch.randint(
                        0, self.noise_scheduler_arm.config.num_train_timesteps,
                        (Batch_size,)).long().cuda()

                    # Forward diffusion
                    noisy_arm_actions = self.noise_scheduler_arm.add_noise(
                        rel_arm, noise_arm, timesteps)

                    # Predict noise
                    arm_noise_pred = self.arm_diffusion(noisy_arm_actions, timesteps, global_cond=concatenated_kv)
                    
                    # Predict hand action with MLP
                    action_hand = self.Hand_head(q).view(-1, self.pred_time, self.hand_degree)

                    # Set targets
                    if self.noise_scheduler_arm.config.prediction_type == 'epsilon':
                        target_arm = noise_arm
                        target_hand = rel_hand
                    elif self.noise_scheduler_arm.config.prediction_type == 'sample':
                        target_arm = rel_arm
                        target_hand = rel_hand
                    
                    return arm_noise_pred, target_arm, action_hand, target_hand

                if self.config['use_hand_diffusion'] and not self.config['use_arm_diffusion']:
                    # Hand diffusion + arm MLP prediction
                    rel_arm = rel_actions[:, :, :self.arm_degree]
                    rel_hand = rel_actions[:, :, self.arm_degree:]
                    
                    # Sample noise and timesteps
                    noise_hand = torch.randn(rel_hand.shape).cuda()
                    timesteps = torch.randint(
                        0, self.noise_scheduler_hand.config.num_train_timesteps,
                        (Batch_size,)).long().cuda()

                    # Forward diffusion
                    noisy_hand_actions = self.noise_scheduler_hand.add_noise(
                        rel_hand, noise_hand, timesteps)

                    # Predict noise
                    hand_noise_pred = self.hand_diffusion(noisy_hand_actions, timesteps, global_cond=q)
                    
                    # Predict arm action with MLP
                    action_arm = self.Arm_head(concatenated_kv).view(-1, self.pred_time, self.arm_degree)

                    # Set targets
                    if self.noise_scheduler_hand.config.prediction_type == 'epsilon':
                        target_arm = rel_arm
                        target_hand = noise_hand
                    elif self.noise_scheduler_hand.config.prediction_type == 'sample':
                        target_arm = rel_arm
                        target_hand = rel_hand

                    return action_arm, target_arm, hand_noise_pred, target_hand

    def infer(self, mini_pointnet_input, point_encoder_input, overall_point_encoder_input, 
              force_encoder_input, state_input):

        Batch_size = mini_pointnet_input.shape[0]
        
        # Reshape inputs
        mini_pointnet_input = mini_pointnet_input.view(Batch_size * self.obs_t, -1, 3)
        point_encoder_input = point_encoder_input.view(Batch_size * self.obs_t, -1, 3)
        overall_point_encoder_input = overall_point_encoder_input.view(Batch_size * self.obs_t, -1, 6)
        B, T, H, W, C = force_encoder_input.shape
        force_encoder_input = force_encoder_input.view(Batch_size * self.obs_t, 3, H, W)
        
        # Normalize state input
        state_input = min_max_normalize_robot_state(state_input, self.config)
        state_input = state_input.view(Batch_size, -1)
        
        # Zero out tactile inputs if disabled
        if self.config['no_tactile']:
            mini_pointnet_input = torch.zeros_like(mini_pointnet_input)
            force_encoder_input = torch.zeros_like(force_encoder_input)

        # Encode inputs
        mini_pointnet_output = self.mini_pointnet(mini_pointnet_input)
        point_encoder_cls_output, point_encoder_output = self.point_encoder(point_encoder_input)
        overall_point_encoder_output = self.overall_point_encoder(overall_point_encoder_input)
        force_encoder_output = self.force_encoder(force_encoder_input)
        
        # Reshape outputs
        mini_pointnet_output = mini_pointnet_output.view(Batch_size, self.obs_t, -1, self.Trans_dim)
        point_encoder_cls_output = point_encoder_cls_output.view(Batch_size, self.obs_t, -1)
        overall_point_encoder_output = overall_point_encoder_output.view(Batch_size, self.obs_t, -1)
        force_encoder_output = force_encoder_output.view(Batch_size, self.obs_t, -1, self.Trans_dim)

        # Prepare attention inputs
        q = force_encoder_output
        q = q.view(Batch_size, self.obs_t * q.size(2), self.Trans_dim)
        
        k_list = []
        for t in range(self.obs_t):
            k_t = torch.cat((
                point_encoder_cls_output[:, t].unsqueeze(1), 
                overall_point_encoder_output[:, t].unsqueeze(1)
            ), dim=1)
            k_list.append(k_t)
        k = torch.cat(k_list, dim=1)

        v = mini_pointnet_output.view(Batch_size, self.obs_t * mini_pointnet_output.size(2), self.Trans_dim)
        concatenated_kv = torch.cat((k, v), dim=1)

        # Attention blocks
        num_blocks = min(len(self.self_attention_blocks), len(self.cross_attention_blocks))
        for i in range(num_blocks):
            concatenated_kv = self.self_attention_blocks[i](concatenated_kv)
            q = self.cross_attention_blocks[i](q, concatenated_kv, concatenated_kv)

        # Flatten features
        q = torch.flatten(q, start_dim=1)
        concatenated_kv = torch.flatten(concatenated_kv, start_dim=1)

        # Encode robot state
        state_output = self.state_mlp(state_input)

        # Concatenate with state features
        q = torch.cat([q, state_output], dim=1)
        concatenated_kv = torch.cat([concatenated_kv, state_output], dim=1)

        # Diffusion inference
        if self.config['use_one_diffusion']:
            # Initialize with random noise
            naction = torch.randn((Batch_size, self.pred_time, self.arm_degree + self.hand_degree)).cuda()
            self.noise_scheduler_action.set_timesteps(self.num_inference_steps)
            
            # Denoising loop
            for k in self.noise_scheduler_action.timesteps:
                noise_pred = self.action_diffusion(
                    sample=naction,
                    timestep=k,
                    global_cond=q
                )
                naction = self.noise_scheduler_action.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
            
            # Denormalize to original scale
            naction = min_max_denormalize_robot_state(naction, self.config)
            return naction

        else:
            if self.config['use_two_diffusion']:
                # Dual diffusion inference
                naction_arm = torch.randn((Batch_size, self.pred_time, self.arm_degree)).cuda()
                naction_hand = torch.randn((Batch_size, self.pred_time, self.hand_degree)).cuda()

                self.noise_scheduler_arm.set_timesteps(self.num_inference_steps)
                self.noise_scheduler_hand.set_timesteps(self.num_inference_steps)

                # Denoise arm actions
                for k in self.noise_scheduler_arm.timesteps:
                    arm_noise_pred = self.arm_diffusion(
                        sample=naction_arm,
                        timestep=k,
                        global_cond=concatenated_kv
                    )
                    naction_arm = self.noise_scheduler_arm.step(
                        model_output=arm_noise_pred,
                        timestep=k,
                        sample=naction_arm
                    ).prev_sample

                # Denoise hand actions
                for k in self.noise_scheduler_hand.timesteps:
                    hand_noise_pred = self.hand_diffusion(
                        sample=naction_hand,
                        timestep=k,
                        global_cond=q
                    )
                    naction_hand = self.noise_scheduler_hand.step(
                        model_output=hand_noise_pred,
                        timestep=k,
                        sample=naction_hand
                    ).prev_sample

                # Combine and denormalize
                predicted_action = torch.cat([naction_arm, naction_hand], dim=-1)
                predicted_action = min_max_denormalize_robot_state(predicted_action, self.config)
                return predicted_action
            
            else:
                if self.config['use_arm_diffusion'] and not self.config['use_hand_diffusion']:
                    # Arm diffusion + hand MLP
                    naction_arm = torch.randn((Batch_size, self.pred_time, self.arm_degree)).cuda()
                    self.noise_scheduler_arm.set_timesteps(self.num_inference_steps)

                    # Denoise arm actions
                    for k in self.noise_scheduler_arm.timesteps:
                        arm_noise_pred = self.arm_diffusion(
                            sample=naction_arm,
                            timestep=k,
                            global_cond=concatenated_kv
                        )
                        naction_arm = self.noise_scheduler_arm.step(
                            model_output=arm_noise_pred,
                            timestep=k,
                            sample=naction_arm
                        ).prev_sample
                    
                    # Predict hand actions
                    action_hand = self.Hand_head(q).view(-1, self.pred_time, self.hand_degree)

                    # Combine and denormalize
                    predicted_action = torch.cat([naction_arm, action_hand], dim=-1)
                    predicted_action = min_max_denormalize_robot_state(predicted_action, self.config)
                    return predicted_action

                if self.config['use_hand_diffusion'] and not self.config['use_arm_diffusion']:
                    # Hand diffusion + arm MLP
                    naction_hand = torch.randn((Batch_size, self.pred_time, self.hand_degree)).cuda()
                    self.noise_scheduler_hand.set_timesteps(self.num_inference_steps)

                    # Denoise hand actions
                    for k in self.noise_scheduler_hand.timesteps:
                        hand_noise_pred = self.hand_diffusion(
                            sample=naction_hand,
                            timestep=k,
                            global_cond=q
                        )
                        naction_hand = self.noise_scheduler_hand.step(
                            model_output=hand_noise_pred,
                            timestep=k,
                            sample=naction_hand
                        ).prev_sample

                    # Predict arm actions
                    action_arm = self.Arm_head(concatenated_kv).view(-1, self.pred_time, self.arm_degree)

                    # Combine and denormalize
                    predicted_action = torch.cat([action_arm, naction_hand], dim=-1)
                    predicted_action = min_max_denormalize_robot_state(predicted_action, self.config)
                    return predicted_action


def load_pretrained_maskpoint_weights2(target_model, pretrained_path, component_name, logger=None):

    # Print loading information
    if logger:
        logger.info(f"Loading pretrained weights from: {pretrained_path}")
    else:
        print(f"Loading pretrained weights from: {pretrained_path}")
    
    # Load checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Extract transformer_q weights
    if 'base_model' in checkpoint:
        # Handle nested state dict
        state_dict = {k.replace("module.transformer_q.", ""): v 
                      for k, v in checkpoint['base_model'].items() 
                      if k.startswith('module.transformer_q.')}
    else:
        state_dict = {k.replace("transformer_q.", ""): v 
                      for k, v in checkpoint.items() 
                      if k.startswith('transformer_q.')}
    
    # Remove unnecessary keys
    keys_to_remove = [k for k in state_dict.keys() 
                      if k.startswith('decoder') or k.startswith('bin_cls_head')]
    for k in keys_to_remove:
        del state_dict[k]
    
    # Print matched weights count
    if logger:
        logger.info(f"Found {len(state_dict)} pretrained weight parameters")
    else:
        print(f"Found {len(state_dict)} pretrained weight parameters")
    
    # Load weights to target component
    component = getattr(target_model, component_name)
    incompatible_keys = component.load_state_dict(state_dict, strict=False)
    
    # Print mismatched keys
    if incompatible_keys.missing_keys:
        if logger:
            logger.info(f"Missing parameters: {incompatible_keys.missing_keys}")
        else:
            print(f"Missing parameters: {incompatible_keys.missing_keys}")
    
    if incompatible_keys.unexpected_keys:
        if logger:
            logger.info(f"Unexpected parameters: {incompatible_keys.unexpected_keys}")
        else:
            print(f"Unexpected parameters: {incompatible_keys.unexpected_keys}")
    
    if logger:
        logger.info("Pretrained weights loaded successfully")
    else:
        print("Pretrained weights loaded successfully")
    
    return target_model


def load_pretrained_maskpoint_weights(target_model, pretrained_path, logger=None):

    # Print loading information
    if logger:
        logger.info(f"Loading pretrained weights from: {pretrained_path}")
    else:
        print(f"Loading pretrained weights from: {pretrained_path}")
    
    # Load checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Extract transformer_q weights
    if 'base_model' in checkpoint:
        # Handle nested state dict
        state_dict = {k.replace("module.transformer_q.", ""): v 
                      for k, v in checkpoint['base_model'].items() 
                      if k.startswith('module.transformer_q.')}
    else:
        state_dict = {k.replace("transformer_q.", ""): v 
                      for k, v in checkpoint.items() 
                      if k.startswith('transformer_q.')}
    
    # Remove unnecessary keys
    keys_to_remove = [k for k in state_dict.keys() 
                      if k.startswith('decoder') or k.startswith('bin_cls_head')]
    for k in keys_to_remove:
        del state_dict[k]
    
    # Print matched weights count
    if logger:
        logger.info(f"Found {len(state_dict)} pretrained weight parameters")
    else:
        print(f"Found {len(state_dict)} pretrained weight parameters")
    
    # Load weights to point_encoder
    point_encoder = target_model.point_encoder
    incompatible_keys = point_encoder.load_state_dict(state_dict, strict=False)
    
    # Print mismatched keys
    if incompatible_keys.missing_keys:
        if logger:
            logger.info(f"Missing parameters: {incompatible_keys.missing_keys}")
        else:
            print(f"Missing parameters: {incompatible_keys.missing_keys}")
    
    if incompatible_keys.unexpected_keys:
        if logger:
            logger.info(f"Unexpected parameters: {incompatible_keys.unexpected_keys}")
        else:
            print(f"Unexpected parameters: {incompatible_keys.unexpected_keys}")
    
    if logger:
        logger.info("Pretrained weights loaded successfully")
    else:
        print("Pretrained weights loaded successfully")
    
    return target_model