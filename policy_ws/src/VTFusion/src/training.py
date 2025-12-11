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
import FusionNetwork as FusionNetwork
import os
import argparse
import yaml
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PretrainPoint.utils.grad_utils import IterativePercentile  # Assume gradient percentile tool is imported
from VTFusion_dataset import VT_dataset
import torch.distributed as dist
from torch.utils.data import random_split
# Configuration file path
config_path = "config/config.yaml"


def load_pretrained_maskpoint_weights(target_model, pretrained_path, component_name, logger=None):
    """
    Load weights from pretrained MaskPoint model to the specified component of the target model
    
    Args:
        target_model: Target model instance (FusionNet1 or FusionNet2)
        pretrained_path: Path to pretrained model weights
        component_name: Target component name, e.g., 'point_encoder' or 'force_encoder'
        logger: Logger instance (optional)
    """
    # Print loading information
    if logger:
        logger.info(f"Loading pretrained weights from: {pretrained_path}")
    else:
        print(f"Loading pretrained weights from: {pretrained_path}")
    
    # Load pretrained weights
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Get transformer_q weights
    if 'base_model' in checkpoint:
        # Handle nested structure
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
    
    # Print matched weights
    if logger:
        logger.info(f"Found {len(state_dict)} pretrained weight parameters")
    else:
        print(f"Found {len(state_dict)} pretrained weight parameters")
    
    # Load weights to the specified component of the target model
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




def train(rank, args, config):
    torch.autograd.set_detect_anomaly(True)
    # 1. Set CUDA device (must be before initializing process group)
    torch.cuda.set_device(rank)  # rank is the ID of current process (0, 1, ..., world_size-1)

    # 2. Initialize distributed process group (critical step)
    torch.distributed.init_process_group(
        backend='nccl',  # Recommended to use nccl backend (GPU training)
        init_method='env://',  # Read distributed configuration from environment variables
        world_size=config['world_size'],  # Total number of processes
        rank=rank  # Current process ID
    )

    # 3. Verify process group initialization (optional, for debugging)
    if not torch.distributed.is_initialized():
        raise RuntimeError("Distributed process group initialization failed!")
    # Create save directories
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['ckpt_dir'], exist_ok=True)
    
    # Create TensorBoard writers
    if rank == 0:
        log_tag = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
        train_writer = SummaryWriter(log_dir=f"{config['log_dir']}/{log_tag}/train")
        val_writer = SummaryWriter(log_dir=f"{config['log_dir']}/{log_tag}/val")
    else:
        train_writer = None
        val_writer = None

    model = FusionNetwork.FusionNet1(config).to(rank)
    # Call in train function
    if config['pointEncoder_pretrain']:
        ckpt = config['pointEncoder_pretrain_ckpt']
        model = load_pretrained_maskpoint_weights(
            target_model=model,
            pretrained_path=ckpt,
            component_name='point_encoder'
        )
        if rank == 0:  # Print only in main process
            print('Load point encoder pretrain ckpt from: ', ckpt)

    if config['forceEncoder_pretrain']:
        ckpt = config['forceEncoder_pretrain_ckpt']
        model = load_pretrained_maskpoint_weights(
            target_model=model,
            pretrained_path=ckpt,
            component_name='force_encoder'
        )
        if rank == 0:  # Print only in main process
            print('Load force encoder pretrain ckpt from: ', ckpt)

    # Load pretrained parameters and freeze
    start_epoch = 0
    best_metric = float('inf')  # Initialize best metric
    
    # if config.get('pointEncoder_pretrain_ckpt', None):
    #     # Freeze all point_encoder parameters except self.fc
    #     for name, param in model.point_encoder.named_parameters():
    #         if 'fc' not in name:  # Exclude layers containing 'fc'
    #             param.requires_grad = False
    #         else:
    #             param.requires_grad = True  # Ensure fc layers are trainable
    #             if rank == 0:  # Print only in main process
    #                 print(f"Parameter {name} is not frozen and trainable")

    # if config.get('forceEncoder_pretrain_ckpt', None):
    #     for param in model.force_encoder.parameters():
    #         param.requires_grad = False

    # Group optimizer settings
    # Dynamically group model parameters according to config
    if config['use_one_diffusion']:
        # Scenario 1: Use single action_diffusion, only one optimizer needed (merge all parameters)
        # Extract all trainable parameters (including action_diffusion and other parameters)
        all_params = [
            param for param in model.parameters()
            if param.requires_grad  # Only include parameters requiring gradient update
        ]
        # Initialize single optimizer to optimize all parameters uniformly
        optimizer = torch.optim.AdamW(all_params, lr=config['lr'], weight_decay=config['weight_decay'])
                # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['num_epochs'], 
            eta_min=config['min_lr']
        )


    elif config['use_two_diffusion']:
        # Scenario 2: Use arm_diffusion and hand_diffusion (dual diffusion)
        arm_params = list(model.arm_diffusion.parameters()) if hasattr(model, 'arm_diffusion') else []
        hand_params = list(model.hand_diffusion.parameters()) if hasattr(model, 'hand_diffusion') else []
        other_params = [
            param for name, param in model.named_parameters()
            if 'arm_diffusion' not in name and 'hand_diffusion' not in name and param.requires_grad
        ]
        optimizer_arm = torch.optim.AdamW(arm_params, lr=config['lr'], weight_decay=config['weight_decay'])
        optimizer_hand = torch.optim.AdamW(hand_params, lr=config['lr'], weight_decay=config['weight_decay'])
        optimizer_other = torch.optim.AdamW(other_params, lr=config['lr'], weight_decay=config['weight_decay'])
        # Learning rate schedulers
        lr_scheduler_arm = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_arm, 
            T_max=config['num_epochs'], 
            eta_min=config['min_lr']
        )
        lr_scheduler_hand = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_hand, 
            T_max=config['num_epochs'], 
            eta_min=config['min_lr']
        )
        lr_scheduler_other = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_other, 
            T_max=config['num_epochs'], 
            eta_min=config['min_lr']
        )
    else:
        # Scenario 3: Use only arm_diffusion or hand_diffusion
        if config['use_arm_diffusion'] and not config['use_hand_diffusion']:
            # Only arm_diffusion + Hand_head
            arm_params = list(model.arm_diffusion.parameters()) if hasattr(model, 'arm_diffusion') else []
            hand_params = list(model.Hand_head.parameters()) if hasattr(model, 'Hand_head') else []
            other_params = [
                param for name, param in model.named_parameters()
                if 'arm_diffusion' not in name and 'Hand_head' not in name and param.requires_grad
            ]
            optimizer_arm = torch.optim.AdamW(arm_params, lr=config['lr'], weight_decay=config['weight_decay'])
            optimizer_hand = torch.optim.AdamW(hand_params, lr=config['lr'], weight_decay=config['weight_decay'])
            optimizer_other = torch.optim.AdamW(other_params, lr=config['lr'], weight_decay=config['weight_decay'])

        elif config['use_hand_diffusion'] and not config['use_arm_diffusion']:
            # Only hand_diffusion + Arm_head
            hand_params = list(model.hand_diffusion.parameters()) if hasattr(model, 'hand_diffusion') else []
            arm_params = list(model.Arm_head.parameters()) if hasattr(model, 'Arm_head') else []
            other_params = [
                param for name, param in model.named_parameters()
                if 'hand_diffusion' not in name and 'Arm_head' not in name and param.requires_grad
            ]
            optimizer_hand = torch.optim.AdamW(hand_params, lr=config['lr'], weight_decay=config['weight_decay'])
            optimizer_arm = torch.optim.AdamW(arm_params, lr=config['lr'], weight_decay=config['weight_decay'])
            optimizer_other = torch.optim.AdamW(other_params, lr=config['lr'], weight_decay=config['weight_decay'])


        # Learning rate schedulers
        lr_scheduler_arm = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_arm, 
            T_0=config['T_0'], T_mult=config['T_mult'],
            eta_min=config['min_lr']
        )
        lr_scheduler_hand = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_hand, 
            T_0=config['T_0'], T_mult=config['T_mult'],
            eta_min=config['min_lr']
        )
        lr_scheduler_other = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_other, 
            T_0=config['T_0'], T_mult=config['T_mult'],
            eta_min=config['min_lr']
        )

    # Model resume functionality
    if args.resume:
        try:
            checkpoint = torch.load(args.resume_ckpt, map_location=f'cuda:{rank}',weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            if config['use_one_diffusion']:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                optimizer_arm.load_state_dict(checkpoint['optimizer_arm_state_dict'])
                optimizer_hand.load_state_dict(checkpoint['optimizer_hand_state_dict'])
                optimizer_other.load_state_dict(checkpoint['optimizer_other_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_metric = checkpoint.get('best_metric', float('inf'))
            if rank == 0:  # Print only in main process
                print_log(f"Successfully resumed training from {args.resume_ckpt}...",  logger=None)
        except Exception as e:
            if rank == 0:  # Print only in main process
                print_log(f"Failed to resume training: {str(e)}",  logger=None)
                print_log("Will start training from scratch",  logger=None)

    # Adaptive gradient clipping settings
    if config['grad_clip_percentile'] > 0:
        grad_history_arm = IterativePercentile(p=config['grad_clip_percentile'])
        grad_history_hand = IterativePercentile(p=config['grad_clip_percentile'])
        grad_history_other = IterativePercentile(p=config['grad_clip_percentile'])
    else:
        grad_history_arm = None
        grad_history_hand = None
        grad_history_other = None

    # Data loading
    dataset = VT_dataset(config['dataset_dir'], config, task_name=config['task_name'])
    
    # Split ratio (configurable in config, default 8:2)
    test_size = config.get('test_size', 0.2)  # Test set ratio
    train_size = len(dataset) - int(len(dataset) * test_size)
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, int(len(dataset) * test_size)],
        generator=torch.Generator().manual_seed(42)  # Fix random seed to ensure consistent split
    )
    if rank == 0:  # Print only in main process
        print_log(f"Training set size: {len(train_dataset)}, Test set size: {len(test_dataset)}", logger=None)


    # Training set DataLoader
    # Create distributed sampler with drop_last=True
    train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config['world_size'],
            rank=rank,
            shuffle=True,
            drop_last=True  # Discard incomplete batches
        )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,  # Controlled by DistributedSampler
        pin_memory=True,
        persistent_workers=True,
        sampler=train_sampler
    )

    # Test set DataLoader (newly added)
    test_sampler = DistributedSampler(
        test_dataset,
        shuffle=False  # Do not shuffle test set
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,  # No need to shuffle test set
        pin_memory=True,
        persistent_workers=True,
        sampler=test_sampler
    )
    global_step = 0
    for epoch_idx in range(start_epoch, config['num_epochs']):
        if rank == 0:  # Print only in main process
            print('Train Started!!')
        train_sampler.set_epoch(epoch_idx)
        test_sampler.set_epoch(epoch_idx)
        model.train()
        # Initialize loss lists (distinguish single/multi scenarios)
        if config['use_one_diffusion']:
            epoch_loss = []  # Total loss list for single scenario
        else:
            epoch_loss_arm = []
            epoch_loss_hand = []
            epoch_loss_other = []
        
        with tqdm(train_dataloader, desc=f'Epoch {epoch_idx}/{config["num_epochs"]}', disable=rank != 0) as tepoch:
            
            for nbatch in tepoch:
                # Data preprocessing
                mini_pointnet_input, point_encoder_input, overall_point_encoder_input, force_encoder_input, state_input, rel_actions = nbatch
                mini_pointnet_input = mini_pointnet_input.to(rank)
                point_encoder_input = point_encoder_input.to(rank)
                overall_point_encoder_input = overall_point_encoder_input.to(rank)
                force_encoder_input = force_encoder_input.to(rank)
                state_input = state_input.to(rank)
                rel_actions = rel_actions.to(rank)

                # Forward propagation and loss calculation (single scenario)
                if config['use_one_diffusion']:
                    noise_pred, noise = model(
                        mini_pointnet_input, point_encoder_input, overall_point_encoder_input, force_encoder_input, state_input, rel_actions
                    )
                    loss = nn.functional.mse_loss(noise_pred, noise, reduction='mean')
                    
                    # Backward propagation and parameter update
                    optimizer.zero_grad()
                    loss.backward()
                    if config['grad_clip_value'] > 0:
                        all_params = [param for param in model.parameters() if param.requires_grad]
                        grad_norm = torch.cat([p.grad.view(-1) for p in all_params if p.grad is not None]).norm().item()
                        if train_writer and rank == 0:
                            train_writer.add_scalar('Gradients/ClipValue/action', grad_norm, global_step)
                        torch.nn.utils.clip_grad_norm_(all_params, config['grad_clip_value'])
                    optimizer.step()

                    # Step-level operations (executed per batch)
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)  # Only record, no average calculation
                    tepoch.set_postfix(loss=loss_cpu)  # Update progress bar
                    if train_writer and rank == 0:
                        train_writer.add_scalar('Loss/Step/total', loss_cpu, global_step)  # Step-level logging
                        train_writer.add_scalar('LearningRate/total', optimizer.param_groups[0]['lr'], global_step)
                        if config.get('log_gradients', False):
                            total_grad_norm = torch.cat([p.grad.view(-1) for p in all_params if p.grad is not None]).norm().item()
                            train_writer.add_scalar('Gradients/Norm/total', total_grad_norm, global_step)
                    global_step += 1

                # Forward propagation and loss calculation (multi scenario)
                else:
                    if config['use_two_diffusion']:
                        arm_noise_pred, noise_arm, hand_noise_pred, noise_hand = model(
                            mini_pointnet_input, point_encoder_input, overall_point_encoder_input, force_encoder_input, state_input, rel_actions
                        )
                        loss_arm = nn.functional.mse_loss(arm_noise_pred, noise_arm, reduction='mean')
                        loss_hand = nn.functional.mse_loss(hand_noise_pred, noise_hand, reduction='mean')
                        loss_other = config['arm_ratio'] * loss_arm + config['hand_ratio'] * loss_hand
                    elif config['use_arm_diffusion'] and not config['use_hand_diffusion']:
                        arm_noise_pred, noise_arm, action_hand, rel_hand = model(
                            mini_pointnet_input, point_encoder_input, overall_point_encoder_input, force_encoder_input, state_input, rel_actions
                        )
                        loss_arm = nn.functional.mse_loss(arm_noise_pred, noise_arm, reduction='mean')
                        loss_hand = nn.functional.mse_loss(action_hand, rel_hand, reduction='mean')
                        loss_other = config['arm_ratio'] * loss_arm + config['hand_ratio'] * loss_hand
                    elif config['use_hand_diffusion'] and not config['use_arm_diffusion']:
                        action_arm, rel_arm, hand_noise_pred, noise_hand = model(
                            mini_pointnet_input, point_encoder_input, overall_point_encoder_input, force_encoder_input, state_input, rel_actions
                        )
                        loss_arm = nn.functional.mse_loss(action_arm, rel_arm, reduction='mean')
                        loss_hand = nn.functional.mse_loss(hand_noise_pred, noise_hand, reduction='mean')
                        loss_other = config['arm_ratio'] * loss_arm + config['hand_ratio'] * loss_hand
                    
                    # Backward propagation and parameter update
                    optimizer_arm.zero_grad()
                    loss_arm.backward(retain_graph=True)
                    if config['grad_clip_value'] > 0:
                        if grad_history_arm:
                            grad_norm = torch.cat([p.grad.view(-1) for p in arm_params if p.grad is not None]).norm().item()
                            clip_val = grad_history_arm.add(grad_norm)
                            if train_writer and rank == 0:
                                train_writer.add_scalar('Gradients/ClipValue/arm', clip_val, global_step)
                            torch.nn.utils.clip_grad_norm_(arm_params, clip_val)
                        else:
                            torch.nn.utils.clip_grad_norm_(arm_params, config['grad_clip_value'])
                    

                    optimizer_hand.zero_grad()
                    loss_hand.backward(retain_graph=True)
                    if config['grad_clip_value'] > 0:
                        if grad_history_hand:
                            grad_norm = torch.cat([p.grad.view(-1) for p in hand_params if p.grad is not None]).norm().item()
                            clip_val = grad_history_hand.add(grad_norm)
                            if train_writer and rank == 0:
                                train_writer.add_scalar('Gradients/ClipValue/hand', clip_val, global_step)
                            torch.nn.utils.clip_grad_norm_(hand_params, clip_val)
                        else:
                            torch.nn.utils.clip_grad_norm_(hand_params, config['grad_clip_value'])
                    

                    optimizer_other.zero_grad()
                    loss_other.backward()
                    if config['grad_clip_value'] > 0:
                        if grad_history_other:
                            grad_norm = torch.cat([p.grad.view(-1) for p in other_params if p.grad is not None]).norm().item()
                            clip_val = grad_history_other.add(grad_norm)
                            if train_writer and rank == 0:
                                train_writer.add_scalar('Gradients/ClipValue/other', clip_val, global_step)
                            torch.nn.utils.clip_grad_norm_(other_params, clip_val)
                        else:
                            torch.nn.utils.clip_grad_norm_(other_params, config['grad_clip_value'])
                    optimizer_arm.step()
                    optimizer_hand.step()
                    optimizer_other.step()

                    # Step-level operations (executed per batch)
                    loss_arm_cpu = loss_arm.item()
                    loss_hand_cpu = loss_hand.item()
                    loss_other_cpu = loss_other.item()
                    epoch_loss_arm.append(loss_arm_cpu)
                    epoch_loss_hand.append(loss_hand_cpu)
                    epoch_loss_other.append(loss_other_cpu)
                    tepoch.set_postfix(loss_arm=loss_arm_cpu, loss_hand=loss_hand_cpu, loss_other=loss_other_cpu)
                    if train_writer and rank == 0:
                        train_writer.add_scalar('Loss/Step/arm', loss_arm_cpu, global_step)
                        train_writer.add_scalar('Loss/Step/hand', loss_hand_cpu, global_step)
                        train_writer.add_scalar('Loss/Step/other', loss_other_cpu, global_step)
                        train_writer.add_scalar('LearningRate/arm', optimizer_arm.param_groups[0]['lr'], global_step)
                        train_writer.add_scalar('LearningRate/hand', optimizer_hand.param_groups[0]['lr'], global_step)
                        train_writer.add_scalar('LearningRate/other', optimizer_other.param_groups[0]['lr'], global_step)
                        if config.get('log_gradients', False):
                            arm_grad_norm = torch.cat([p.grad.view(-1) for p in arm_params if p.grad is not None]).norm().item()
                            hand_grad_norm = torch.cat([p.grad.view(-1) for p in hand_params if p.grad is not None]).norm().item()
                            other_grad_norm = torch.cat([p.grad.view(-1) for p in other_params if p.grad is not None]).norm().item()
                            train_writer.add_scalar('Gradients/Norm/arm', arm_grad_norm, global_step)
                            train_writer.add_scalar('Gradients/Norm/hand', hand_grad_norm, global_step)
                            train_writer.add_scalar('Gradients/Norm/other', other_grad_norm, global_step)
                    global_step += 1

            # -------------- Epoch-level operations (executed once after entire epoch) --------------
            # 1. Calculate average loss for the entire epoch
            if config['use_one_diffusion']:
                avg_loss = np.mean(epoch_loss)  # Calculate based on all batch losses in the epoch
            else:
                avg_loss_arm = np.mean(epoch_loss_arm)
                avg_loss_hand = np.mean(epoch_loss_hand)
                avg_loss_other = np.mean(epoch_loss_other)

            # 2. Record epoch-level logs (once per epoch)
            if train_writer and rank == 0:
                if config['use_one_diffusion']:
                    train_writer.add_scalar('Loss/Epoch/total', avg_loss, epoch_idx)
                else:
                    train_writer.add_scalar('Loss/Epoch/arm', avg_loss_arm, epoch_idx)
                    train_writer.add_scalar('Loss/Epoch/hand', avg_loss_hand, epoch_idx)
                    train_writer.add_scalar('Loss/Epoch/other', avg_loss_other, epoch_idx)

            # 3. Model saving (executed only in main process to avoid duplicate writing by multiple processes)
            if rank == 0:
                # Save latest model (once per epoch)
                save_dict = {
                    'epoch': epoch_idx,
                    'model_state_dict': model.state_dict(),
                    'best_metric': best_metric
                }
                if config['use_one_diffusion']:
                    save_dict['optimizer_state_dict'] = optimizer.state_dict()
                else:
                    save_dict['optimizer_arm_state_dict'] = optimizer_arm.state_dict()
                    save_dict['optimizer_hand_state_dict'] = optimizer_hand.state_dict()
                    save_dict['optimizer_other_state_dict'] = optimizer_other.state_dict()
                if (epoch_idx + 1) % 20 == 0:
                    torch.save(save_dict, f"{config['ckpt_dir']}/model-latest.pt")
                    print_log(f"Saved model at epoch {epoch_idx}", logger=None)
                if (epoch_idx + 1) % 601 == 0:
                    torch.save(save_dict, f"{config['ckpt_dir']}/model-latest-601.pt")
                    print_log(f"Saved model at epoch 601", logger=None)
                # Save every 20 epochs (with validation)
                if (epoch_idx + 1) % config['validate_every'] == 0:
                    # Note: It is recommended to replace with validation set dataloader (training set is used currently, results are meaningless)
                    if rank == 0:  # Print only in main process
                        print('Testing Started!!')
                    test_avg_diffs = validate(model, test_dataloader, rank, config)  # Use test set DataLoader
                    test_metric = np.sum(np.abs(test_avg_diffs))
                     # Record test set logs
                    if train_writer and rank == 0:
                        train_writer.add_scalar('Test/TotalMetric', test_metric, epoch_idx)
                        for i, diff in enumerate(test_avg_diffs):
                            train_writer.add_scalar(f'Test/JointDiff/Joint_{i+1}', diff, epoch_idx)
                    torch.save(save_dict, f"{config['ckpt_dir']}/model-epoch-{epoch_idx}-new.pt")
                    print_log(f"Saved model at epoch {epoch_idx}", logger=None)

                # Save best model (based on average loss of entire epoch)
                if config['use_one_diffusion']:
                    current_metric = avg_loss
                else:
                    current_metric = avg_loss_other
                if current_metric < best_metric:
                    best_metric = current_metric
                    torch.save(save_dict, f"{config['ckpt_dir']}/model-best-new.pt")
                    if rank == 0:  # Print only in main process
                        print_log(f"Saved best model at epoch {epoch_idx}, metric: {best_metric:.6f}",  logger=None)

            # 4. Learning rate update (once per epoch)
            if config['use_one_diffusion']:
                lr_scheduler.step()
            else:
                lr_scheduler_arm.step()
                lr_scheduler_hand.step()
                lr_scheduler_other.step()


    # Close TensorBoard writers
    if train_writer and rank == 0:
        train_writer.close()
    if val_writer and rank == 0:
        val_writer.close()

    # Clean up distributed environment
    dist.destroy_process_group()

def validate(model, dataloader, rank, config):
    model.eval()
    # Initialize cumulative sum of differences and sample count for six joint angles
    joint_diffs_sum = np.zeros(config['fusion_net']['arm_degree']+config['fusion_net']['hand_degree'])  # Assume 22 joint angles
    total_samples = 0
    
    with torch.no_grad():
        for nbatch in dataloader:
            # Data processing
            mini_pointnet_input, point_encoder_input, overall_point_encoder_input, force_encoder_input, state_input, rel_actions = nbatch
            mini_pointnet_input = mini_pointnet_input.to(rank)
            point_encoder_input = point_encoder_input.to(rank)
            overall_point_encoder_input = overall_point_encoder_input.to(rank)
            force_encoder_input = force_encoder_input.to(rank)
            state_input = state_input.to(rank) 
            rel_actions = rel_actions.to(rank)
            
            # Model inference
            actions_pred = model.infer(
                mini_pointnet_input, 
                point_encoder_input, 
                overall_point_encoder_input,
                force_encoder_input,
                state_input
            )
            
            # Calculate difference: pred - target (shape: [batch_size, pred_time, 23])
            batch_diffs = (actions_pred - rel_actions).cpu().numpy()
            
            # Accumulate total sum of differences for 22 joint angles
            joint_diffs_sum += np.sum(np.abs(batch_diffs), axis=(0, 1))  # Sum over batch and time dimensions
            total_samples += batch_diffs.shape[0] * batch_diffs.shape[1]  # Total samples = batch_size * pred_time
    
    # Calculate average difference for each of the six joint angles
    avg_diffs = joint_diffs_sum / total_samples
    
    # Print results (optional)
    log_msg = "Average prediction difference for each joint angle:\n"
    for i, diff in enumerate(avg_diffs):
        log_msg += f"  Joint {i+1}: {diff:.6f}\n"
    if rank == 0:  # Print only in main process
        print_log(log_msg, logger=None)
    
    model.train()  # Restore training mode
    return avg_diffs  # Return numpy array with shape [22]

if __name__ == "__main__":
    # Set global random seed in main function
    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Training configuration')
    # Reserved command line arguments
    parser.add_argument('--resume', action='store_true', help="Whether to resume training from checkpoint")
    parser.add_argument('--resume_ckpt', type=str, default='./checkpoints/model-latest.pt', help="Path to checkpoint for resuming training")
    parser.add_argument('--config_path', type=str, default='config/config.yaml', help="Configuration file path, default is config/config.yaml")
    args = parser.parse_args()

    # Read configuration file
    config_path = args.config_path
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print_log(f"Successfully loaded configuration from {config_path}", None)

    # Ensure environment variables required for distributed training
    os.environ['MASTER_ADDR'] = config.get('master_addr', 'localhost')
    os.environ['MASTER_PORT'] = config.get('master_port', '12345')
    
    # Start multi-process training
    torch.multiprocessing.spawn(train, args=(args, config), nprocs=config['world_size'], join=True)