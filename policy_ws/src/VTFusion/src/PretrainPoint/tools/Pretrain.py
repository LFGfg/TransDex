import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from utils.grad_utils import IterativePercentile
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append("..") 
from models.Dataset_process_nor import PCDDataset
from typing import Any, BinaryIO, List, Optional, Tuple, Union
from types import FunctionType
import os
import open3d as o3d
from torchmetrics import Accuracy
from timm.models.layers import trunc_normal_
from easydict import EasyDict as edict
import random
from torch.utils.data import Dataset
import copy
import math

def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")


class Compose_:

    def __init__(self, transforms):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _log_api_usage_once(self)
        self.transforms = transforms

    def __call__(self, img,img_):
        for t in self.transforms:
            img, img_= t(img,img_)

        return img,img_

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string



train_transforms = Compose_(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

# -------------------------- CD Calculation Core Function --------------------------
def compute_chamfer_distance(pred_pc, labels_pc):
    """Calculate the Chamfer Distance between two point clouds"""
    # Convert to Open3D point cloud object
    if not isinstance(pred_pc, o3d.geometry.PointCloud):
        pred_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_pc))
    if not isinstance(labels_pc, o3d.geometry.PointCloud):
        labels_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(labels_pc))
    
    # Build KDTree
    labels_tree = o3d.geometry.KDTreeFlann(labels_pc)
    pred_tree = o3d.geometry.KDTreeFlann(pred_pc)
    
    pred_points = np.asarray(pred_pc.points)
    labels_points = np.asarray(labels_pc.points)
    
    # Empty point cloud protection
    if len(pred_points) == 0 or len(labels_points) == 0:
        return float('inf')
    
    # Distance from prediction to label
    pred_to_labels = 0.0
    for point in pred_points:
        [k, idx, _] = labels_tree.search_knn_vector_3d(point, 1)
        pred_to_labels += np.sum((point - labels_points[idx[0]]) ** 2)
    pred_to_labels /= len(pred_points)
    
    # Distance from label to prediction
    labels_to_pred = 0.0
    for point in labels_points:
        [k, idx, _] = pred_tree.search_knn_vector_3d(point, 1)
        labels_to_pred += np.sum((point - pred_points[idx[0]]) ** 2)
    labels_to_pred /= len(labels_points)
    
    return pred_to_labels + labels_to_pred

def farthest_point_sample(data, npoints):
    """Farthest Point Sampling (FPS), adapted for torch tensor"""
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    N, D = data.shape
    xyz = data[:, :3]
    centroids = torch.zeros(npoints, dtype=torch.long)
    distance = torch.ones(N) * 1e10
    farthest = torch.randint(0, N, (1,)).item()
    
    for i in range(npoints):
        centroids[i] = farthest
        centroid = xyz[farthest:farthest+1]
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.argmax(distance).item()
    
    return data[centroids].numpy()


# -------------------------- Test Set CD Calculation Main Function --------------------------
def compute_cd_on_testset(args, config, logger, test_dataset):
    """After training, load the best model and calculate CD values on the test set (20% validation set)"""
    # 1. Build dataset (consistent split with training)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # batch_size=1 to avoid inconsistent point cloud lengths
    config.model.transformer_config.mask_ratio=0.7
    # 2. Build model and load the best weights
    base_model = builder.model_builder(config.model)  # Corresponding to your MaskPoint model
    if args.use_gpu:
        base_model = base_model.cuda()
    base_model.transformer_q.mask_ratio = 0.7
    # Load the last model weights
    last_ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location='cpu')
        # Handle the module. prefix from distributed training
        state_dict = ckpt['base_model']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        base_model.load_state_dict(state_dict, strict=False)
        print_log(f"Loaded last model from {last_ckpt_path}", logger=logger)
    else:
        print_log(f"Last checkpoint not found at {last_ckpt_path}", logger=logger)
        return
    
    # 3. Set model to evaluation mode
    base_model.eval()
    all_cd_values = []
    
    # 4. Iterate over test set to calculate CD
    with torch.no_grad():
        for idx, (data, label) in enumerate(test_dataloader):
            # Data preprocessing (consistent with training)
            data = data.cuda()
            label = label.cuda()
            npoints = config.dataset.npoints
            data = misc.fps(data, npoints)  # FPS sampling during training
            
            # Model forward inference (get predicted point cloud)
            # Adapt to the forward output of MaskPoint model: query_points are the core predicted points
            neighborhood, center = base_model.group_divider(data)
            q_cls_feature, query_preds, query_labels, query_points = base_model.transformer_q(
                neighborhood, center, data, label,return_cd=True
            )
            # Threshold filtering of valid predicted points
            query_preds = torch.sigmoid(query_preds.transpose(1, 2))
            threshold = 0.45  # Adjustable threshold
            pos_indices = (query_preds[..., 1] > threshold).nonzero(as_tuple=True)
            selected_points = query_points[pos_indices].cpu().numpy()
            
            # FPS sampling to fixed number of points (avoid excessive number of point cloud points)
            if len(selected_points) > 1000:
                selected_points = farthest_point_sample(selected_points, 1000)
            elif len(selected_points) == 0:
                print_log(f"Sample {idx+1}: No valid predicted points", logger=logger)
                all_cd_values.append(float('inf'))
                continue
            
            # Label point cloud processing
            label_np = label[0].cpu().numpy()  # batch_size=1, take the first sample
            
            # Calculate CD
            cd = compute_chamfer_distance(selected_points, label_np)
            all_cd_values.append(cd)
            print_log(f"Test Sample {idx+1}/{len(test_dataloader)} CD: {cd:.6f}", logger=logger)
    
    # 5. Statistical results
    if all_cd_values:
        valid_cd = [v for v in all_cd_values if not np.isinf(v)]
        if valid_cd:
            avg_cd = np.mean(valid_cd)

            print_log(f"\n===== CD Statistics =====", logger=logger)
            print_log(f"Average CD: {avg_cd:.6f}", logger=logger)

        else:
            print_log("No valid CD values (all predicted point clouds are empty)", logger=logger)
    else:
        print_log("No CD values computed", logger=logger)

def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    
    dataset = PCDDataset(config.dataset['data_dir'])

   
    train_size = int(config.dataset.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, val_size]) 
    train_sampler = None 
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    extra_train_dataloader=None
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)
        print_log(f"Loaded pretrained weights from {args.start_ckpts}", logger=logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Data on cuda:0' , logger = logger)
        base_model = nn.DataParallel(base_model,device_ids=[0]).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
    spike_idx = 0
    gradclip_percentile = config.get('gradclip_percentile', -1)
    if gradclip_percentile > 0:
        grad_history = IterativePercentile(p=gradclip_percentile)
    else:
        grad_history = None

    # trainval
    # training
    base_model.zero_grad()
    # fixed_sample = None
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss1', 'Loss2'])

        num_iter = 0
        grad_clip_val = config.grad_norm_clip

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (data,label_data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.npoints
            points = data.cuda()
            label_points=label_data.cuda()
            # points = data[0].cuda()
            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            points,label_points = train_transforms(points,label_points)

            # if args.overfit_single_batch:
            #     if fixed_sample is None:
            #         fixed_sample = points.clone()
            #     else:
            #         points = fixed_sample.clone()

            loss_1, loss_2 = base_model(points, label_points)
            loss_1 = loss_1.mean()  # Use mean instead of sum for better scaling
            loss_2 = loss_2.mean()  # Use mean instead of sum for better scaling

            _loss = loss_1+ loss_2 # Combine losses for optimization


            # for name, param in base_model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.norm()}")
            #     else:
            #         print(f"{name}: None")
            # for name, param in base_model.named_parameters():
            #     print(f"{name}: requires_grad = {param.requires_grad}")

  
            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    if grad_history is not None:
                        grad_norm = torch.cat([p.grad.view(-1) for p in base_model.parameters() if p.grad is not None]).norm().item()
                        grad_clip_val = grad_history.add(grad_norm)
                        if train_writer is not None:
                            train_writer.add_scalar('Loss/Batch/clip_val', grad_clip_val, n_itr)
                    grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip_val, norm_type=2)
                    if train_writer is not None:
                        train_writer.add_scalar('Loss/grad_norm', grad_norm.item(), n_itr)
                num_iter = 0
                optimizer.zero_grad()
                _loss.sum().backward()

                optimizer.step()

                
            if args.distributed:
                loss_1 = dist_utils.reduce_tensor(loss_1, args)
                loss_2 = dist_utils.reduce_tensor(loss_2, args)
                losses.update([loss_1.item(), loss_2.item()])
            else:
                losses.update([loss_1.item(), loss_2.item()])

            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss_1', loss_1.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_2', loss_2.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # if idx % 20 == 0:
            print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_2', losses.avg(1), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)
        # if epoch % args.val_freq == 0 and epoch != 0:
        if epoch % 10 == 0 and epoch != 0:

            # Validate the current model
            metrics = validate(base_model, train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)  

    # After the training: calculate the CD value of the test set
    print_log("\n===== Start computing Chamfer Distance on test set =====", logger=logger)
    compute_cd_on_testset(args, config, logger,test_dataset) 

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode
    total_accuracy = 0
    total_samples = 0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    npoints = config.dataset.npoints
    with torch.no_grad():
        for idx, (data,lable_data) in enumerate(test_dataloader):
            points = data.cuda()
            label = lable_data.cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            points,label = train_transforms(points,label)

            _, acc, precision, recall = base_model(points, label, noaug=True, return_acc_=True)
            batch_size = points.size(0)
            # print(batch_size)
            # print('acc_mean',acc.mean())
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
            total_accuracy += acc.mean() * batch_size
            total_precision += precision * batch_size
            total_recall += recall * batch_size
            total_f1 += f1 * batch_size
            total_samples += batch_size

        print('total_samples:',total_samples)
        if total_samples > 0:
            average_accuracy = total_accuracy / total_samples
            average_precision = total_precision / total_samples
            average_recall = total_recall / total_samples
            average_f1 = total_f1 / total_samples
            
            if isinstance(average_accuracy, torch.Tensor):
                average_accuracy = average_accuracy.item()  
            if isinstance(average_precision, torch.Tensor):
                average_precision = average_precision.item()
            if isinstance(average_recall, torch.Tensor):
                average_recall = average_recall.item()
            if isinstance(average_f1, torch.Tensor):
                average_f1 = average_f1.item()
        else:
            average_accuracy = 0
            average_precision = 0
            average_recall = 0
            average_f1 = 0

        print_log(f'[Validation] EPOCH: {epoch}  acc = {average_accuracy:.4f}  '
          f'precision = {average_precision:.4f}  recall = {average_recall:.4f}  '
          f'f1 = {average_f1:.4f}', logger=logger)
        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', average_accuracy, epoch)
        val_writer.add_scalar('Metric/Precision', average_precision, epoch)
        val_writer.add_scalar('Metric/Recall', average_recall, epoch)
        val_writer.add_scalar('Metric/f1', average_f1, epoch)


    return Acc_Metric(average_accuracy)
def test_net():
    pass


if __name__ == "__main__":
    run_net()