# -*- coding: utf-8 -*-
# @Author  : Juntao Wu, XinZhe Xie
# @University  : University of Science and Technology of China, ZheJiang University

import argparse
import time
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from tqdm import tqdm
import torch
import os
import torch.nn as nn
import pandas as pd
from network EventStackMFF
from Dataloader import get_event_dataloader
from utils import to_image, count_parameters, config_model_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for EventStackMFF")

    parser.add_argument('--save_name', default='train_runs_event', help='Name for saving the model and logs')

    parser.add_argument('--train_img_stack', default='', type=str, help='Path to training image stack root (dataset 1)')
    parser.add_argument('--train_evt_stack', default='', type=str, help='Path to training event stack root (dataset 1)')
    parser.add_argument('--train_depth_continuous', default='', type=str, help='Path to training depth maps (dataset 1)')

    parser.add_argument('--train_img_stack_2', default='', type=str, help='Path to training image stack root (dataset 2)')
    parser.add_argument('--train_evt_stack_2', default='', type=str, help='Path to training event stack root (dataset 2)')
    parser.add_argument('--train_depth_continuous_2', default='', type=str, help='Path to training depth maps (dataset 2)')

    parser.add_argument('--train_img_stack_3', default='', type=str,help='Path to training image stack root (dataset 3)')
    parser.add_argument('--train_evt_stack_3', default='', type=str,help='Path to training event stack root (dataset 3)')
    parser.add_argument('--train_depth_continuous_3', default='', type=str,help='Path to training depth maps (dataset 3)')

    parser.add_argument('--val_img_stack', default='', type=str, help='Path to val image stack root (dataset 1)')
    parser.add_argument('--val_evt_stack', default='', type=str, help='Path to val event stack root (dataset 1)')
    parser.add_argument('--val_depth_continuous', default='', type=str, help='Path to val depth maps (dataset 1)')

    parser.add_argument('--val_img_stack_2', default='', type=str, help='Path to val image stack root (dataset 2)')
    parser.add_argument('--val_evt_stack_2', default='', type=str, help='Path to val event stack root (dataset 2)')
    parser.add_argument('--val_depth_continuous_2', default='', type=str, help='Path to val depth maps (dataset 2)')

    parser.add_argument('--val_img_stack_3', default='', type=str, help='Path to val image stack root (dataset 3)')
    parser.add_argument('--val_evt_stack_3', default='', type=str, help='Path to val event stack root (dataset 3)')
    parser.add_argument('--val_depth_continuous_3', default='', type=str, help='Path to val depth maps (dataset 3)')

    parser.add_argument('--use_train_dataset_1', type=bool, default=True)
    parser.add_argument('--use_train_dataset_2', type=bool, default=True)
    parser.add_argument('--use_train_dataset_3', type=bool, default=True)

    parser.add_argument('--use_val_dataset_1', type=bool, default=True)
    parser.add_argument('--use_val_dataset_2', type=bool, default=True)
    parser.add_argument('--use_val_dataset_3', type=bool, default=True)

    parser.add_argument('--subset_fraction_train', type=float, default=1.0)
    parser.add_argument('--subset_fraction_val', type=float, default=1.0)
    parser.add_argument('--training_image_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--loss_ratio', type=list, default=[0, 1])
    parser.add_argument('--num_workers', type=int, default=8)

    return parser.parse_args()


def _format_flops_params(flops, params):
    def _num_to_str(n):
        if n is None: return "N/A"
        if n >= 1e12: return f"{n/1e12:.2f} TFLOPs"
        if n >= 1e9:  return f"{n/1e9:.2f} GFLOPs"
        if n >= 1e6:  return f"{n/1e6:.2f} MFLOPs"
        if n >= 1e3:  return f"{n/1e3:.2f} KFLOPs"
        return f"{n:.2f} FLOPs"
    def _p_to_str(n):
        if n is None: return "N/A"
        if n >= 1e9:  return f"{n/1e9:.2f} B"
        if n >= 1e6:  return f"{n/1e6:.2f} M"
        if n >= 1e3:  return f"{n/1e3:.2f} K"
        return f"{n:.2f}"
    return _num_to_str(flops), _p_to_str(params)


def profile_flops(model):

    try:
        from thop import profile
    except Exception as e:
        print("[FLOPs] thop is not installed or failed to import. Skip FLOPs profiling.")
        print(f"[FLOPs] Detail: {e}")
        return None, None

    model_cpu = model.cpu().eval()
    with torch.no_grad():
        x_evt = torch.randn(1, 2, 256, 256)
        x_img = torch.randn(1, 2, 256, 256)
        try:
            flops, params = profile(model_cpu, inputs=(x_img, x_evt), verbose=False)
            return flops, params
        except Exception as e:
            print("[FLOPs] profile() failed. Skip FLOPs profiling.")
            print(f"[FLOPs] Detail: {e}")
            return None, None


def _rgb_gather_by_index(rgb_stack, depth_index):

    B, N, C, H, W = rgb_stack.shape
    assert C == 3, "rgb_stack must have 3 channels."
    # idx for dim=1 gather
    idx = depth_index.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(B, 1, 3, H, W)  # [B,1,3,H,W]
    fused_rgb = torch.gather(rgb_stack, dim=1, index=idx).squeeze(1)  # [B,3,H,W]
    return fused_rgb


def _to_gray_if_rgb_stack(image_stack):

    if image_stack.dim() == 5 and image_stack.size(2) == 3:
        r = image_stack[:, :, 0, :, :]
        g = image_stack[:, :, 1, :, :]
        b = image_stack[:, :, 2, :, :]
        gray = 0.299 * r + 0.587 * g + 0.114 * b  # [B,N,H,W]
        return gray, image_stack
    else:
        return image_stack, None

def train(model, train_loader, criterion_depth, optimizer, device, loss_ratio, epoch):

    model.train()
    train_loss = 0.0
    loss_depth_total = 0.0

    progress_bar = tqdm(train_loader, desc="Training")
    for i, (image_stack, event_stack, depth_map_gt, stack_size) in enumerate(progress_bar):
        batch_start = time.time()

        # to device
        t0 = time.time()
        image_stack = image_stack.to(device)           # [B,N,H,W] 或 [B,N,3,H,W]
        event_stack = event_stack.to(device)           # [B,N,H,W]
        depth_map_gt = depth_map_gt.to(device)         # [B,1,H,W]
        t1 = time.time()
        data_time = t1 - t0

        image_stack_gray, _ = _to_gray_if_rgb_stack(image_stack)

        # forward/backward
        t2 = time.time()
        optimizer.zero_grad()
        _, depth_map, _ = model(image_stack, event_stack)  # EventStackMFF: (x_evt, x_img_gray)

        loss_depth = criterion_depth(depth_map, depth_map_gt)
        total_loss = loss_ratio[1] * loss_depth

        total_loss.backward()
        optimizer.step()
        t3 = time.time()
        compute_time = t3 - t2

        train_loss += total_loss.item()
        loss_depth_total += loss_depth.item()

        batch_time = time.time() - batch_start
        progress_bar.set_postfix({
            "Epoch": f"{epoch}",
            "batch_time": f"{batch_time:.2f}s",
            "data_time": f"{data_time:.2f}s",
            "compute_time": f"{compute_time:.2f}s",
            "total_loss": f"{total_loss.item():.6f}",
            "loss_depth": f"{loss_depth.item():.6f}",
        })

    return (train_loss / len(train_loader),
            loss_depth_total / len(train_loader))


def validate_dataset(model, val_loader, criterion_depth, device, epoch, save_path, loss_ratio):
    
    start_time = time.time()

    model.eval()
    val_loss = 0.0
    loss_depth_total = 0.0

    depth_mse_metric = MeanSquaredError().to(device)
    depth_mae_metric = MeanAbsoluteError().to(device)

    total_depth_mse = 0.0
    total_depth_mae = 0.0

    progress_bar = tqdm(val_loader, desc=f"Validating Epoch {epoch}")

    with torch.no_grad():
        for i, (image_stack, event_stack, depth_map_gt, stack_size) in enumerate(progress_bar):
            image_stack = image_stack.to(device)       # [B,N,H,W] or [B,N,3,H,W]
            event_stack = event_stack.to(device)       # [B,N,H,W]
            depth_map_gt = depth_map_gt.to(device)     # [B,1,H,W]

            image_stack_gray, image_stack_rgb = _to_gray_if_rgb_stack(image_stack)

            fused_image, depth_map, depth_map_index = model(image_stack, event_stack)

            loss_depth = criterion_depth(depth_map, depth_map_gt)
            total_loss = loss_ratio[1] * loss_depth

            val_loss += total_loss.item()
            loss_depth_total += loss_depth.item()

            depth_mse = depth_mse_metric(depth_map, depth_map_gt)
            depth_mae = depth_mae_metric(depth_map, depth_map_gt)
            total_depth_mse += depth_mse.item()
            total_depth_mae += depth_mae.item()

            progress_bar.set_postfix({
                "Total Loss": f"{total_loss.item():.6f}",
                "Depth MSE": f"{depth_mse.item():.6f}",
                "Depth MAE": f"{depth_mae.item():.6f}"
            })

            if i == len(val_loader) - 1:
                visualization_path = os.path.join(save_path, f'validation_visualization/epoch_{epoch}')
                os.makedirs(visualization_path, exist_ok=True)

                to_image(depth_map_gt, epoch, 'depth_map_gt', visualization_path)
                to_image(depth_map, epoch, 'depth_map', visualization_path)

                if image_stack.dim() == 5:
                    N = image_stack.size(1)
                else:
                    N = image_stack.size(1)
                disp = depth_map_index.float() / max(1, (N - 1))
                disp = disp.unsqueeze(1)  # [B,1,H,W]
                to_image(disp, epoch, 'depth_index_vis', visualization_path)

                to_image(fused_image, epoch, 'fused_image', visualization_path)

    elapsed = time.time() - start_time
    print(f"Epoch {epoch} Validation Time: {elapsed:.2f} seconds")

    num_batches = len(val_loader)
    avg_depth_mse = total_depth_mse / num_batches
    avg_depth_mae = total_depth_mae / num_batches

    return (val_loss / num_batches,
            loss_depth_total / num_batches,
            avg_depth_mse, avg_depth_mae)


def main():
    args = parse_args()
    model_save_path = config_model_dir(resume=False, subdir_name=args.save_name)

    train_dataset_params = []
    if args.use_train_dataset_1:
        train_dataset_params.append({
            'img_root': args.train_img_stack,
            'evt_root': args.train_evt_stack,
            'continuous_depth_dir': args.train_depth_continuous,
            'subset_fraction': args.subset_fraction_train
        })
    if args.use_train_dataset_2:
        train_dataset_params.append({
            'img_root': args.train_img_stack_2,
            'evt_root': args.train_evt_stack_2,
            'continuous_depth_dir': args.train_depth_continuous_2,
            'subset_fraction': args.subset_fraction_train
        })

    if args.use_train_dataset_3:
        train_dataset_params.append({
            'img_root': args.train_img_stack_3,
            'evt_root': args.train_evt_stack_3,
            'continuous_depth_dir': args.train_depth_continuous_3,
            'subset_fraction': args.subset_fraction_train
        })

    train_loader = get_event_dataloader(
        train_dataset_params,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        target_size=args.training_image_size
    ) if train_dataset_params else None

    val_loaders = []
    if args.use_val_dataset_1:
        val_loader_1 = get_event_dataloader(
            [{
                'img_root': args.val_img_stack,
                'evt_root': args.val_evt_stack,
                'continuous_depth_dir': args.val_depth_continuous,
                'subset_fraction': args.subset_fraction_val
            }],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=False,
            target_size=args.training_image_size
        )
        val_loaders.append(val_loader_1)

    if args.use_val_dataset_2:
        val_loader_2 = get_event_dataloader(
            [{
                'img_root': args.val_img_stack_2,
                'evt_root': args.val_evt_stack_2,
                'continuous_depth_dir': args.val_depth_continuous_2,
                'subset_fraction': args.subset_fraction_val
            }],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=False,
            target_size=args.training_image_size
        )
        val_loaders.append(val_loader_2)

    if args.use_val_dataset_3:
        val_loader_3 = get_event_dataloader(
            [{
                'img_root': args.val_img_stack_3,
                'evt_root': args.val_evt_stack_3,
                'continuous_depth_dir': args.val_depth_continuous_3,
                'subset_fraction': args.subset_fraction_val
            }],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=False,
            target_size=args.training_image_size
        )
        val_loaders.append(val_loader_3)

    print(f"Training samples: {len(train_loader.dataset) if train_loader else 0}")
    for i, val_loader in enumerate(val_loaders, 1):
        print(f"Validation samples (Dataset {i}): {len(val_loader.dataset)}")

    model = EventStackMFF()

    flops, params_thop = profile_flops(model)
    if flops is not None and params_thop is not None:
        flops_str, params_str = _format_flops_params(flops, params_thop)
        print(f"{flops_str}, Params: {params_str}")
    else:
        print("[FLOPs] Skipped.")

    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model.to(device)
    model = nn.DataParallel(model)

    criterion_depth = torch.nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)

    best_val_loss = float('inf')
    start_time = time.time()
    val_results_data = []

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")

        if train_loader:
            train_loss, train_depth_loss = train(model, train_loader, criterion_depth, optimizer, device, args.loss_ratio, epoch)

        val_results = []
        epoch_val_data = {'epoch': epoch + 1}
        for i, val_loader in enumerate(val_loaders, 1):
            results = validate_dataset(model, val_loader, criterion_depth, device, epoch,
                                       os.path.join(model_save_path, f'val_dataset_{i}'), args.loss_ratio)
            val_results.append(results)
            (val_loss, val_depth_loss, avg_depth_mse, avg_depth_mae) = results

            epoch_val_data.update({
                f'val_dataset_{i}_loss': val_loss,
                f'val_dataset_{i}_depth_loss': val_depth_loss,
                f'val_dataset_{i}_depth_mse': avg_depth_mse,
                f'val_dataset_{i}_depth_mae': avg_depth_mae
            })

            print(f"Validation Dataset {i} Results:")
            print(f"  Loss: {val_loss:.6f}")
            print(f"  Depth MSE: {avg_depth_mse:.6f}")
            print(f"  Depth MAE: {avg_depth_mae:.6f}")

        if train_loader:
            epoch_val_data.update({
                'train_loss': train_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })

        val_results_data.append(epoch_val_data)
        val_results_df = pd.DataFrame(val_results_data)
        os.makedirs(model_save_path, exist_ok=True)
        val_results_df.to_csv(os.path.join(model_save_path, 'validation_results.csv'), index=False)

        os.makedirs(os.path.join(model_save_path, 'model_save'), exist_ok=True)
        sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(sd, f"{os.path.join(model_save_path, 'model_save')}/epoch_{epoch}.pth")

        if val_loaders:
            avg_val_loss = sum(res[0] for res in val_results) / len(val_results)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(sd, f"{model_save_path}/best_model.pth")
                print(f"Saved new best model with validation loss: {best_val_loss:.6f}")

        scheduler.step()

    end_time = time.time()
    training_time_hours = (end_time - start_time) / 3600
    print(f"Training completed in {training_time_hours:.2f} hours")
    print(f"Model saved at: {model_save_path}")


if __name__ == "__main__":
    main()
