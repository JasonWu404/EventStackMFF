import os
import argparse
import collections
import json

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.utils import save_image
import pytorch_lightning as pl

from models.models import MODELS

def ensure_dir(path: str):

    if path is None or path == "":
        return
    if not os.path.exists(path):
        os.makedirs(path)


def to_tensor_rgb(img_pil: Image.Image) -> torch.Tensor:

    img_pil = img_pil.convert("RGB")
    arr = np.array(img_pil).astype(np.float32) / 255.0  # H x W x 3
    arr = np.transpose(arr, (2, 0, 1))  # 3 x H x W
    tensor = torch.from_numpy(arr)  # 3 x H x W
    return tensor.unsqueeze(0)  # 1 x 3 x H x W


def postprocess_mask(score_map: np.ndarray, area_ratio: float = 0.01) -> np.ndarray:

    h, w = score_map.shape
    flat = score_map.reshape(-1)

    k = max(int(len(flat) * area_ratio), 1)

    idx_topk = np.argpartition(flat, -k)[-k:]
    thresh = flat[idx_topk].min()

    mask = (score_map >= thresh).astype(np.uint8)  # 0/1
    return mask.reshape(h, w)

class CoolSystem(pl.LightningModule):
    def __init__(self, config: dict, ckpt_path: str = None,
                 device: torch.device = torch.device("cpu")):

        super(CoolSystem, self).__init__()

        self.config = config
        self.device_used = device

        self.save_path_eval = "./save_images"
        ensure_dir(self.save_path_eval)

        fusion_net_name = config.get("fusion_net", "FusionNet")
        rec_net_name = config.get("rec_net", "RecNet")

        self.FusionNet = MODELS[fusion_net_name](config)
        self.RecNet = MODELS[rec_net_name](config)

        if ckpt_path is not None:
            print(f"Loading FusionNet weights from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)

            new_state_dict_f = collections.OrderedDict()
            new_state_dict_r = collections.OrderedDict()

            for k, v in state_dict.items():
                # FusionNet.*
                if k.startswith("FusionNet."):
                    name = k[len("FusionNet."):]
                    new_state_dict_f[name] = v
                # RecNet.*
                if k.startswith("RecNet."):
                    name = k[len("RecNet."):]
                    new_state_dict_r[name] = v

            self.FusionNet.load_state_dict(new_state_dict_f, strict=True)
            if len(new_state_dict_r) > 0:
                self.RecNet.load_state_dict(new_state_dict_r, strict=True)
        else:
            print("Warning: ckpt_path is None, FusionNet will use random init.")

        self.FusionNet.to(self.device_used)
        self.FusionNet.eval()

        self.RecNet.to(self.device_used)
        self.RecNet.eval()

    def fuse_pair(self,
                  model: nn.Module,
                  img_a_pil: Image.Image,
                  img_b_pil: Image.Image,
                  device: torch.device,
                  area_ratio: float = 0.01):

        img_a_pil = img_a_pil.convert("RGB")
        img_b_pil = img_b_pil.convert("RGB")

        if img_a_pil.size != img_b_pil.size:
            raise ValueError(f"图像尺寸不一致: {img_a_pil.size} vs {img_b_pil.size}")

        x = to_tensor_rgb(img_a_pil).float().to(device)  # 1 x 3 x H x W
        y = to_tensor_rgb(img_b_pil).float().to(device)  # 1 x 3 x H x W

        with torch.no_grad():

            out_xy = model(x, y)      # 1 x 3 x H x W
            out_yx = model(y, x)      # 1 x 3 x H x W
            out = (out_xy + out_yx) / 2.0
            out = torch.clamp(out, 0.0, 1.0)

        fused = out.squeeze(0).detach().cpu().numpy()  # 3 x H x W
        fused = np.transpose(fused, (1, 2, 0))         # H x W x 3
        fused = np.clip(fused * 255.0, 0, 255).astype(np.uint8)

        fused_pil = Image.fromarray(fused)

        gray = np.dot(fused[..., :3], [0.299, 0.587, 0.114]) / 255.0  # H x W
        mask = postprocess_mask(gray.astype(np.float32), area_ratio=area_ratio)
        mask_uint8 = (mask * 255).astype(np.uint8)

        return fused_pil, mask_uint8

    def fuse_stack_recursive(self,
                             model: nn.Module,
                             image_paths,
                             device: torch.device,
                             area_ratio: float = 0.01):

        fused_img = Image.open(image_paths[0]).convert("RGB")

        for idx in range(1, len(image_paths)):
            next_img = Image.open(image_paths[idx]).convert("RGB")
            fused_img, _ = self.fuse_pair(
                model=model,
                img_a_pil=fused_img,
                img_b_pil=next_img,
                device=device,
                area_ratio=area_ratio,
            )

        return fused_img

def process_all_stacks(input_root,
                       output_root,
                       model_path,
                       config,
                       area_ratio=0.01,
                       save_intermediate=False,
                       gpu="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")

    system = CoolSystem(config=config, ckpt_path=model_path, device=device)
    fusion_model = system.FusionNet

    stack_folders = [
        os.path.join(input_root, folder)
        for folder in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, folder))
    ]
    if len(stack_folders) == 0:
        print(f"[Error] 在 {input_root} 下没有找到有效的图像栈文件夹")
        return

    print(f"Found {len(stack_folders)} stack folder(s).")

    for stack_idx, stack_folder in enumerate(stack_folders, 1):
        image_paths = sorted([
            os.path.join(stack_folder, f)
            for f in os.listdir(stack_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".bmp"))
        ])
        if len(image_paths) < 2:
            print(f"[Warning] {stack_folder} 中有效图片数 < 2，跳过。")
            continue

        rel_path = os.path.relpath(stack_folder, input_root)
        out_dir = os.path.join(output_root, rel_path)
        ensure_dir(out_dir)

        print(f"[{stack_idx}/{len(stack_folders)}] Processing: {stack_folder}")
        print(f"    Number of images: {len(image_paths)}")

        if save_intermediate:
            intermediate_dir = os.path.join(out_dir, "intermediate")
            ensure_dir(intermediate_dir)

        fused_img = system.fuse_stack_recursive(
            model=fusion_model,
            image_paths=image_paths,
            device=device,
            area_ratio=area_ratio,
        )

        stack_name = os.path.basename(stack_folder.rstrip("/\\"))
        save_path = os.path.join(out_dir, f"{stack_name}_fused.png")
        fused_img.save(save_path)

        print(f"    Saved: {save_path}")

    print("===> Finished stack fusion!")

def main():
    parser = argparse.ArgumentParser(description="Image Stack Fusion with Fusion2Void FusionNet")
    parser.add_argument("--input_root", type=str, required=True,
                        help="Path to the root folder containing all image stacks")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Path to save the fused images")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint, e.g. ./ckpt/model_weight.ckpt")
    parser.add_argument("--gpu", type=str, default="0",
                        help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Save intermediate fusion results")
    parser.add_argument("--config", type=str, default="./configs/Eval_MMnet.json",
                        help="Path to the config json (默认使用官方 Eval_MMnet.json)")

    args = parser.parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    with open(args.config, "r") as f:
        config = json.load(f)

    process_all_stacks(
        input_root=args.input_root,
        output_root=args.output_root,
        model_path=args.model_path,
        config=config,
        gpu=args.gpu,
        save_intermediate=args.save_intermediate
    )

if __name__ == "__main__":
    main()