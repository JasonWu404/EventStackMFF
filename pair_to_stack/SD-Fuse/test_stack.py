#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from network import SD_Fuse_Net, ResT, build_structure_extractor


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image_file(p: str) -> bool:
    return os.path.splitext(p.lower())[1] in IMG_EXTS


def _last_int_in_name(fname: str) -> Optional[int]:

    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r"(\d+)(?!.*\d)", base)
    if m is None:
        return None
    return int(m.group(1))


def list_scene_dirs(root_dir: str) -> List[str]:

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"root_dir not found: {root_dir}")

    scene_dirs = []
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if not os.path.isdir(p):
            continue

        has_image = any(_is_image_file(f) for f in os.listdir(p))
        if has_image:
            scene_dirs.append(p)

    return scene_dirs


def list_images_in_scene(scene_dir: str, order: str = "asc") -> List[str]:

    files = [os.path.join(scene_dir, f) for f in os.listdir(scene_dir) if _is_image_file(f)]

    def sort_key(p: str):
        n = _last_int_in_name(p)
        if n is None:
            return (1, os.path.basename(p).lower())
        return (0, n)

    files = sorted(files, key=sort_key)

    if order == "desc":
        files = files[::-1]

    return files


def pil_to_tensor_rgb(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)
    return t


def tensor_to_pil_rgb(x: torch.Tensor) -> Image.Image:
    """
    x: (3,H,W) in [0,1]
    """
    x = x.detach().clamp(0, 1).cpu()
    arr = (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def tensor_to_pil_gray(x: torch.Tensor) -> Image.Image:
    """
    x: (1,H,W) or (H,W), in [0,1]
    """
    if x.dim() == 3:
        x = x[0]
    x = x.detach().clamp(0, 1).cpu()
    arr = (x.numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def pad_to_multiple(x: torch.Tensor, mult: int = 16):

    _, _, h, w = x.shape
    pad_h = (mult - (h % mult)) % mult
    pad_w = (mult - (w % mult)) % mult
    pt, pb = 0, pad_h
    pl, pr = 0, pad_w
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    x = F.pad(x, (pl, pr, pt, pb), mode="reflect")
    return x, (pl, pr, pt, pb)


def unpad(x: torch.Tensor, pad):

    pl, pr, pt, pb = pad
    if (pl, pr, pt, pb) == (0, 0, 0, 0):
        return x
    _, _, h, w = x.shape
    return x[:, :, pt:h - pb, pl:w - pr]


def load_model(ckpt_path: str, device: torch.device,
               extractor_name: Optional[str] = None,
               out_stage: Optional[int] = None,
               r: Optional[int] = None,
               lam: Optional[float] = None,
               strict: bool = True) -> SD_Fuse_Net:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    extractor_name = extractor_name or ckpt_args.get("extractor", "sobel")
    out_stage = int(out_stage if out_stage is not None else ckpt_args.get("out_stage", 2))
    r = int(r if r is not None else ckpt_args.get("r", 5))
    lam = float(lam if lam is not None else ckpt_args.get("lam", 0.5))

    extractor = build_structure_extractor(extractor_name)
    transformer = ResT(in_chans=1, out_stage=out_stage)

    model = SD_Fuse_Net(
        extractor=extractor,
        transformer=transformer,
        r=r,
        lam=lam,
        num_marm=6,
        guidance_from="mean",
        transformer_in_channels=1,
    )

    state = None
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
    if state is None:
        state = ckpt

    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError:
        if strict:
            raise
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("[WARN] load_state_dict(strict=False)")
        print("  missing keys:", missing)
        print("  unexpected keys:", unexpected)

    model.to(device)
    model.eval()
    return model


def infer_dm(model: torch.nn.Module, near: torch.Tensor, far: torch.Tensor,
             pad_mult: int = 16) -> torch.Tensor:

    near_p, pad = pad_to_multiple(near, mult=pad_mult)
    far_p, _ = pad_to_multiple(far, mult=pad_mult)

    out = model(near_p, far_p)

    if out.dim() == 3:
        out = out.unsqueeze(1)
    if out.size(1) != 1:
        out = out.mean(dim=1, keepdim=True)

    out_min = float(out.min())
    out_max = float(out.max())
    if out_min < -1e-3 or out_max > 1.0 + 1e-3:
        dm = torch.sigmoid(out)
    else:
        dm = out.clamp(0.0, 1.0)

    dm = unpad(dm, pad)
    return dm


def fuse(near: torch.Tensor, far: torch.Tensor, dm: torch.Tensor) -> torch.Tensor:

    return dm * near + (1.0 - dm) * far


def progressive_fuse_stack(model,
                           image_paths: List[str],
                           device: torch.device,
                           pad_mult: int = 16,
                           save_intermediate: bool = False,
                           inter_dir: Optional[str] = None,
                           save_dm: bool = False) -> torch.Tensor:

    if len(image_paths) == 0:
        raise RuntimeError("Empty image stack.")
    if len(image_paths) == 1:
        img = Image.open(image_paths[0]).convert("RGB")
        fused_t = pil_to_tensor_rgb(img).unsqueeze(0).to(device)
        return fused_t

    img0 = Image.open(image_paths[0]).convert("RGB")
    fused_t = pil_to_tensor_rgb(img0).unsqueeze(0).to(device)
    base_size = img0.size

    with torch.no_grad():
        for i in range(1, len(image_paths)):
            p_cur = image_paths[i]
            img_cur = Image.open(p_cur).convert("RGB")

            if img_cur.size != base_size:
                raise RuntimeError(
                    f"Size mismatch in stack: first={base_size}, current={img_cur.size}, file={p_cur}"
                )

            cur_t = pil_to_tensor_rgb(img_cur).unsqueeze(0).to(device)

            dm = infer_dm(model, fused_t, cur_t, pad_mult=pad_mult)
            fused_t = fuse(fused_t, cur_t, dm).detach()

            if save_intermediate and inter_dir is not None:
                os.makedirs(inter_dir, exist_ok=True)
                out_step = os.path.join(inter_dir, f"step_{i:02d}_fused.png")
                tensor_to_pil_rgb(fused_t[0]).save(out_step)

                if save_dm:
                    out_dm = os.path.join(inter_dir, f"step_{i:02d}_dm.png")
                    tensor_to_pil_gray(dm[0]).save(out_dm)

    return fused_t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True,
                    help="总目录，里面包含多个场景子文件夹")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="checkpoint .pt")
    ap.add_argument("--out_root", type=str, required=True,
                    help="输出总目录，每个场景输出一张融合结果")

    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--extractor", type=str, default=None, help="override extractor")
    ap.add_argument("--out_stage", type=int, default=None, help="override ResT out_stage")
    ap.add_argument("--r", type=int, default=None, help="override SGF radius r")
    ap.add_argument("--lam", type=float, default=None, help="override SGF lambda lam")
    ap.add_argument("--pad_mult", type=int, default=16, help="pad H/W to multiple of this")

    ap.add_argument("--order", type=str, default="asc", choices=["asc", "desc"],
                    help="栈图像排序方式：asc=数字从小到大，desc=反过来")
    ap.add_argument("--max_scenes", type=int, default=0, help="0表示处理全部场景")
    ap.add_argument("--save_intermediate", action="store_true",
                    help="是否保存逐步融合中间结果")
    ap.add_argument("--save_dm", action="store_true",
                    help="是否保存每一步的决策图")

    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("device:", device)

    model = load_model(
        ckpt_path=args.ckpt,
        device=device,
        extractor_name=args.extractor,
        out_stage=args.out_stage,
        r=args.r,
        lam=args.lam,
        strict=True,
    )

    scene_dirs = list_scene_dirs(args.root_dir)
    if args.max_scenes > 0:
        scene_dirs = scene_dirs[:args.max_scenes]

    if len(scene_dirs) == 0:
        raise RuntimeError("No valid scene folders found under root_dir.")

    print(f"found scenes: {len(scene_dirs)}")

    for sidx, scene_dir in enumerate(scene_dirs, start=1):
        scene_name = os.path.basename(scene_dir.rstrip("/\\"))
        print(f"\n[{sidx}/{len(scene_dirs)}] processing scene: {scene_name}")

        image_paths = list_images_in_scene(scene_dir, order=args.order)
        if len(image_paths) == 0:
            print(f"[SKIP] no images in {scene_dir}")
            continue
        if len(image_paths) == 1:
            print(f"[WARN] only one image in {scene_dir}, output will be the image itself")

        print(f"  stack size: {len(image_paths)}")

        scene_out_dir = os.path.join(args.out_root, scene_name)
        os.makedirs(scene_out_dir, exist_ok=True)

        inter_dir = None
        if args.save_intermediate:
            inter_dir = os.path.join(scene_out_dir, "intermediate")

        try:
            fused_t = progressive_fuse_stack(
                model=model,
                image_paths=image_paths,
                device=device,
                pad_mult=args.pad_mult,
                save_intermediate=args.save_intermediate,
                inter_dir=inter_dir,
                save_dm=args.save_dm,
            )

            out_final = os.path.join(scene_out_dir, f"{scene_name}_fused.png")
            tensor_to_pil_rgb(fused_t[0]).save(out_final)

            print(f"[OK] saved: {out_final}")

        except Exception as e:
            print(f"[FAIL] scene={scene_name}, reason={e}")

    print("\ndone.")


if __name__ == "__main__":
    main()