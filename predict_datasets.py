# -*- coding: utf-8 -*-
# @Author  : Juntao Wu, XinZhe Xie
# @University  : University of Science and Technology of China, ZheJiang University

import argparse
import os
import re
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

from network import EventStackMFF

def sort_key(name: str):

    nums = re.findall(r"\d+\.?\d*", name)
    return float(nums[0]) if nums else float("inf")


def list_images(d):
    return sorted(
        [f for f in os.listdir(d)
         if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=sort_key
    )


def ensure_same_hw(images_rgb):

    H0, W0 = images_rgb[0].shape[:2]
    out = []
    for im in images_rgb:
        if im.shape[:2] != (H0, W0):
            im = cv2.resize(im, (W0, H0), interpolation=cv2.INTER_LINEAR)
        out.append(im)
    return out, (H0, W0)


def gray_to_colormap(img01, cmap='rainbow'):
    img01 = np.clip(img01, 0, 1)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    mapr = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    return (mapr.to_rgba(img01)[:, :, :3] * 255).astype(np.uint8)


def resize_to_multiple_of_32(x):

    h, w = x.shape[-2:]
    nh = ((h - 1) // 32 + 1) * 32
    nw = ((w - 1) // 32 + 1) * 32
    return F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False), (h, w)


def create_color_fused_image(fused_gray01, depth_idx_hw, color_stack_rgb):

    H, W = fused_gray01.shape
    K = len(color_stack_rgb)
    depth_idx = np.clip(depth_idx_hw.astype(np.int32), 0, K - 1)

    color_arr = np.stack(color_stack_rgb, axis=0)  # uint8
    rr = np.arange(H)[:, None]
    cc = np.arange(W)[None, :]
    fused_rgb = color_arr[depth_idx, rr, cc]  # [H,W,3] RGB
    fused_bgr = cv2.cvtColor(fused_rgb, cv2.COLOR_RGB2BGR)
    return fused_bgr

def load_scene_stacks(image_dir, event_dir):

    img_files = list_images(image_dir)
    evt_files = list_images(event_dir)
    if len(img_files) == 0:
        raise RuntimeError(f"No images under: {image_dir}")
    if len(evt_files) == 0:
        raise RuntimeError(f"No events under: {event_dir}")

    color_list = []
    gray_list = []
    for f in img_files:
        p = os.path.join(image_dir, f)
        bgr = cv2.imread(p)
        if bgr is None:
            raise RuntimeError(f"Read fail: {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        color_list.append(rgb)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray_list.append(gray)

    color_list, (H0, W0) = ensure_same_hw(color_list)
    gray_list = [
        cv2.resize(g, (W0, H0), interpolation=cv2.INTER_LINEAR) if g.shape != (H0, W0) else g
        for g in gray_list
    ]

    evt_list = []
    for f in evt_files:
        p = os.path.join(event_dir, f)
        e = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if e is None:
            raise RuntimeError(f"Read fail: {p}")
        e = e.astype(np.float32) / 255.0

        if e.shape != (H0, W0):
            e = cv2.resize(e, (W0, H0), interpolation=cv2.INTER_LINEAR)
        evt_list.append(e)

    K = min(len(gray_list), len(evt_list))
    if len(gray_list) != len(evt_list):
        print(f"[Warn] frame mismatch: image={len(gray_list)} vs event={len(evt_list)}; use first {K}")

    gray_stack = torch.from_numpy(
        np.stack(gray_list[:K], axis=0).astype(np.float32)
    )  # [N,H,W]
    evt_stack = torch.from_numpy(
        np.stack(evt_list[:K], axis=0).astype(np.float32)
    )  # [N,H,W]
    color_stack = color_list[:K]  # list length K

    return gray_stack, evt_stack, color_stack, (H0, W0)

def infer_one_scene(model, img_stack, evt_stack, color_stack, device, dataset_name, scene_name):

    model.eval()
    with torch.no_grad():
        img = img_stack.unsqueeze(0).to(device)  # [1,N,H,W]
        evt = evt_stack.unsqueeze(0).to(device)  # [1,N,H,W]

        img_r, (H0, W0) = resize_to_multiple_of_32(img)
        evt_r, _ = resize_to_multiple_of_32(evt)

        out = model(img_r, evt_r, dataset_name=dataset_name, scene_name=scene_name)

        fused, depth01, depth_idx = out[0], out[1], out[2]  # [1,1,h,w], [1,1,h,w], [1,h,w]

        fused_01 = cv2.resize(
            fused.squeeze().float().cpu().numpy(),
            (W0, H0)
        )
        depth_01 = cv2.resize(
            depth01.squeeze().float().cpu().numpy(),
            (W0, H0)
        )
        depth_idx_hw = cv2.resize(
            depth_idx.squeeze().long().cpu().numpy().astype(np.int32),
            (W0, H0),
            interpolation=cv2.INTER_NEAREST
        )

        fused_color_bgr = create_color_fused_image(fused_01, depth_idx_hw, color_stack)
        depth_cmap_bgr = cv2.cvtColor(gray_to_colormap(depth_01), cv2.COLOR_RGB2BGR)
        return fused_01, depth_01, depth_idx_hw, fused_color_bgr, depth_cmap_bgr

def parse_args():
    ap = argparse.ArgumentParser("Batch inference for image/event roots")
    ap.add_argument("--image_root", required=True, type=str, help="图像栈根目录")
    ap.add_argument("--event_root", required=True, type=str, help="事件栈根目录")
    ap.add_argument("--model_path", required=True, type=str, help="模型权重 .pth")
    ap.add_argument("--output_root", default="./batch_output", type=str, help="输出根目录")
    ap.add_argument("--datasets", nargs="*", default=None, help="跑这些数据集名")
    return ap.parse_args()


def collect_scenes(root):
    out = {}
    if not os.path.isdir(root):
        return out

    entries = sorted(os.listdir(root))
    subdirs = [d for d in entries if os.path.isdir(os.path.join(root, d))]

    def has_images(d):
        return any(
            f.lower().endswith((".png", ".jpg", ".jpeg"))
            for f in os.listdir(d)
            if os.path.isfile(os.path.join(d, f))
        )

    scene_level_scenes = [d for d in subdirs if has_images(os.path.join(root, d))]
    if scene_level_scenes:
        parent = os.path.basename(os.path.dirname(root.rstrip("\\/")))
        dataset_name = parent if parent else os.path.basename(root.rstrip("\\/"))
        out[dataset_name] = sorted(scene_level_scenes)
        return out

    for dataset in sorted(subdirs):
        d_path = os.path.join(root, dataset)
        if not os.path.isdir(d_path):
            continue
        scenes = []
        for scene in sorted(os.listdir(d_path)):
            s_path = os.path.join(d_path, scene)
            if not os.path.isdir(s_path):
                continue
            if len(list_images(s_path)) > 0:
                scenes.append(scene)
        if scenes:
            out[dataset] = scenes
    return out

def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = EventStackMFF()
    state = torch.load(args.model_path, map_location=device)
    state = {
        k.replace("module.", ""): v
        for k, v in state.items()
        if not (k.endswith("total_ops") or k.endswith("total_params"))
    }
    model.load_state_dict(state, strict=False)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    img_tree = collect_scenes(args.image_root)
    evt_tree = collect_scenes(args.event_root)

    datasets = args.datasets if args.datasets else sorted(img_tree.keys())

    total_scenes = 0
    t0 = time.time()
    for dataset in datasets:
        if dataset not in img_tree:
            print(f"[Skip] dataset '{dataset}' not in image_root")
            continue
        if dataset not in evt_tree:
            print(f"[Skip] dataset '{dataset}' not in event_root")
            continue

        scenes = sorted(set(img_tree[dataset]).intersection(evt_tree[dataset]))
        if not scenes:
            print(f"[Skip] dataset '{dataset}' has no matched scenes")
            continue

        print(f"\n=== Dataset: {dataset} | Scenes: {len(scenes)} ===")
        for scene in scenes:
            cand_img = os.path.join(args.image_root, dataset, scene)
            cand_evt = os.path.join(args.event_root, dataset, scene)
            img_dir = cand_img if os.path.isdir(cand_img) else os.path.join(args.image_root, scene)
            evt_dir = cand_evt if os.path.isdir(cand_evt) else os.path.join(args.event_root, scene)

            if not (os.path.isdir(img_dir) and os.path.isdir(evt_dir)):
                print(f"[Skip] {dataset}/{scene}: missing dir -> img:{img_dir} evt:{evt_dir}")
                continue

            out_dir = os.path.join(args.output_root, dataset, scene)
            os.makedirs(out_dir, exist_ok=True)

            scene_prefix = re.sub(r"[^0-9A-Za-z_\-]+", "_", scene)

            try:
                gray_stack, evt_stack, color_stack, (H, W) = load_scene_stacks(img_dir, evt_dir)
                fused01, depth01, depth_idx, fused_bgr, depth_cmap_bgr = \
                    infer_one_scene(model, gray_stack, evt_stack, color_stack, device, dataset, scene)

                cv2.imwrite(
                    os.path.join(out_dir, f"{scene_prefix}_fused_gray.png"),
                    (fused01 * 255).astype(np.uint8)
                )
                cv2.imwrite(
                    os.path.join(out_dir, f"{scene_prefix}_depth_map.png"),
                    (depth01 * 255).astype(np.uint8)
                )
                cv2.imwrite(
                    os.path.join(out_dir, f"{scene_prefix}_color_fused.png"),
                    fused_bgr
                )
                cv2.imwrite(
                    os.path.join(out_dir, f"{scene_prefix}_depth_colormap.png"),
                    depth_cmap_bgr
                )

                np.save(
                    os.path.join(out_dir, f"{scene_prefix}_depth_index.npy"),
                    depth_idx.astype(np.int16)
                )

                total_scenes += 1
                print(f"[OK] {dataset}/{scene} -> {out_dir}")
            except Exception as e:
                print(f"[ERR] {dataset}/{scene}: {e}")

    dt = time.time() - t0
    print(f"\nDone. Processed {total_scenes} scenes in {dt:.2f}s (avg {dt/max(total_scenes,1):.3f}s/scene)")
    print(f"Outputs in: {args.output_root}")


if __name__ == "__main__":
    main()
