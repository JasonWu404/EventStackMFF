import os
import re
from typing import List

import torch
from torch.nn import functional as F
from tqdm import tqdm
from torchvision.transforms import transforms, ToPILImage
from PIL import Image

import network
from utils import guide_filter
# from thop import profile

EPSILON = 1e-10
img_size = 224


def topatch(img: Image.Image) -> List[Image.Image]:

    w, h = img.size
    tensor_img = transforms.ToTensor()(img)  # [1,H,W]
    pad = torch.nn.ReflectionPad2d([
        0, img_size - w % img_size,
        0, img_size - h % img_size
    ])
    tensor_img = pad(tensor_img)
    img_pad = ToPILImage()(tensor_img)

    nh = h // img_size + 1
    nw = w // img_size + 1

    cis = []
    for j in range(nh):
        for i in range(nw):
            area = (img_size * i, img_size * j,
                    img_size * (i + 1), img_size * (j + 1))
            cropped_img = img_pad.crop(area)
            cis.append(cropped_img)
    return cis

def fuse_two_Y_images(
    Y1: Image.Image,
    Y2: Image.Image,
    fuse_model,
    device: str = "cuda"
) -> Image.Image:

    assert Y1.size == Y2.size, "两张图尺寸必须一致才能融合"

    w, h = Y1.size
    patches1 = topatch(Y1)
    patches2 = topatch(Y2)

    nh = h // img_size + 1
    nw = w // img_size + 1

    fused_patches = []

    with torch.no_grad():
        for idx in range(nh * nw):
            img1_patch = transforms.ToTensor()(patches1[idx]).to(device)  # [1,H,W]
            img2_patch = transforms.ToTensor()(patches2[idx]).to(device)

            img1_patch = img1_patch.unsqueeze(0)  # [1,1,H,W]
            img2_patch = img2_patch.unsqueeze(0)

            _ = guide_filter(I=img1_patch, p=img1_patch, window_size=11, eps=0.2)
            _ = guide_filter(I=img2_patch, p=img2_patch, window_size=11, eps=0.2)

            x, y = fuse_model.forward_encoder(img1_patch, img2_patch)
            c_in_x, c_in_y, common, positive_x, nagetive_x, positive_y, nagetive_y, pred = fuse_model.forward_decoder(x, y)

            pred = fuse_model.unpatchify(pred)
            pred = torch.clamp(pred, min=0.0, max=1.0)

            pred = pred.detach().cpu().squeeze(0).squeeze(0)  # [H,W]
            fused_patch = ToPILImage()(pred)
            fused_patches.append(fused_patch)

    fused_Y = Image.new("L", (img_size * nw, img_size * nh))
    index_cis = 0
    for j in range(nh):
        for i in range(nw):
            fused_Y.paste(fused_patches[index_cis],
                          (img_size * i, img_size * j))
            index_cis += 1

    fused_Y = fused_Y.crop((0, 0, w, h))
    return fused_Y


def fuse_stack_scene(
    image_paths: List[str],
    fuse_model,
    device: str = "cuda"
) -> Image.Image:

    assert len(image_paths) >= 2, "至少需要两张图像才能做融合"

    def sort_key(p):
        name = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"\d+", name)
        return int(m.group()) if m else name

    image_paths = sorted(image_paths, key=sort_key)

    img0 = Image.open(image_paths[0]).convert("YCbCr")
    Y0, Cb0, Cr0 = img0.split()
    fused_Y = Y0

    for path in tqdm(image_paths[1:], desc="  Fusing stack", leave=False):
        img = Image.open(path).convert("YCbCr")
        Y, Cb, Cr = img.split()

        fused_Y = fuse_two_Y_images(fused_Y, Y, fuse_model, device=device)

    fused_img_ycbcr = Image.merge("YCbCr", (fused_Y, Cb0, Cr0))
    fused_img_rgb = fused_img_ycbcr.convert("RGB")
    return fused_img_rgb


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_ROOT = "yourinput"

    # 输出目录
    SAVE_ROOT = os.path.join("fusedata", os.path.basename(DATA_ROOT.rstrip("/")))
    os.makedirs(SAVE_ROOT, exist_ok=True)

    fuse_model = network.__dict__["mae_vit_large_patch16"](norm_pix_loss=False)
    fuse_model.mode = "fuse"

    state_dict = torch.load("weights/best_fusion.pt", map_location=device)
    fuse_model.load_state_dict(state_dict["weight"])
    fuse_model.to(device)
    fuse_model.eval()

    total = sum(p.numel() for p in fuse_model.parameters())
    print("Number of params: {:.2f} M".format(total / 1e6))
    print("Model epoch: {}".format(state_dict.get("epoch", -1)))

    scene_dirs = [
        os.path.join(DATA_ROOT, d)
        for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ]
    scene_dirs = sorted(scene_dirs)

    if not scene_dirs:
        print(f"[Error] 在 {DATA_ROOT} 下没有找到任何场景文件夹")
        return

    print(f"Found {len(scene_dirs)} scenes under {DATA_ROOT}")

    for idx, scene_dir in enumerate(scene_dirs):
        scene_name = os.path.basename(scene_dir.rstrip("/"))
        print(f"\nProcessing scene {idx + 1}/{len(scene_dirs)}: {scene_name}")

        img_files = [
            f for f in os.listdir(scene_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]
        if not img_files:
            print(f"  [Warning] {scene_dir} 内没有找到图片，跳过")
            continue

        image_paths = [os.path.join(scene_dir, f) for f in img_files]

        with torch.no_grad():
            fused_img = fuse_stack_scene(image_paths, fuse_model, device=device)

        save_path = os.path.join(SAVE_ROOT, f"{scene_name}.jpg")
        fused_img.save(save_path)
        print(f"  Saved fused image to: {save_path}")

if __name__ == "__main__":
    main()
