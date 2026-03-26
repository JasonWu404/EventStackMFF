import os
import re
import argparse
import numpy as np
from PIL import Image
from skimage import morphology

import torch
from torchvision import transforms

from CCSR_Net import CCSR_Net


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTS


def natural_key(s):

    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def find_stack_folders(root):

    stack_folders = []
    for current_root, _, files in os.walk(root):
        image_files = [f for f in files if is_image_file(f)]
        if len(image_files) >= 2:
            stack_folders.append(current_root)
    stack_folders.sort(key=natural_key)
    return stack_folders


def load_model(model_path, device):
    model = CCSR_Net().to(device)

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)

    model.eval()
    return model


def to_tensor_rgb(pil_img):
    return transforms.ToTensor()(pil_img).unsqueeze(0)


def postprocess_mask(prob_map, area_ratio=0.01):

    mask = (prob_map > 0.5).astype(np.uint8)

    h, w = mask.shape
    area_threshold = max(1, int(area_ratio * h * w))


    tmp = morphology.remove_small_holes(mask == 0, area_threshold=area_threshold)
    tmp = np.where(tmp, 0, 1)
    tmp = morphology.remove_small_holes(tmp == 1, area_threshold=area_threshold)
    mask_final = np.where(tmp, 1, 0).astype(np.float32)

    return mask_final


@torch.no_grad()
def fuse_pair(model, img_a_pil, img_b_pil, device, area_ratio=0.01):

    img_a_pil = img_a_pil.convert('RGB')
    img_b_pil = img_b_pil.convert('RGB')

    if img_a_pil.size != img_b_pil.size:
        raise ValueError(f"图像尺寸不一致: {img_a_pil.size} vs {img_b_pil.size}")

    x = to_tensor_rgb(img_a_pil).float().to(device)
    y = to_tensor_rgb(img_b_pil).float().to(device)

    d = model(x, y)                 # [1, 1, H, W]
    d = torch.sigmoid(d)
    d = d.squeeze().detach().cpu().numpy()   # H x W

    mask = postprocess_mask(d, area_ratio=area_ratio)  # H x W, 0/1

    img_a = np.array(img_a_pil).astype(np.float32) / 255.0
    img_b = np.array(img_b_pil).astype(np.float32) / 255.0

    mask_3c = np.expand_dims(mask, axis=2)   # H x W x 1
    fused = img_a * mask_3c + img_b * (1.0 - mask_3c)
    fused = np.clip(fused * 255.0, 0, 255).astype(np.uint8)

    mask_uint8 = (mask * 255).astype(np.uint8)

    fused_pil = Image.fromarray(fused)
    return fused_pil, mask_uint8


def get_sorted_image_paths(folder):
    image_files = [f for f in os.listdir(folder) if is_image_file(f)]
    image_files.sort(key=natural_key)
    return [os.path.join(folder, f) for f in image_files]


def fuse_stack_recursive(model, image_paths, device, area_ratio=0.01, save_intermediate=False, intermediate_dir=None):

    if len(image_paths) < 2:
        raise ValueError("图像栈至少需要两张图像")

    fused_img = Image.open(image_paths[0]).convert('RGB')

    for idx in range(1, len(image_paths)):
        next_img = Image.open(image_paths[idx]).convert('RGB')
        fused_img, mask_uint8 = fuse_pair(
            model=model,
            img_a_pil=fused_img,
            img_b_pil=next_img,
            device=device,
            area_ratio=area_ratio
        )

        if save_intermediate and intermediate_dir is not None:
            os.makedirs(intermediate_dir, exist_ok=True)
            fused_img.save(os.path.join(intermediate_dir, f"step_{idx:02d}_fused.tif"))
            Image.fromarray(mask_uint8).save(os.path.join(intermediate_dir, f"step_{idx:02d}_mask.tif"))

    return fused_img


def process_all_stacks(input_root, output_root, model_path, area_ratio=0.01, save_intermediate=False, gpu='0'):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')
    print(f'Loading model from: {model_path}')
    model = load_model(model_path, device)

    stack_folders = find_stack_folders(input_root)
    if len(stack_folders) == 0:
        print(f'[Error] 在 {input_root} 下没有找到有效的图像栈文件夹')
        return

    print(f'Found {len(stack_folders)} stack folder(s).')

    for stack_idx, stack_folder in enumerate(stack_folders, 1):
        image_paths = get_sorted_image_paths(stack_folder)
        if len(image_paths) < 2:
            continue

        rel_path = os.path.relpath(stack_folder, input_root)
        out_dir = os.path.join(output_root, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        print(f'[{stack_idx}/{len(stack_folders)}] Processing: {stack_folder}')
        print(f'    Number of images: {len(image_paths)}')

        intermediate_dir = None
        if save_intermediate:
            intermediate_dir = os.path.join(out_dir, 'intermediate')

        fused_img = fuse_stack_recursive(
            model=model,
            image_paths=image_paths,
            device=device,
            area_ratio=area_ratio,
            save_intermediate=save_intermediate,
            intermediate_dir=intermediate_dir
        )

        stack_name = os.path.basename(stack_folder.rstrip('/\\'))
        save_path = os.path.join(out_dir, f'{stack_name}_CCSR_stack_fused.tif')
        fused_img.save(save_path)

        print(f'    Saved: {save_path}')

    print('===> Finished stack fusion!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, default='yourinput',
                        help='输入根目录，内部包含多个场景文件夹/图像栈文件夹')
    parser.add_argument('--output_root', type=str, default='youroutput',
                        help='输出目录')
    parser.add_argument('--model_path', type=str, default='models/CCSR.pth',
                        help='模型权重路径')
    parser.add_argument('--area_ratio', type=float, default=0.01,
                        help='小区域后处理比例')
    parser.add_argument('--gpu', type=str, default='0',
                        help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='是否保存每一步递归融合的中间结果')

    args = parser.parse_args()

    process_all_stacks(
        input_root=args.input_root,
        output_root=args.output_root,
        model_path=args.model_path,
        area_ratio=args.area_ratio,
        save_intermediate=args.save_intermediate,
        gpu=args.gpu
    )