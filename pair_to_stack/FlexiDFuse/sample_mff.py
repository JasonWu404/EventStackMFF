from functools import partial
import os
import argparse
import yaml
import torch
import cv2
import numpy as np
from skimage.io import imsave
import warnings
import time
warnings.filterwarnings('ignore')

from guided_diffusion.models_DiT_Mamba import DiT_B_4
from guided_diffusion.gaussian_diffusion import create_sampler
from util.logger import get_logger


def image_read(path, mode='RGB'):
    img_bgr = cv2.imread(path).astype('float32')
    assert mode in ['RGB', 'GRAY', 'YCrCb'], 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    return img


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def preprocess_mff_image(img_or_path, target_hw, device, mode='GRAY'):
    """
    统一把输入（路径 or numpy）变成 [-1,1] 的 [1,1,H,W]，并 resize 到 target_hw
    """
    if isinstance(img_or_path, str):
        img = image_read(img_or_path, mode=mode)
    else:
        img = img_or_path

    img = np.array(img, dtype=np.float32)


    if img.ndim == 3:

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if img.max() > 1.0:
        img = img / 255.0

    # [H,W] -> [1,1,H,W]
    img = img[np.newaxis, np.newaxis, ...]
    img = torch.from_numpy(img).to(device)

    # [0,1] -> [-1,1]
    img = img * 2.0 - 1.0


    _, _, h, w = img.shape
    th, tw = target_hw
    if h != th or w != tw:
        img = torch.nn.functional.interpolate(
            img, size=(th, tw), mode='bilinear', align_corners=False
        )

    return img


def fusion_pair(
    img_a,
    img_b,
    sample_fn,
    target_hw,
    device,
    lamb,
    rho,
    seed,
    save_root,
    img_index,
    image_mode='GRAY'
):

    th, tw = target_hw

    I = preprocess_mff_image(img_a, (th, tw), device, mode=image_mode)
    V = preprocess_mff_image(img_b, (th, tw), device, mode=image_mode)

    img_3 = torch.zeros_like(I, device=device)

    torch.manual_seed(seed)
    x_start = torch.randn((I.repeat(1, 3, 1, 1)).shape, device=device)

    with torch.no_grad():
        sample = sample_fn(
            x_start=x_start,
            record=False,
            I=I,
            V=V,
            img_3=img_3,
            save_root=save_root,
            img_index=img_index,
            lamb=lamb,
            rho=rho,
        )

    sample = sample.detach().cpu().squeeze().numpy()
    # sample: [C,H,W] or [H,W]
    if sample.ndim == 3:
        sample = np.transpose(sample, (1, 2, 0))  # H,W,C
        sample_y = cv2.cvtColor(sample.astype(np.float32), cv2.COLOR_RGB2YCrCb)[:, :, 0]
    else:
        sample_y = sample

    sample_y = sample_y.astype(np.float32)
    minv, maxv = sample_y.min(), sample_y.max()
    if maxv > minv:
        sample_y = (sample_y - minv) / (maxv - minv)
    else:
        sample_y = np.zeros_like(sample_y)
    sample_y = (sample_y * 255.0).astype(np.uint8)

    return sample_y


def build_ycbcr_stacks(image_paths):

    y_list, cb_list, cr_list = [], [], []
    for p in image_paths:
        bgr = cv2.imread(p)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y, cb, cr = cv2.split(ycrcb)
        y_list.append(y)
        cb_list.append(cb)
        cr_list.append(cr)

    y_stack = np.stack(y_list, axis=-1)
    cb_stack = np.stack(cb_list, axis=-1)
    cr_stack = np.stack(cr_list, axis=-1)
    return y_stack, cb_stack, cr_stack


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FlexiD-Fuse for Multi-Focus Fusion')
    parser.add_argument('--model_config', type=str, default='configs/model_config_imagenet.yaml',
                        help='Path to model configuration file')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml',
                        help='Path to diffusion configuration file')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--save_dir', type=str, default='./result_mff',
                        help='Directory to save output results')
    parser.add_argument('--parent_dir', type=str,
                        default='root',
                        help='Root dir of multi-focus stacks (each subdir = one scene)')
    parser.add_argument('--scale', type=int, default=30,
                        help='(unused now) kept for compatibility')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed for reproducibility')
    parser.add_argument('--lamb', type=float, default=0.5,
                        help='Lambda parameter for sampling function')
    parser.add_argument('--rho', type=float, default=0.001,
                        help='Rho parameter for sampling function')
    parser.add_argument('--image_mode', type=str, default='GRAY',
                        help='Image reading mode (for single-channel fusion use GRAY)')
    parser.add_argument('--model_path', type=str, default='./weight/model_weights.pth',
                        help='Path to pretrained model weights (.pth file)')
    parser.add_argument('--img_ext', type=str, default='jpg',
                        help='Image extension in each stack dir')
    args = parser.parse_args()

    logger = get_logger()

    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)

    model = DiT_B_4()
    model = model.to(device)

    if args.model_path is not None and os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        if hasattr(model, 'pos_embed') and 'pos_embed' in state_dict:
            ck_shape = state_dict['pos_embed'].shape
            model_shape = model.pos_embed.shape
            if ck_shape != model_shape:
                print(f"[Warning] pos_embed shape mismatch: ckpt {ck_shape}, model {model_shape}.")
                print("          Removing ckpt pos_embed and using model's own initialization.")
                del state_dict['pos_embed']

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {args.model_path}")
        if missing:
            print("Missing keys:", missing)
        if unexpected:
            print("Unexpected keys:", unexpected)
    else:
        print("No pretrained model loaded, using random initialization.")

    model.eval()

    if hasattr(model, 'x_embedder') and hasattr(model.x_embedder, 'img_size'):
        target_h, target_w = model.x_embedder.img_size
        print(f"[Info] Model expects image size: {target_h}x{target_w}")
    else:
        target_h = target_w = 256
        print(f"[Warning] Cannot find model.x_embedder.img_size, fallback to {target_h}x{target_w}")
    target_hw = (target_h, target_w)

    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model)

    os.makedirs(args.save_dir, exist_ok=True)
    progress_dir = os.path.join(args.save_dir, 'progress')
    result_dir = os.path.join(args.save_dir, 'result_mff')
    os.makedirs(progress_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    parent_dir = args.parent_dir
    stack_dirs = [
        os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]

    total_start_time = time.time()
    processed_stacks = 0
    # ========================

    for stack_dir in stack_dirs:
        stack_name = os.path.basename(stack_dir)
        logger.info(f">> Processing stack: {stack_name}, dir: {stack_dir}")

        ext = args.img_ext.lower()
        all_imgs = [
            os.path.join(stack_dir, f) for f in os.listdir(stack_dir)
            if f.lower().endswith(f'.{ext}')
        ]

        all_imgs = sorted(
            all_imgs,
            key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or 0)
        )

        if len(all_imgs) < 2:
            logger.info(f">> Not enough images in {stack_dir}, skip.")
            continue

        logger.info(f">> Found {len(all_imgs)} images in stack {stack_name}.")

        first_bgr = cv2.imread(all_imgs[0])
        orig_h, orig_w = first_bgr.shape[:2]

        fused_y = None

        num_steps = len(all_imgs) - 1
        for i in range(num_steps):
            if fused_y is None:
                img_a = all_imgs[i]
            else:
                img_a = fused_y

            img_b = all_imgs[i + 1]

            fused_y = fusion_pair(
                img_a=img_a,
                img_b=img_b,
                sample_fn=sample_fn,
                target_hw=target_hw,
                device=device,
                lamb=args.lamb,
                rho=args.rho,
                seed=args.seed + i,
                save_root=progress_dir,
                img_index=f"{stack_name}_step{i+1:02d}",
                image_mode=args.image_mode
            )

            logger.info(f"  Fusion step {i+1}/{num_steps} done for {stack_name}")

        if fused_y.shape[0] != orig_h or fused_y.shape[1] != orig_w:
            fused_y_full = cv2.resize(fused_y, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        else:
            fused_y_full = fused_y

        y_stack, cb_stack, cr_stack = build_ycbcr_stacks(all_imgs)
        H, W, depth = y_stack.shape

        assert fused_y_full.shape == (H, W), "Fused Y size mismatch with source stack."

        target = fused_y_full.astype(np.int16)
        y_stack_int = y_stack.astype(np.int16)

        dist = np.abs(y_stack_int - target[..., None])
        color_index = np.argmin(dist, axis=2)  # [H,W]

        rows = np.arange(H)[:, None]
        cols = np.arange(W)[None, :]

        color_y = fused_y_full
        color_cb = cb_stack[rows, cols, color_index]
        color_cr = cr_stack[rows, cols, color_index]

        ycrcb_fused = cv2.merge([color_y, color_cb, color_cr])
        rgb_fused = cv2.cvtColor(ycrcb_fused, cv2.COLOR_YCrCb2BGR)

        gray_path = os.path.join(result_dir, f"{stack_name}.{ext}")
        rgb_path = os.path.join(result_dir, f"{stack_name}_rgb.{ext}")

        cv2.imwrite(gray_path, fused_y_full)
        cv2.imwrite(rgb_path, rgb_fused)

        logger.info(f">> Fused Y saved to:   {gray_path}")
        logger.info(f">> Fused RGB saved to: {rgb_path}")

        processed_stacks += 1

    total_elapsed = time.time() - total_start_time
    if processed_stacks > 0:
        avg_time = total_elapsed / processed_stacks
        logger.info(f">> Total fusion time: {total_elapsed:.2f} s "
                    f"for {processed_stacks} stacks "
                    f"(avg {avg_time:.2f} s/stack).")
    else:
        logger.info(f">> No stacks processed. Total time: {total_elapsed:.2f} s.")

    logger.info(">> All stacks processed.")
