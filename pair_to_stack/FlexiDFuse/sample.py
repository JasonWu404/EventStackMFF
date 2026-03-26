from functools import partial
import os
import argparse
import yaml
import torch
# from guided_diffusion.unet import create_model
# from guided_diffusion.models import DiT_XXS_8
from guided_diffusion.models_DiT_Mamba import DiT_B_4
from guided_diffusion.gaussian_diffusion import create_sampler
from util.logger import get_logger
import cv2
import numpy as np
from skimage.io import imsave
import warnings
warnings.filterwarnings('ignore')

def image_read(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FlexiD-Fuse Image Fusion')
    parser.add_argument('--model_config', type=str, default='configs/model_config_imagenet.yaml',
                        help='Path to model configuration file')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml',
                        help='Path to diffusion configuration file')                     
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--save_dir', type=str, default='./result/',
                        help='Directory to save output results')
    parser.add_argument('--test_folder', type=str, 
                        default='./Dataset/Medical_Image',
                        help='Path to test dataset folder')
    parser.add_argument('--scale', type=int, default=30,
                        help='Scale factor for image cropping to make it divisible')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed for reproducibility')
    parser.add_argument('--lamb', type=float, default=0.5,
                        help='Lambda parameter for sampling function')
    parser.add_argument('--rho', type=float, default=0.001,
                        help='Rho parameter for sampling function')
    parser.add_argument('--output_subdirs', type=str, nargs='+', default=['recon', 'progress'],
                        help='Subdirectories to create in output directory')
    parser.add_argument('--image_mode', type=str, default='GRAY',
                        help='Image reading mode (RGB, GRAY, YCrCb)')
    parser.add_argument('--model_path', type=str, default='./weight/model_weights.pth',
                        help='Path to pretrained model weights (.pth file)')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)  
    diffusion_config = load_yaml(args.diffusion_config)
   
    # Load model
    # model = create_model(**model_config)
    model = DiT_B_4()
    model = model.to(device)

    # Load pretrained weights if provided
    if args.model_path is not None and os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)

        # 兼容几种常见的 checkpoint 格式
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 处理 pos_embed 尺寸不一致的问题：直接删掉，让模型自己用随机初始化
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

    # 从模型的 patch embedding 里读取期望的输入尺寸
    if hasattr(model, 'x_embedder') and hasattr(model.x_embedder, 'img_size'):
        target_h, target_w = model.x_embedder.img_size
        print(f"[Info] Model expects image size: {target_h}x{target_w}")
    else:
        target_h = target_w = 256  # 兜底
        print(f"[Warning] Cannot find model.x_embedder.img_size, fallback to {target_h}x{target_w}")
  
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model)
   
    # Working directory
    test_folder = args.test_folder     
    out_path = args.save_dir
    os.makedirs(out_path, exist_ok=True)
    for img_dir in args.output_subdirs:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    i=0
    for img_name in os.listdir(os.path.join(test_folder,"vi")):
        inf_img = image_read(os.path.join(test_folder,"ir",img_name), mode=args.image_mode)[np.newaxis,np.newaxis, ...]/255.0 
        vis_img = image_read(os.path.join(test_folder,"vi",img_name), mode=args.image_mode)[np.newaxis,np.newaxis, ...]/255.0 
        # If img_3 path is empty, set img_3 to all zeros
        if os.path.exists(os.path.join(test_folder,"3",img_name)):
            img_3 = image_read(os.path.join(test_folder,"3",img_name), mode=args.image_mode)[np.newaxis,np.newaxis, ...]/255.0
        else:
            img_3 = np.zeros((1,1,inf_img.shape[2],inf_img.shape[3]))
        # img_3 = image_read(os.path.join(test_folder,"3",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0

        inf_img = inf_img*2-1
        vis_img = vis_img*2-1
        # Check if img_3 is all zeros
        if np.sum(img_3) == 0:
            img_3 = img_3
        else:
            img_3 = img_3*2-1
        # crop to make divisible
        inf_img = torch.FloatTensor(inf_img)
        vis_img = torch.FloatTensor(vis_img)
        img_3 = torch.FloatTensor(img_3)

        # 当前尺寸
        _, _, h, w = inf_img.shape

        # 如果和模型期望尺寸不一致，就 resize 到 target_h x target_w
        if h != target_h or w != target_w:
            inf_img = torch.nn.functional.interpolate(
                inf_img, size=(target_h, target_w),
                mode='bilinear', align_corners=False
            )
            vis_img = torch.nn.functional.interpolate(
                vis_img, size=(target_h, target_w),
                mode='bilinear', align_corners=False
            )
            img_3 = torch.nn.functional.interpolate(
                img_3, size=(target_h, target_w),
                mode='bilinear', align_corners=False
            )

        # 放到 GPU / CPU
        inf_img = inf_img.to(device)
        vis_img = vis_img.to(device)
        img_3 = img_3.to(device)
        assert inf_img.shape == vis_img.shape
        logger.info(f"Inference for image {i}")

        # Sampling
        seed = args.seed
        torch.manual_seed(seed)
        x_start = torch.randn((inf_img.repeat(1, 3, 1, 1)).shape, device=device)  
        
        with torch.no_grad():
            sample = sample_fn(x_start=x_start, record=False, I=inf_img, V=vis_img, img_3=img_3, 
                             save_root=out_path, img_index=os.path.splitext(img_name)[0], 
                             lamb=args.lamb, rho=args.rho)

        sample= sample.detach().cpu().squeeze().numpy()
        sample=np.transpose(sample, (1,2,0))
        sample=cv2.cvtColor(sample,cv2.COLOR_RGB2YCrCb)[:,:,0]
        sample=(sample-np.min(sample))/(np.max(sample)-np.min(sample))
        sample=((sample)*255).astype(np.uint8)
        imsave(os.path.join(os.path.join(out_path, 'recon'), "{}.png".format(img_name.split(".")[0])),sample)
        i = i+1
