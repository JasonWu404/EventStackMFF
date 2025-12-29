# -*- coding: utf-8 -*-
# @Author  : Juntao Wu, XinZhe Xie
# @University  : University of Science and Technology of China, ZheJiang University

import argparse
import torch
import cv2
import numpy as np
import os
from datetime import datetime
import matplotlib
from network_event import EventStackMFF
import torch.nn.functional as F

def gray_to_colormap(img, cmap='rainbow'):
    
    img = np.clip(img, 0, 1)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    return colormap


def parse_args():
    parser = argparse.ArgumentParser(description="Image Stack and Event Stack Fusion Inference Script")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input image stack')
    parser.add_argument('--event_dir', type=str, required=True,
                        help='Directory containing input event stack')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Directory for saving results')
    parser.add_argument('--model_path', type=str, default='model.pth',
                       help='Path to the trained model weights')
    return parser.parse_args()


def load_image_stack(input_dir):
    
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    gray_tensors = []
    color_images = []

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        
        bgr_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        color_images.append(rgb_img)

        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        gray_tensor = torch.from_numpy(gray_img.astype(np.float32) / 255.0)
        gray_tensors.append(gray_tensor)

    return torch.stack(gray_tensors), color_images

def load_event_stack(event_dir):
    
    event_files = sorted([f for f in os.listdir(event_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    event_tensors = []

    for evt_file in event_files:
        evt_path = os.path.join(event_dir, evt_file)
        evt_img = cv2.imread(evt_path, cv2.IMREAD_GRAYSCALE)  # events already grayscale
        evt_tensor = torch.from_numpy(evt_img.astype(np.float32) / 255.0)
        event_tensors.append(evt_tensor)

    return torch.stack(event_tensors)

def create_fused_color_image(fused_image, depth_map_index, color_stack):
    
    height, width = fused_image.shape
    num_images = len(color_stack)

    depth_map_index = np.clip(depth_map_index, 0, num_images - 1).astype(int)

    color_array = np.stack(color_stack, axis=0)

    fused_color = color_array[depth_map_index, np.arange(height)[:, None], np.arange(width)]

    fused_color_bgr = cv2.cvtColor(fused_color.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return fused_color_bgr


def resize_to_multiple_of_32(image):
   
    h, w = image.shape[-2:]
    new_h = ((h - 1) // 32 + 1) * 32
    new_w = ((w - 1) // 32 + 1) * 32
    resized_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return resized_image, (h, w)


def process_stack(model, image_stack, event_stack, color_stack, device):
    
    model.eval()
    with torch.no_grad():

        original_size = image_stack.shape[-2:]

        resized_stack, _ = resize_to_multiple_of_32(image_stack.unsqueeze(0))
        resized_evt_stack, _ = resize_to_multiple_of_32(event_stack.unsqueeze(0))
        resized_stack = resized_stack.to(device)
        resized_evt_stack = resized_evt_stack.to(device)

        fused_image, estimated_depth, depth_map_index = model(resized_stack, resized_evt_stack)

        fused_image = cv2.resize(fused_image.cpu().numpy().squeeze(),
                               (original_size[1], original_size[0]))
        estimated_depth = cv2.resize(estimated_depth.cpu().numpy().squeeze(),
                                   (original_size[1], original_size[0]))
        depth_map_index = cv2.resize(depth_map_index.cpu().numpy().squeeze(),
                                   (original_size[1], original_size[0]),
                                   interpolation=cv2.INTER_NEAREST)

        color_fused = create_fused_color_image(fused_image, depth_map_index, color_stack)

        depth_colormap = gray_to_colormap(estimated_depth)

        return fused_image, estimated_depth, color_fused, depth_colormap


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = EventStackMFF()
    state_dict = torch.load(args.model_path, map_location=device)

    new_state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
        if not (k.endswith("total_ops") or k.endswith("total_params"))
    }
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)

    image_stack, color_stack = load_image_stack(args.input_dir)
    event_stack = load_event_stack(args.event_dir)
    image_stack = image_stack.to(device)
    event_stack = event_stack.to(device)

    fused_image, estimated_depth, color_fused, depth_colormap = process_stack(
        model, image_stack, event_stack, color_stack, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(args.output_dir, f'fused_gray_{timestamp}.jpg'),
                (fused_image * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.output_dir, f'depth_map_{timestamp}.jpg'),
                (estimated_depth * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.output_dir, f'color_fused_{timestamp}.jpg'),
                color_fused)
    cv2.imwrite(os.path.join(args.output_dir, f'depth_colormap_{timestamp}.jpg'),
                cv2.cvtColor(depth_colormap, cv2.COLOR_RGB2BGR))

    print(f"Results saved to {args.output_dir} with timestamp {timestamp}")


if __name__ == "__main__":
    main()