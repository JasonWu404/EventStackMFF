
import argparse
import utils.utils_image as util
import glob
import numpy as np
import os
import torch
import cv2
import time

from models.network_swinfusion1 import SwinFusion as net

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='NYU Depth V2', help='model name: (default: arch+timestamp)')
    parser.add_argument('--type', default='jpg', type=str)
    parser.add_argument('--parent_dir', default=r"/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_depth_from_focus/AiFDepthNet/data/Datasets_StackMFF/NYU Depth V2/image stack", type=str)
    parser.add_argument('--model_path', type=str, default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_pair_fusion/swinfusion/Model/Multi_Focus_Fusion/Multi_Focus_Fusion/models')
    parser.add_argument('--iter_number', type=str, default='10000')
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--tile', type=int, default=None)
    parser.add_argument('--tile_overlap', type=int, default=32)
    parser.add_argument('--in_channel', type=int, default=1)
    args = parser.parse_args()
    return args
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def test(img_a, img_b, model, args, window_size):
    if args.tile is None:
        img_a = img_a.cuda()
        img_b = img_b.cuda()
        output = model(img_a, img_b)
    else:
        # Tile-based testing code (unchanged)
        pass
    return output


def fusion(img1, img2, model, args, window_size):
    if isinstance(img1, str):
        img1 = Image.open(img1).convert('L')
        img_a = np.array(img1)
    else:
        img1 = Image.fromarray(img1).convert('L')
        img_a = np.array(img1)

    if isinstance(img2, str):
        img2 = Image.open(img2).convert('L')
        img_b = np.array(img2)
    else:
        img2 = Image.fromarray(img2).convert('L')
        img_b = np.array(img2)

    img_a = util.uint2single(img_a)
    img_b = util.uint2single(img_b)
    img_a = torch.tensor(img_a).unsqueeze(0).unsqueeze(0)
    img_b = torch.tensor(img_b).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        _, _, h_old, w_old = img_a.size()
        window_size = 8
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_a = torch.cat([img_a, torch.flip(img_a, [2])], 2)[:, :, :h_old + h_pad, :]
        img_a = torch.cat([img_a, torch.flip(img_a, [3])], 3)[:, :, :, :w_old + w_pad]
        img_b = torch.cat([img_b, torch.flip(img_b, [2])], 2)[:, :, :h_old + h_pad, :]
        img_b = torch.cat([img_b, torch.flip(img_b, [3])], 3)[:, :, :, :w_old + w_pad]

        output = test(img_a, img_b, model, args, window_size)
        output = output[..., :h_old * args.scale, :w_old * args.scale]
        output = output.detach()[0].float().cpu()

    output = util.tensor2uint(output)
    return output


def define_model(args):
    model = net(upscale=args.scale, in_chans=args.in_channel, img_size=128, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                mlp_ratio=2, upsampler=None, resi_connection='1conv')
    param_key_g = 'params'
    model_path = os.path.join(args.model_path, args.iter_number + '_G.pth')
    pretrained_model = torch.load(model_path)
    model.load_state_dict(
        pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    return model


def process_stack(stack_dir, model, args, window_size):
    pic_sequence_list = glob.glob(os.path.join(stack_dir, '*.{}'.format(args.type)))
    pic_sequence_list.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))

    fused_Y = None
    for i in range(len(pic_sequence_list) - 1):
        if i == 0:
            fused_Y = fusion(pic_sequence_list[i], pic_sequence_list[i + 1], model, args, window_size)
        else:
            fused_Y = fusion(fused_Y, pic_sequence_list[i + 1], model, args, window_size)

    # 颜色信息通过Y通道内找最相近的值找到索引图像层,寻找颜色索引矩阵color_index
    target = np.array(fused_Y).astype(np.int64)

    def stack_y_channels(folder_path):
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                       f.endswith('.jpg') or f.endswith('.png')]

        y_channels = []
        for img_path in image_paths:
            img = Image.open(img_path).convert('YCbCr')
            y = np.array(img.split()[0])
            y_channels.append(y)

        y_channels = np.stack(y_channels, axis=-1)

        # if y_channels.shape[-1] < 7:
        #     pad_width = ((0, 0), (0, 0), (0, 16 - y_channels.shape[-1]))
        #     y_channels = np.pad(y_channels, pad_width, mode='constant', constant_values=0)

        return y_channels, image_paths
    img_stack_np, image_paths = stack_y_channels(stack_dir)
    H, W, depth = img_stack_np.shape
    img_stack_np = img_stack_np.astype(np.int64)

    dist_list = []
    for depth_index in range(depth):
        dist = np.abs(img_stack_np[:, :, depth_index] - target)
        dist_list.append(dist)
    dist_list = np.stack(dist_list, -1)
    color_index = np.argmin(dist_list, axis=2)
    color_index_smoothed = cv2.GaussianBlur(color_index.astype(np.uint8), (11, 11), 0)

    cb_channels = []
    cr_channels = []
    # 每一个像素都根据索引矩阵取原始Cb和Cr
    for img_path in image_paths:
        img = Image.open(img_path).convert('YCbCr')
        cb = np.array(img.split()[1])
        cr = np.array(img.split()[2])
        cb_channels.append(cb)
        cr_channels.append(cr)

    # 每一像素逐一合成颜色
    color_img = np.zeros((H, W, 3)).astype(np.int32)
    for i in range(H):
        for j in range(W):
            color_index_number = color_index[i, j]
            cb_pixel = cb_channels[color_index_number][i, j]
            cr_pixel = cr_channels[color_index_number][i, j]
            y_pixel = fused_Y[i, j]

            color_img[i, j, 0] = y_pixel
            color_img[i, j, 1] = cb_pixel
            color_img[i, j, 2] = cr_pixel

    color_img = color_img.astype(np.uint8)
    color_img = Image.fromarray(color_img, 'YCbCr')
    rgb_img = color_img.convert('RGB')

    return fused_Y, np.array(rgb_img)


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = define_model(args)
    model.eval()
    model = model.to(device)

    window_size = 8

    # 创建保存结果的文件夹
    save_dir = os.path.join(os.getcwd(),'result_stack', args.name)
    os.makedirs(save_dir, exist_ok=True)

    stack_dirs = [os.path.join(args.parent_dir, d) for d in os.listdir(args.parent_dir)
                  if os.path.isdir(os.path.join(args.parent_dir, d))]

    total_time = 0
    for stack_dir in stack_dirs:
        start_time = time.time()

        fused_Y, fused_RGB = process_stack(stack_dir, model, args, window_size)

        stack_name = os.path.basename(stack_dir)

        # 保存Y通道图像
        y_save_path = os.path.join(save_dir, f"{stack_name}.{args.type}")
        cv2.imwrite(y_save_path, fused_Y)

        # 保存RGB图像
        rgb_save_path = os.path.join(save_dir, f"{stack_name}_rgb.{args.type}")
        cv2.imwrite(rgb_save_path, cv2.cvtColor(fused_RGB, cv2.COLOR_RGB2BGR))

        end_time = time.time()
        stack_time = end_time - start_time
        total_time += stack_time
        print(f"Processed {stack_name} in {stack_time:.2f} seconds")

    avg_time = total_time / len(stack_dirs)
    print(f"Average processing time per stack: {avg_time:.2f} seconds")