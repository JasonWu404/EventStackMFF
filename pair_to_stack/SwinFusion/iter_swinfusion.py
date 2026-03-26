# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import torch
import argparse
from torchvision import transforms
import time
import numpy as np
import os
import utils.utils_image as util
import glob
import random
import cv2
from PIL import Image
import numpy as np
import os
import torch
import cv2
import time
import imageio
import pydensecrf.densecrf as dcrf
import torchvision.transforms as transforms
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from models.network_swinfusion1 import SwinFusion as net

from PIL import Image
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='sandwich', help='model name: (default: arch+timestamp)')
    parser.add_argument('--type',default='jpg',type=str)
    parser.add_argument('--fuse_Data_dir', default="/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/dataset_image_stack/boxes",type=str)
    parser.add_argument('--task', type=str, default='fusion', help='classical_sr, lightweight_sr, real_sr, '
                                                                   'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_pair_fusion/swinfusion/Model/Multi_Focus_Fusion/Multi_Focus_Fusion/models')
    parser.add_argument('--iter_number', type=str,
                        default='10000')
    parser.add_argument('--root_path', type=str, default='./Dataset/valsets/',
                        help='input test image root folder')
    parser.add_argument('--dataset', type=str, default='gray',
                        help='input test image name')
    parser.add_argument('--A_dir', type=str, default='A_Y',
                        help='input test image name')
    parser.add_argument('--B_dir', type=str, default='B_Y',
                        help='input test image name')
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--in_channel', type=int, default=1, help='3 means color image and 1 means gray image')
    args = parser.parse_args()
    return args


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn
def test(img_a, img_b, model, args, window_size):

    if args.tile is None:
        # test the image as a whole
        print(img_a.size(), img_b.size())
        img_a=img_a.cuda()
        img_b=img_b.cuda()
        output = model(img_a, img_b)

    else:
        # test the image tile by tile
        b, c, h, w = img_a.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_a)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_a[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                print(in_patch.size())
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output
def fusion(img1,img2):
    from PIL import Image
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
    img_a=torch.tensor(img_a)
    img_b = torch.tensor(img_b)
    img_a=torch.unsqueeze(img_a,0)
    img_b = torch.unsqueeze(img_b, 0)

    img_a=img_a.unsqueeze(0)
    img_b = img_b.unsqueeze(0)

    # inference
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_a.size()
        window_size=8
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

if __name__ == '__main__':
    print('start fusion stack')
    t=time.time()
    args = parse_args()
    device = torch.device('cuda:0')
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('GPU Mode Acitavted')
    else:
        print('CPU Mode Acitavted')
    if not os.path.exists('result/%s'% args.name+'sequence'):
        os.makedirs('result/%s' % args.name+'sequence')
    pic_sequence_list=glob.glob(args.fuse_Data_dir+'/'+'*.png')
    print(len(pic_sequence_list))
    import re
    pic_sequence_list.sort(key=lambda x: int(str(re.findall("\d+", x.split('/')[-1])[-1])))#Sort by the number in the file name

    temp_pic_sequence_list=[None]*(len(pic_sequence_list)-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model_path = os.path.join(args.model_path, args.iter_number + '_G.pth')


    def define_model(args):
        model = net(upscale=args.scale, in_chans=args.in_channel, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler=None, resi_connection='1conv')
        param_key_g = 'params'
        model_path = os.path.join(args.model_path, args.iter_number + '_G.pth')
        print(model_path)
        pretrained_model = torch.load(model_path)
        model.load_state_dict(
            pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

        return model
    model = define_model(args)
    model.eval()
    model = model.to(device)
    for i,data in enumerate(temp_pic_sequence_list):
        if i ==0:
            t1=time.time()
            fuse=fusion(pic_sequence_list[i],pic_sequence_list[i+1])
            temp_pic_sequence_list[i]=fuse
            # cv2.imwrite("result/{}/fusion_{}.{}".format(args.name+'sequence',str(i),args.type),temp_pic_sequence_list[i])
            print('Complete the transition fusion{},cost:{}'.format(str(i),time.time()-t1))
        else:
            t1 = time.time()
            fuse = fusion(temp_pic_sequence_list[i-1], pic_sequence_list[i+1])
            temp_pic_sequence_list[i] = fuse
            cv2.imwrite("result/{}/fusion_{}.{}".format(args.name + 'sequence', str(i), args.type),
                        temp_pic_sequence_list[i])
            print('Complete the transition fusion{},cost:{}'.format(str(i),time.time()-t1))
    result = cv2.cvtColor(temp_pic_sequence_list[-1], cv2.COLOR_RGB2BGR)
    cv2.imwrite("result/{}/fusion_result.{}".format(args.name + 'sequence', args.type),
                result)
    #存一张rgb
    stack_basedir_path = args.fuse_Data_dir
    fusion_y = r"result/{}/fusion_result.{}".format(args.name + 'sequence', args.type)
    save_path = r"result/{}/fusion_result_rgb.{}".format(args.name + 'sequence', args.type)


    def RGB2YCbCr(img_rgb):
        R = img_rgb[:, :, 0]
        G = img_rgb[:, :, 1]
        B = img_rgb[:, :, 2]

        # RGB to YCbCr
        Y = 0.257 * R + 0.564 * G + 0.098 * B + 16
        Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128
        Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128

        return Y, Cb, Cr


    def YCbCr2RGB(img_YCbCr):
        Y = img_YCbCr[:, :, 0]
        Cb = img_YCbCr[:, :, 1]
        Cr = img_YCbCr[:, :, 2]

        # YCbCr to RGB
        R = 1.164 * (Y - 16) + 1.596 * (Cr - 128)
        G = 1.164 * (Y - 16) - 0.392 * (Cb - 128) - 0.813 * (Cr - 128)
        B = 1.164 * (Y - 16) + 2.017 * (Cb - 128)

        image_RGB = np.dstack((R, G, B))
        return image_RGB


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


    stack_dir_list = os.listdir(stack_basedir_path)
    fusion_y = cv2.imread(fusion_y, 0)
    # 颜色信息通过Y通道内找最相近的值找到索引图像层,寻找颜色索引矩阵color_index
    target = np.array(fusion_y).astype(np.int64)

    img_stack_np, image_paths = stack_y_channels(stack_basedir_path)
    H, W, depth = img_stack_np.shape
    img_stack_np = img_stack_np.astype(np.int64)

    dist_list = []
    for depth_index in range(depth):
        dist = np.abs(img_stack_np[:, :, depth_index] - target)
        dist_list.append(dist)
    dist_list = np.stack(dist_list, -1)
    color_index = np.argmin(dist_list, axis=2)
    color_index_smoothed = cv2.GaussianBlur(color_index.astype(np.uint8), (11, 11), 0)
    import matplotlib.pyplot as plt

    # plt.imsave(os.path.join(predict_save_path, 'index depth.{}'.format(args.out_format)), color_index)
    # plt.imsave(os.path.join(predict_save_path, 'smooth index depth.{}'.format(args.out_format)), color_index_smoothed)
    cb_channels = []
    cr_channels = []
    # 每一个像素都根据索引矩阵取原始Cb和Cr
    for img_path in image_paths:
        img = Image.open(img_path).convert('YCbCr')
        y = np.array(img.split()[0])
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
            y_pixel = fusion_y[i, j]

            color_img[i, j, 0] = y_pixel
            color_img[i, j, 1] = cb_pixel
            color_img[i, j, 2] = cr_pixel

    color_img = color_img.astype(np.int8)

    color_img = Image.fromarray(color_img, 'YCbCr')
    rgb_img = color_img.convert('RGB')

    # 保存图片
    rgb_img.save(save_path, quality=100)
    print('Finish fusion!!!')
    print('cost:',time.time()-t)