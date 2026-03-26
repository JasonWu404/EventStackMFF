# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
from models.network_swinfusion1 import SwinFusion as net
import argparse
import os
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='fusion', help='classical_sr, lightweight_sr, real_sr, '
                                                                 'gray_dn, color_dn, jpeg_car')
parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
parser.add_argument('--model_path', type=str,
                    default='./Model/Multi_Focus_Fusion/Multi_Focus_Fusion/models')
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
parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
parser.add_argument('--in_channel', type=int, default=1, help='3 means color image and 1 means gray image')
args = parser.parse_args()
def define_model(args):
    model = net(upscale=args.scale, in_chans=args.in_channel, img_size=128, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                mlp_ratio=2, upsampler=None, resi_connection='1conv')

    return model
model=define_model(args)
input = torch.randn(1, 1, 256, 256)
from thop import profile
flops, params = profile(model, inputs=(input,input))
print(flops)
print(params)
print('Number of models parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
