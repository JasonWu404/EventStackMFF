# -*- coding: utf-8 -*-
# @Author  : Juntao Wu, XinZhe Xie
# @University  : University of Science and Technology of China, ZheJiang University

import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from matplotlib.widgets import Button

def get_image_formats(folder):
    formats = []
    if not os.path.exists(folder):
        print(f"警告：文件夹不存在: {folder}")
        return 'jpg'
    for root, dirs, files in os.walk(folder):
        for file in files:
            filename, ext = os.path.splitext(file)
            ext = ext[1:].lower()
            if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
                if ext not in formats:
                    formats.append(ext)
    if len(formats) == 0:
        print(f"警告：在文件夹 {folder} 中没有找到图片文件")
        return 'jpg'
    return formats[0]

methods = ["CVT","DSIFT","DTCWT","NSCT","U2Fusion","SDNet","MFF-GAN",
           "SwinFusion","MUFusion","SwinMFF","FlexiDFuse","DDBFusion","StackMFF","StackMFFV2","EventStackMFF","Ground Truth"]

dataset='FlyingThings3D'
index="0000590"
basepath=r'D:\Code\EventStackMFF\results_methods_self'

if_save=True

point_x=351
point_y=127
height =40
width  =40

point2_x = point_x + width + 10
point2_y = point_y

font = {'family': 'Arial'}
fontsize=24

num_cols = 8
num_methods = len(methods)
num_rows = (num_methods + num_cols - 1) // num_cols

TARGET_HEIGHT = 360
TARGET_WIDTH  = 640

ZOOM_W_FRAC = 0.5
ZOOM_H_FRAC = 0.5
ZOOM_PAD    = 0.0   

ROW_SHIFT = {1: 0.048}

FIGSIZE = (35, 6.6)
BOTTOM_MARGIN = 0.1412
TOP_MARGIN = 0.92
TITLE_PAD = 10

ROI1_COLOR = 'red'
ROI2_COLOR = 'deepskyblue'

img_data     = []
axins1_list  = []   # 左侧放大块
axins2_list  = []   # 右侧放大块
rect1_list   = []   # 主图上的 ROI1 矩形
rect2_list   = []   # 主图上的 ROI2 矩形
axs          = []
current_point1_x = point_x
current_point1_y = point_y
current_point2_x = point2_x
current_point2_y = point2_y

def clamp_roi(x, y):
    x = max(width//2,  min(TARGET_WIDTH - width//2,  x))
    y = max(height//2, min(TARGET_HEIGHT - height//2, y))
    return x, y

def on_click(event):
    """左键更新 ROI1；右键更新 ROI2。"""
    global current_point1_x, current_point1_y, current_point2_x, current_point2_y
    if event.inaxes is None or event.xdata is None or event.ydata is None:
        return
    for i, ax in enumerate(axs[:num_methods]):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            x, y = clamp_roi(x, y)
            new_x = x - width//2
            new_y = y - height//2
            if event.button == 1:      # 左键 -> ROI1
                current_point1_x, current_point1_y = new_x, new_y
                update_all_zoom_areas(which=1)
                print(f"[ROI1] 点击({x},{y}) -> 左上=({new_x},{new_y})")
            elif event.button in (2,3): # 中/右键 -> ROI2
                current_point2_x, current_point2_y = new_x, new_y
                update_all_zoom_areas(which=2)
                print(f"[ROI2] 点击({x},{y}) -> 左上=({new_x},{new_y})")
            fig.canvas.draw()
            break

def update_all_zoom_areas(which=1):
    do_roi1 = (which in (0,1))
    do_roi2 = (which in (0,2))
    for i in range(num_methods):
        
        if do_roi1:
            if i < len(rect1_list) and rect1_list[i] is not None:
                rect1_list[i].remove()
            r1 = plt.Rectangle((current_point1_x, current_point1_y), width, height,
                               edgecolor=ROI1_COLOR, linestyle='--', fill=False, linewidth=2)
            axs[i].add_patch(r1)
            rect1_list[i] = r1
        if do_roi2:
            if i < len(rect2_list) and rect2_list[i] is not None:
                rect2_list[i].remove()
            r2 = plt.Rectangle((current_point2_x, current_point2_y), width, height,
                               edgecolor=ROI2_COLOR, linestyle='--', fill=False, linewidth=2)
            axs[i].add_patch(r2)
            rect2_list[i] = r2

        if do_roi1 and i < len(axins1_list) and axins1_list[i] is not None:
            crop1 = img_data[i][current_point1_y:current_point1_y + height,
                                 current_point1_x:current_point1_x + width]
            axins1_list[i].clear()
            axins1_list[i].imshow(crop1)
            axins1_list[i].set_xticks([]); axins1_list[i].set_yticks([])
            for s in ['top','right','bottom','left']:
                axins1_list[i].spines[s].set_color(ROI1_COLOR)
                axins1_list[i].spines[s].set_linestyle('--')
                axins1_list[i].spines[s].set_linewidth(2)

        if do_roi2 and i < len(axins2_list) and axins2_list[i] is not None:
            crop2 = img_data[i][current_point2_y:current_point2_y + height,
                                 current_point2_x:current_point2_x + width]
            axins2_list[i].clear()
            axins2_list[i].imshow(crop2)
            axins2_list[i].set_xticks([]); axins2_list[i].set_yticks([])
            for s in ['top','right','bottom','left']:
                axins2_list[i].spines[s].set_color(ROI2_COLOR)
                axins2_list[i].spines[s].set_linestyle('--')
                axins2_list[i].spines[s].set_linewidth(2)

fig, axs_grid = plt.subplots(num_rows, num_cols, figsize=FIGSIZE)
axs = axs_grid.flatten()

flag = 0
for j in range(num_methods):
    ext = get_image_formats(os.path.join(basepath, methods[flag],dataset))
    img_name = f'{index}.{ext}'
    pic_path = os.path.join(basepath, methods[flag],dataset, img_name)
    pic_path_rgb = os.path.join(basepath, methods[flag],dataset, f'{index}_rgb.{ext}')
    img = plt.imread(pic_path_rgb) if os.path.exists(pic_path_rgb) else plt.imread(pic_path)

    if img.dtype in (np.float32, np.float64):
        img = (img * 255).astype(np.uint8)
    img_resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    img_data.append(img_resized)

    ax = axs[j]
    ax.imshow(img_resized)
    title = 'Proposed' if methods[flag]=='StackMFF V2' else methods[flag]
    ax.set_title(title, font=font, fontsize=fontsize+8, pad=TITLE_PAD)
    ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')

    r1 = plt.Rectangle((current_point1_x, current_point1_y), width, height,
                       edgecolor=ROI1_COLOR, linestyle='--', fill=False, linewidth=2)
    r2 = plt.Rectangle((current_point2_x, current_point2_y), width, height,
                       edgecolor=ROI2_COLOR, linestyle='--', fill=False, linewidth=2)
    ax.add_patch(r1); ax.add_patch(r2)
    rect1_list.append(r1); rect2_list.append(r2)

    axins1_list.append(None)
    axins2_list.append(None)

    flag += 1

for k in range(num_methods, num_rows * num_cols):
    fig.delaxes(axs[k])

plt.tight_layout()
plt.subplots_adjust(left=0.03, right=0.97, top=TOP_MARGIN, bottom=BOTTOM_MARGIN,
                    wspace=-0.6, hspace=0.80)
fig.canvas.draw()

renderer = fig.canvas.get_renderer()
for i in range(num_methods):
    ax = axs[i]
    im_artist = ax.images[0]
    bbox_disp = im_artist.get_window_extent(renderer=renderer)
    bbox_fig  = bbox_disp.transformed(fig.transFigure.inverted())

    img_left   = bbox_fig.x0
    img_bottom = bbox_fig.y0

    axpos   = ax.get_position()
    zoom_w  = axpos.width  * ZOOM_W_FRAC
    zoom_h  = axpos.height * ZOOM_H_FRAC

    left1  = img_left
    left2  = img_left + zoom_w  # 第二个紧接在第一个右侧
    bottom = img_bottom - zoom_h - ZOOM_PAD
    row_idx = i // num_cols
    bottom -= ROW_SHIFT.get(row_idx, 0.0)
    bottom = max(bottom, 0.002)

    ax_zoom1 = fig.add_axes([left1, bottom, zoom_w, zoom_h])
    ax_zoom2 = fig.add_axes([left2, bottom, zoom_w, zoom_h])

    for a, col in [(ax_zoom1, ROI1_COLOR), (ax_zoom2, ROI2_COLOR)]:
        for s in ['top','right','bottom','left']:
            a.spines[s].set_color(col)
            a.spines[s].set_linestyle('--')
            a.spines[s].set_linewidth(2)
        a.set_xticks([]); a.set_yticks([]); a.axis('off')

    crop1 = img_data[i][current_point1_y:current_point1_y + height,
                       current_point1_x:current_point1_x + width]
    crop2 = img_data[i][current_point2_y:current_point2_y + height,
                       current_point2_x:current_point2_x + width]
    ax_zoom1.imshow(crop1)
    ax_zoom2.imshow(crop2)

    axins1_list[i] = ax_zoom1
    axins2_list[i] = ax_zoom2

fig.canvas.mpl_connect('button_press_event', on_click)

if if_save:
    plt.savefig(
        r'D:\Code\EventStackMFF\results_methods\PPT\EventStackMFF\{}_compare_{}.jpg'.format(dataset,index),
        format='jpg', dpi=300)

plt.show()
