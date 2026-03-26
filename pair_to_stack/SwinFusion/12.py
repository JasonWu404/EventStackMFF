# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import os
from PIL import Image

img_path = r'E:\pycharmproject\SwinFusion-master\right_down.jpg'    # 正方形图片路径
img = Image.open(img_path)

width, height = img.size   # 获取图片宽高

# 设置切片大小为宽高的1/2
size = width // 2

# 分别取左上,右上,左下,右下四个部分
left_up = img.crop((0, 0, size, size))
right_up = img.crop((size, 0, width, size))
left_down = img.crop((0, size, size, height))
right_down = img.crop((size, size, width, height))

# 保存四张切片图片
left_up.save('11100.jpg')
right_up.save('11101.jpg')
left_down.save('11110.jpg')
right_down.save('11111.jpg')

print('Successfully split the square image into four pieces!')