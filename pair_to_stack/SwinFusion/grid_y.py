# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import cv2

img = cv2.imread(r'E:\pycharmproject\SwinFusion-master\A__Y\2.jpg')

grid_size = 4  # 16x16 grid
line_thickness = 3  # thickness of grid lines

height, width = img.shape[:2]
grid_height = height // grid_size
grid_width = width // grid_size

for i in range(grid_size):
    for j in range(grid_size):
        # Draw horizontal lines
        cv2.line(img, (j * grid_width, i * grid_height), (width, i * grid_height), (255, 255, 255), line_thickness)
        # Draw vertical lines
        cv2.line(img, (j * grid_width, i * grid_height), (j * grid_width, height), (255, 255, 255), line_thickness)

cv2.imwrite('square_grid3.jpg', img)