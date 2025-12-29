# -*- coding: utf-8 -*-
# @Author  : Juntao Wu, XinZhe Xie
# @University  : University of Science and Technology of China, ZheJiang University

import numpy as np
import math
import os
import skimage
from PIL import Image
import pandas as pd
import cv2

def AG_function(image):
	width = image.shape[1]
	width = width - 1
	height = image.shape[0]
	height = height - 1
	tmp = 0.0
	[grady, gradx] = np.gradient(image)
	s = np.sqrt((np.square(gradx) + np.square(grady)) / 2)
	AG = np.sum(np.sum(s)) / (width * height)
	return AG

def SF_function(image):
    image_array = np.array(image)
    RF = np.diff(image_array, axis=0)
    RF1 = np.sqrt(np.mean(np.mean(RF ** 2)))
    CF = np.diff(image_array, axis=1)
    CF1 = np.sqrt(np.mean(np.mean(CF ** 2)))
    SF = np.sqrt(RF1 ** 2 + CF1 ** 2)
    return SF

def EN_function(image_array):

    histogram, bins = np.histogram(image_array, bins=256, range=(0, 255))

    histogram = histogram / float(np.sum(histogram))

    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
    return entropy

def Mean(image):
    img_array = np.array(image)
    return np.mean(img_array)

def SD_function(image_array):
    m, n = image_array.shape
    u = np.mean(image_array)
    SD = np.sqrt(np.sum(np.sum((image_array - u) ** 2)) / (m * n))
    return SD

def MSE_function(A, F):
    A = A / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE = np.sum(np.sum((F - A)**2))/(m*n)

    return MSE


def RMSE_function(A, F):

    A = A / 255.0
    F = F / 255.0

    m, n = F.shape

    MSE = np.sum((F - A) ** 2) / (m * n)

    RMSE = np.sqrt(MSE)

    return RMSE


def logRMS_function(A, F):

    A = A / 255.0
    F = F / 255.0

    m, n = F.shape

    logRMS = np.sqrt(np.sum((np.log1p(F) - np.log1p(A)) ** 2) / (m * n))

    return logRMS


def abs_rel_error_function(A, F):

    A = A / 255.0
    F = F / 255.0

    m, n = F.shape

    abs_rel = np.sum(np.abs(F - A) / (A + 1e-8)) / (m * n)

    return abs_rel


def sqr_rel_error_function(A, F):
    
    A = A / 255.0
    F = F / 255.0

    m, n = F.shape

    sqr_rel = np.sum(((F - A) ** 2) / (A + 1e-8)) / (m * n)

    return sqr_rel
def MAE_function(A, F):

    A = np.array(A, dtype=np.float64)
    F = np.array(F, dtype=np.float64)

    A = A / 255.0
    F = F / 255.0

    m, n = F.shape

    absolute_diff = np.abs(F - A)

    MAE = np.sum(absolute_diff) / (m * n)

    return MAE

def mean_diff(A, F):

    A = np.array(A, dtype=np.float64)
    F = np.array(F, dtype=np.float64)

    A = A / 255.0
    F = F / 255.0

    m, n = F.shape

    mean_a=np.mean(A)
    mean_F=np.mean(F)
    mean_diff=mean_a/mean_F

    return round(mean_diff,4)