import torch.utils.data as data
import os
from os.path import join
from PIL import Image, ImageOps
import torch
from torchvision.transforms import Compose, ToTensor
import random


def is_image_file(filename):

    return any(filename.endswith(extension) for extension in
               ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF'])


def transform():

    return Compose([ToTensor()])


def load_img(filepath):

    img = Image.open(filepath)
    return img


def augment(A_image, B_image, flip_h=True, rot=True):

    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        A_image = ImageOps.flip(A_image)
        B_image = ImageOps.flip(B_image)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            A_image = ImageOps.mirror(A_image)
            B_image = ImageOps.mirror(B_image)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            A_image = A_image.rotate(180)
            B_image = B_image.rotate(180)
            info_aug['trans'] = True

    return A_image, B_image, info_aug


class Data_train(data.Dataset):
    def __init__(self, cfg, transform=transform()):
        super(Data_train, self).__init__()
        self.cfg = cfg

        data_dir_root = cfg[cfg['train_dataset']]['data_dir']['data_dir_root']

        self.stack_folders = [join(data_dir_root, folder) for folder in os.listdir(data_dir_root) if
                              os.path.isdir(join(data_dir_root, folder))]

        self.patch_size = cfg[cfg['train_dataset']]['patch_size']
        self.transform = transform
        self.data_augmentation = cfg[cfg['train_dataset']]['data_augmentation']

    def __getitem__(self, index):
        stack_folder = self.stack_folders[index]

        image_filenames = sorted([join(stack_folder, f) for f in os.listdir(stack_folder) if is_image_file(f)])

        assert len(image_filenames) > 1, f"图像栈文件夹 {stack_folder} 中的图像数量不足"

        images = [load_img(f) for f in image_filenames]

        if self.data_augmentation:
            images = [augment(img, img, flip_h=True, rot=True)[0] for img in images]  # 目前暂时只进行水平翻转和旋转

        if self.transform:
            images = [self.transform(img) for img in images]

        return images, stack_folder

    def __len__(self):
        return len(self.stack_folders)


class Data_eval(data.Dataset):
    def __init__(self, cfg, transform=transform()):
        super(Data_eval, self).__init__()
        self.cfg = cfg

        data_dir_root = cfg[cfg['test_dataset']]['data_dir']['data_dir_root']

        self.stack_folders = [join(data_dir_root, folder) for folder in os.listdir(data_dir_root) if
                              os.path.isdir(join(data_dir_root, folder))]

        self.transform = transform

    def __getitem__(self, index):
        stack_folder = self.stack_folders[index]

        image_filenames = sorted([join(stack_folder, f) for f in os.listdir(stack_folder) if is_image_file(f)])

        assert len(image_filenames) > 1, f"图像栈文件夹 {stack_folder} 中的图像数量不足"

        images = [load_img(f) for f in image_filenames]

        if self.transform:
            images = [self.transform(img) for img in images]

        return images, stack_folder 

    def __len__(self):
        return len(self.stack_folders)