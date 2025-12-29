# -*- coding: utf-8 -*-
# @Author  : Juntao Wu, XinZhe Xie
# @University  : University of Science and Technology of China, ZheJiang University

import os
import random
from collections import defaultdict
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
from torchvision import transforms
import torchvision.transforms.functional as TF


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def _has_image_ext(name: str):
    return name.lower().endswith(IMG_EXTS)


def _read_event_magnitude_gray(p):

    im = Image.open(p)
    try:
        if im.mode == 'L':
            return im  
        im_rgb = im.convert('RGB')
        r, g, b = im_rgb.split()
        r = np.asarray(r, dtype=np.float32) / 255.0
        b = np.asarray(b, dtype=np.float32) / 255.0

        magnitude = np.abs(r - b)

        mag255 = np.clip(magnitude * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(mag255, mode='L')
    finally:
        im.close()

class ImageEventStackDataset(Dataset):

    def __init__(self,
                 img_root: str,
                 evt_root: str,
                 continuous_depth_dir: str,
                 transform=None,
                 augment=True,
                 subset_fraction=1.0):
        self.img_root = img_root
        self.evt_root = evt_root
        self.continuous_depth_dir = continuous_depth_dir
        self.transform = transform
        self.augment = augment

        self.img_stacks = []
        self.evt_stacks = []
        self.depth_maps = []
        self.stack_sizes = []

        all_stacks = [d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))]
        all_stacks = sorted(all_stacks, key=self.sort_key)

        subset_size = max(1, int(len(all_stacks) * float(subset_fraction))) if len(all_stacks) > 0 else 0
        if subset_size < len(all_stacks):
            selected = sorted(random.sample(all_stacks, subset_size), key=self.sort_key)
        else:
            selected = all_stacks

        for stack_name in selected:
            img_stack_dir = os.path.join(img_root, stack_name)
            evt_stack_dir = os.path.join(evt_root, stack_name)
            depth_map_path = os.path.join(continuous_depth_dir, stack_name + '.png')

            if not os.path.isdir(img_stack_dir):
                continue
            if not os.path.isdir(evt_stack_dir):
                print(f"[Warn] Event stack dir not found: {evt_stack_dir}")
                continue
            if not os.path.exists(depth_map_path):
                print(f"[Warn] Depth map not found for {stack_name}")
                continue

            img_list = [n for n in os.listdir(img_stack_dir) if _has_image_ext(n)]
            img_list = sorted(img_list, key=self.sort_key)
            img_paths = [os.path.join(img_stack_dir, n) for n in img_list]

            evt_list = [n for n in os.listdir(evt_stack_dir) if _has_image_ext(n)]
            evt_list = sorted(evt_list, key=self.sort_key)
            evt_paths = [os.path.join(evt_stack_dir, n) for n in evt_list]

            if len(img_paths) == 0 or len(evt_paths) == 0:
                print(f"[Warn] Empty stack: {stack_name}")
                continue
            if len(img_paths) != len(evt_paths):
                print(f"[Warn] Mismatch N (img {len(img_paths)} != evt {len(evt_paths)}): {stack_name}")
                n = min(len(img_paths), len(evt_paths))
                img_paths, evt_paths = img_paths[:n], evt_paths[:n]

            self.img_stacks.append(img_paths)
            self.evt_stacks.append(evt_paths)
            self.depth_maps.append(depth_map_path)
            self.stack_sizes.append(len(img_paths))

        if len(self.img_stacks) == 0:
            print("[Warn] No valid stacks found!")

    def __len__(self):
        return len(self.img_stacks)

    @staticmethod
    def sort_key(name: str):
        s = ''.join(ch for ch in name if ch.isdigit())
        if s != '':
            return (0, int(s))
        return (1, name.lower())

    def __getitem__(self, idx):
        img_paths = self.img_stacks[idx]
        evt_paths = self.evt_stacks[idx]
        depth_map_path = self.depth_maps[idx]

        imgs = []
        for p in img_paths:
            im = Image.open(p).convert('YCbCr').split()[0]
            imgs.append(im)

        evts = []
        for p in evt_paths:
            ev = _read_event_magnitude_gray(p)
            evts.append(ev)

        depth = Image.open(depth_map_path).convert('L')

        if self.augment:
            imgs, evts, depth = self.consistent_transform(imgs, evts, depth)

        if self.transform:
            imgs = [self.transform(x) for x in imgs]
            evts = [self.transform(x) for x in evts]
            depth = self.transform(depth)

        imgs = [x.squeeze(0) for x in imgs]
        evts = [x.squeeze(0) for x in evts]
        image_stack = torch.stack(imgs, dim=0)
        event_stack = torch.stack(evts, dim=0)

        return image_stack, event_stack, depth, len(imgs)

    @staticmethod
    def consistent_transform(images, events, depth_map):
        if random.random() > 0.5:
            images = [TF.hflip(x) for x in images]
            events = [TF.hflip(x) for x in events]
            depth_map = TF.hflip(depth_map)
        if random.random() > 0.5:
            images = [TF.vflip(x) for x in images]
            events = [TF.vflip(x) for x in events]
            depth_map = TF.vflip(depth_map)
        return images, events, depth_map


class GroupedBatchSampler(Sampler):
    def __init__(self, stack_sizes, batch_size):
        self.stack_size_groups = defaultdict(list)
        for idx, size in enumerate(stack_sizes):
            self.stack_size_groups[size].append(idx)
        self.batch_size = batch_size
        self.batches = self._create_batches()

    def _create_batches(self):
        batches = []
        for size, indices in self.stack_size_groups.items():
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i:i + self.batch_size])
        random.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class CombinedEventDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.stack_sizes = []
        for ds in datasets:
            self.stack_sizes.extend(ds.stack_sizes)

    def __getitem__(self, idx):
        return super().__getitem__(idx)


def get_event_dataloader(dataset_params, batch_size, num_workers=8, augment=True, target_size=384):
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ])

    datasets = []
    for params in dataset_params:
        ds = ImageEventStackDataset(
            img_root=params['img_root'],
            evt_root=params['evt_root'],
            continuous_depth_dir=params['continuous_depth_dir'],
            transform=transform,
            augment=augment,
            subset_fraction=params.get('subset_fraction', 1.0)
        )
        datasets.append(ds)

    combined = CombinedEventDataset(datasets)
    sampler = GroupedBatchSampler(combined.stack_sizes, batch_size)
    loader = DataLoader(combined, batch_sampler=sampler, num_workers=num_workers)
    return loader
