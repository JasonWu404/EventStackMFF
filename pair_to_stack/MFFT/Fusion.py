import os
import sys
import glob
import time

import cv2
import torch
import numpy as np

from tqdm import tqdm
from torch import einsum

from Nets.MFFT import MFFT
from Utilities import Consistency
from Utilities.CUDA_Check import GPUorCPU
from Utilities.GuidedFiltering import guided_filter


class FusionStack:
    def __init__(self,
                 modelpath='RunTimeData/2023-06-25 13.21.25/best_network.pth',
                 dataroot='your root',
                 dataset_name='datasetname',
                 threshold=0.0015,
                 ):
        self.DEVICE = GPUorCPU().DEVICE
        self.MODELPATH = modelpath
        self.DATAROOT = dataroot
        self.DATASET_NAME = dataset_name
        self.THRESHOLD = threshold

        self.SAVE_ROOT = os.path.join('./Results', self.DATASET_NAME)
        os.makedirs(self.SAVE_ROOT, exist_ok=True)

    def LoadWeights(self, modelpath):
        model = MFFT().to(self.DEVICE)
        state = torch.load(modelpath, map_location=self.DEVICE)
        model.load_state_dict(state)
        model.eval()
        return model

    def PrepareSceneList(self, root_path):

        scene_list = []
        for name in sorted(os.listdir(root_path)):
            full = os.path.join(root_path, name)
            if os.path.isdir(full):
                scene_list.append(full)
        return scene_list

    def img_np_to_tensor(self, img_np):

        tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.DEVICE)  # [1,3,H,W]
        return tensor


    def fuse_two_images(self, model, imgA_np, imgB_np):



        A_tensor = self.img_np_to_tensor(imgA_np)
        B_tensor = self.img_np_to_tensor(imgB_np)

        with torch.no_grad():

            NetOut, D = model(A_tensor, B_tensor)

            D = torch.where(D > 0.5, 1., 0.)
            D = self.ConsisVerif(D, self.THRESHOLD)

            D_np = einsum('c w h -> w h c', D[0]).clone().detach().cpu().numpy()  # H x W x 1


            IniF = imgA_np * D_np + imgB_np * (1 - D_np)
            D_GF = guided_filter(IniF, D_np, 4, 0.1)
            fused_np = imgA_np * D_GF + imgB_np * (1 - D_GF)

            fused_np = np.clip(fused_np, 0, 255).astype(np.uint8)

        return fused_np

    def ConsisVerif(self, img_tensor, threshold):
        Verified_img_tensor = Consistency.Binarization(img_tensor)
        if threshold != 0:
            Verified_img_tensor = Consistency.RemoveSmallArea(
                img_tensor=Verified_img_tensor,
                threshold=threshold
            )
        return Verified_img_tensor

    def FusionProcess_Stack(self, model, scene_list):

        running_time = []

        for scene_path in tqdm(scene_list, colour='blue', file=sys.stdout):
            scene_name = os.path.basename(scene_path.rstrip('/\\'))
            print(f'\nProcessing scene: {scene_name}')

            img_paths = sorted(glob.glob(os.path.join(scene_path, '*.*')))
            if len(img_paths) == 0:
                print(f'Warning: {scene_name} is empty, skip.')
                continue

            fused_np = cv2.imread(img_paths[0])
            if fused_np is None:
                print(f'Warning: fail to read {img_paths[0]}, skip scene.')
                continue

            start_scene_time = time.time()

            for path in img_paths[1:]:
                img_np = cv2.imread(path)
                if img_np is None:
                    print(f'Warning: fail to read {path}, skip this image.')
                    continue

                t0 = time.time()
                fused_np = self.fuse_two_images(model, fused_np, img_np)
                running_time.append(time.time() - t0)

            save_name = f'{scene_name}.png'
            save_path = os.path.join(self.SAVE_ROOT, save_name)
            cv2.imwrite(save_path, fused_np)
            print(f'Scene {scene_name} done, saved to: {save_path}, '
                  f'scene_time: {time.time() - start_scene_time:.4f}s')

        if len(running_time) > 1:
            running_time_total = sum(running_time[1:])
            print("\navg_pair_fusion_time: {:.4f} s".format(
                running_time_total / (len(running_time) - 1)
            ))
        print("\nAll results are saved in: " + self.SAVE_ROOT)

    def __call__(self, *args, **kwargs):

        model = self.LoadWeights(self.MODELPATH)
        scene_list = self.PrepareSceneList(self.DATAROOT)
        if len(scene_list) == 0:
            print("No scene folders found under: ", self.DATAROOT)
            return
        self.FusionProcess_Stack(model, scene_list)


if __name__ == '__main__':
    f = FusionStack(
        modelpath='RunTimeData/2023-06-25 13.21.25/best_network.pth',
        dataroot='root', 
        dataset_name='datasetname'
    )
    f()