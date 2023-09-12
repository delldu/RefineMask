#!/usr/bin/env python
# coding: utf-8

import torch
import cv2
import os
import numpy as np

from PIL import Image
from torchvision.transforms import Compose, ToTensor

import matplotlib.pyplot as plt
from inference_img import _build_model, _build_dataloader, split, merge, debug_var
import pdb

cfg = "../configs/bpr/hrnet18s_128.py"
ckpt = "../ckpts/hrnet18s_128-24055c80.pth"
max_ins = 32 # set to lower value to save GPU memory
model = _build_model(cfg, ckpt)
print(model)
# model -- EncoderDecoderRefine(...)
# model.backbone = HRNetRefine(...)


def _inference_one(img, sub_maskdts, sub_dt_paths):
    dets, patches = split(img, sub_maskdts)
    # dets.size() -- [33, 4]
    # tensor [patches] size: [33, 4, 128, 128] , min: tensor(-1.9959, device='cuda:0') , max: tensor(2.3585, device='cuda:0')

    masks = model(patches)[:,1,:,:]    # [33, 2, 128, 128] ==> [33, 128, 128]              # N, 128, 128
    # tensor [masks] size: [33, 128, 128] , min: tensor(8.3961e-05, device='cuda:0', grad_fn=<MinBackward1>) , max: tensor(0.9999, device='cuda:0', grad_fn=<MaxBackward1>)

    # sub_maskdts.size() -- [1, 1024, 2048]
    refineds = merge(sub_maskdts, dets, masks)
    # list [refineds] len: 1 , [tensor([[False, False, False,  ..., False, False, False],
    #         [False, False, False,  ..., False, False, False],
    #         [False, False, False,  ..., False, False, False],
    #         ...,
    #         [False, False, False,  ..., False, False, False],
    #         [False, False, False,  ..., False, False, False],
    #         [False, False, False,  ..., False, False, False]], device='cuda:0')]

    out = []
    for i, dt_path in enumerate(sub_dt_paths):
        out.append(refineds[i].cpu().numpy().astype(np.uint8) * 255)

    # list [out] len: 1 , [array([[0, 0, 0, ..., 0, 0, 0],
    #        [0, 0, 0, ..., 0, 0, 0],
    #        [0, 0, 0, ..., 0, 0, 0],
    #        ...,
    #        [0, 0, 0, ..., 0, 0, 0],
    #        [0, 0, 0, ..., 0, 0, 0],
    #        [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)]
    return out


img_paths = ['lindau_000000_000019_leftImg8bit.png']
mask_paths = [['lindau_000000_000019_leftImg8bit_15_car.png'], ]

dataloader = _build_dataloader(img_paths, mask_paths, device='cuda:0')

for dc in dataloader:
    dt_paths, img, maskdts = dc.data[0][0]
    # list [dt_paths] len: 1 , ['lindau_000000_000019_leftImg8bit_15_car.png']
    # tensor [img] size: [1024, 2048, 3] , min: tensor(-2.0665, device='cuda:0') , max: tensor(2.6400, device='cuda:0')
    # tensor [maskdts] size: [1, 1024, 2048] , min: tensor(0.) , max: tensor(1.)

    cv2.imwrite("/tmp/rawdata.png", maskdts[0].numpy() * 255)

    img = img.cuda()
    maskdts = maskdts.cuda()

    p = 0
    for sub_maskdts in maskdts.split(max_ins): # 32
        q = p + sub_maskdts.size(0)
        sub_dt_paths = dt_paths[p:q] # p:q ----  0 1
        p = q

        # tensor [sub_maskdts] size: [1, 1024, 2048] , min: tensor(0., device='cuda:0') , max: tensor(1., device='cuda:0')
        # list [sub_dt_paths] len: 1 , ['lindau_000000_000019_leftImg8bit_15_car.png']

        refine_mask = _inference_one(img, sub_maskdts, sub_dt_paths)
        # list [refine_mask] len: 1 , [array([[0, 0, 0, ..., 0, 0, 0],
        #        [0, 0, 0, ..., 0, 0, 0],
        #        [0, 0, 0, ..., 0, 0, 0],
        #        ...,
        #        [0, 0, 0, ..., 0, 0, 0],
        #        [0, 0, 0, ..., 0, 0, 0],
        #        [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)]
        # refine_mask[0].shape -- (1024, 2048)


img = cv2.imread(img_paths[0])[:,:,::-1]
coarse_mask = cv2.imread(mask_paths[0][0], 0)

fig, axs = plt.subplots(3, 1, figsize=(20, 15))
axs[0].imshow(img)
axs[1].imshow(img)
axs[1].imshow(coarse_mask, alpha=0.5, cmap="Reds")
axs[2].imshow(img)
axs[2].imshow(refine_mask[0], alpha=0.5, cmap="Reds")

cv2.imwrite("/tmp/coarse.png", coarse_mask*255)
cv2.imwrite("/tmp/refine.png", refine_mask[0])

plt.show()

import todos
color = Image.open(img_paths[0]).convert("RGB")
mask = Image.open("/tmp/coarse.png").convert("L")
color_tensor = ToTensor()(color).unsqueeze(0)
mask_tensor = ToTensor()(mask).unsqueeze(0)
image_tensor = torch.cat((color_tensor, mask_tensor), dim=1)
todos.data.save_tensor([image_tensor], "/tmp/0099.png")




