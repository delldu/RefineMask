"""Refine Mask Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Tue 12 Sep 2023 12:36:45 AM CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.ops import nms, roi_align

import todos
from RefineMask import refine_mask
from torchvision.transforms import Compose, ToTensor
import pdb

def find_border(mask, kernel_size=3):
    """ Find the boundaries.
    """
    N, C, H, W = mask.size()

    mask = (mask >= 0.5).to(torch.float32)

    kernel = mask.new_ones((1, 1, kernel_size, kernel_size))
    # tensor [kernel] size: [1, 1, 3, 3] , min && max: tensor(1., device='cuda:0') , 

    border = F.conv2d(mask, kernel, stride=1, padding=kernel_size//2)
    # tensor [border] size: [1, 1, 1024, 1024] , min: tensor(0., device='cuda:0') , max: tensor(9., device='cuda:0')

    bml = torch.abs(border - kernel_size*kernel_size)
    bms = torch.abs(border)
    border = torch.min(bml, bms) / (kernel_size*kernel_size/2)
    # tensor [border] size: [1, 1, 1024, 1024] , min: tensor(0., device='cuda:0') , max: tensor(0.8889, device='cuda:0')

    return border

def get_bboxes(border, patch_size=64, iou_thresh=0.25):
    """boundaries of coarse mask -> patch bboxs
    """
    B, C, H, W = border.size()
    assert B == 1 and C == 1, "Only support one batch size! "

    # border.size() -- [1, 1, 1024, 1024]
    # patch_size = 64
    # iou_thresh = 0.25

    ys, xs = torch.nonzero(border[0, 0, :, :], as_tuple=True)
    scores = border[0, 0, ys,xs]

    ys = ys.float()
    xs = xs.float()
    dense_boxes = torch.stack([xs - patch_size//2, ys - patch_size//2, 
            xs + patch_size//2, ys + patch_size//2, scores]).T

    boxes = dense_boxes[:, 0:4]
    scores = dense_boxes[:, 4:5].flatten()
    keep = nms(boxes, scores, iou_thresh)
    bboxes = dense_boxes[keep] # size() from [7428, 5] to [70, 5]

    # box.x1
    s = bboxes[:, 0] < 0 
    bboxes[s, 0] = 0
    bboxes[s, 2] = patch_size # x2

    # box.y1
    s = bboxes[:, 1] < 0 
    bboxes[s, 1] = 0.0
    bboxes[s, 3] = patch_size # y2

    # box.x2
    s = bboxes[:, 2] >= W 
    bboxes[s, 0] = W - 1 - patch_size # x1
    bboxes[s, 2] = W - 1 # x2

    # box.y2
    s = bboxes[:, 3] >= H 
    bboxes[s, 1] = H - 1 - patch_size # y1
    bboxes[s, 3] = H - 1 # y2

    return bboxes # size() -- [70, 5]

def roi_list(bboxes):
    index = bboxes.new_zeros((bboxes.size(0), 1)) # size() -- [33, 1]
    return torch.cat([index, bboxes[:, 0:4]], dim=1).float() # ==> size() -- [33, 5]


def split(image, mask, border_width=3, iou_thresh=0.25, patch_size=64, out_size=128):
    border = find_border(mask, border_width)
    bboxes = get_bboxes(border, patch_size, iou_thresh) # size() -- [70, 5]
    crop_list = roi_list(bboxes)

    image_patches = roi_align(image, crop_list, patch_size)
    image_patches = F.interpolate(image_patches, (out_size, out_size), mode='bilinear')

    mask_patches = roi_align(mask, crop_list, patch_size)
    mask_patches = F.interpolate(mask_patches, (out_size, out_size), mode='nearest')
    mask_patches = 2.0 * mask_patches - 1.0 # convert from [0.0, 1.0] ==> [-1.0, 1.0]

    # pdb.set_trace()

    return bboxes, torch.cat([image_patches, mask_patches], dim=1)


def merge(mask, bboxes, mask_patches, patch_size=64):
    mask_patches = F.interpolate(mask_patches, (patch_size, patch_size), mode='bilinear')
    target = mask.clone()    
    i = 0
    for b in bboxes:
        x1 = int(b[0].item())
        y1 = int(b[1].item())
        x2 = int(b[2].item())
        y2 = int(b[3].item())
        target[:, :, y1:y2, x1:x2] = (mask_patches[i:i+1, :, :] < 0.5).to(torch.float32)
    return target


def inference_one(model, image):
    color = image[:, 0:3, :, :]
    image_normal = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    color = image_normal(color)

    mask = image[:, 3:4, :, :]
    bboxes, input_patches = split(color, mask)

    with torch.no_grad():
        output_patches = model(input_patches)
    
    mask_patches = output_patches[:, 1:2, :, :]

    return merge(mask, bboxes, mask_patches)



def create_model():
    """
    Create model
    """

    model = refine_mask.RefineMask()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running model on {device} ...")

    return model, device


def get_model():
    """Load jit script model."""

    model, device = create_model()
    # print(model)

    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    # model = torch.jit.script(model)
    # todos.data.mkdir("output")
    # if not os.path.exists("output/RefineMask.torch"):
    #     model.save("output/RefineMask.torch")

    return model, device


def predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()
    transform = Compose([
        ToTensor(),
    ])

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    results = []
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        image = Image.open(filename).convert("RGBA")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # with torch.no_grad():
        #     output_tensor = model(input_tensor).cpu()
        
        refine_mask = inference_one(model, input_tensor)

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        output_tensor = input_tensor.clone()
        output_tensor[:, 3:4, :, :] = refine_mask

        todos.data.save_tensor([input_tensor, output_tensor], output_file)

    progress_bar.close()
    todos.model.reset_device()
