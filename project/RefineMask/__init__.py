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

    hi_mask = (mask > 0.90).to(torch.float32) # xxxx8888

    kernel = mask.new_ones((1, 1, kernel_size, kernel_size))
    border = F.conv2d(hi_mask, kernel, stride=1, padding=kernel_size//2)

    bml = torch.abs(border - kernel_size*kernel_size)
    bms = torch.abs(border)
    border = torch.min(bml, bms) / (kernel_size*kernel_size/2)
    # tensor [border] size: [1, 1, 1024, 2048] , min: 0.0 , max: 0.8888888955116272

    return border

def get_bboxes(border, patch_size=64, iou_thresh=0.25):
    """boundaries of coarse mask -> patch bboxs
    """
    B, C, H, W = border.size()
    assert B == 1 and C == 1, "Only support one batch size! "

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
    index = bboxes.new_zeros((bboxes.size(0), 1))
    return torch.cat([index, bboxes[:, 0:4]], dim=1).float() # size() [33, 1] ==> [33, 5]

def split(image, mask, border_width=3, patch_size=64, out_size=128):
    border = find_border(mask, border_width)
    bboxes = get_bboxes(border, patch_size)
    # tensor [bboxes] size: [33, 5] , min: 0.6666666865348816 , max: 1252.0

    crop_list = roi_list(bboxes)

    image_patches = roi_align(image, crop_list, patch_size)
    image_patches = F.interpolate(image_patches, (out_size, out_size), mode='bilinear')
    # tensor [image_patches] size: [33, 3, 128, 128] , min: -1.9188287258148193 , max: 2.248908281326294

    mask_patches = roi_align(mask, crop_list, patch_size)
    mask_patches = F.interpolate(mask_patches, (out_size, out_size), mode='nearest')
    # tensor [mask_patches] size: [33, 1, 128, 128] , min: 0.0 , max: 1.0
    mask_patches = 2.0 * mask_patches - 1.0 # convert from [0.0, 1.0] ==> [-1.0, 1.0]

    return bboxes, torch.cat([image_patches, mask_patches], dim=1)


def merge(mask, bboxes, mask_patches, patch_size=64):
    mask_patches = F.interpolate(mask_patches, (patch_size, patch_size), mode='bilinear')

    target = mask.clone()
    mask_sum = torch.zeros_like(mask)
    mask_count =torch.zeros_like(mask)

    box_list = bboxes[:, 0:4].int()
    for i, b in enumerate(box_list):
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]

        m_source = mask[:, :, y1:y2, x1:x2]
        m_refine = mask_patches[i:i+1, :, :, :]

        mask_sum[:, :, y1:y2, x1:x2] += m_source*m_refine # xxxx8888
        mask_count[:, :, y1:y2, x1:x2] += 1.0

    s = mask_count > 0
    mask_sum[s] /= mask_count[s]

    target[s] = mask_sum[s]
    return target


def inference_one(model, image):
    color = image[:, 0:3, :, :]
    mask = image[:, 3:4, :, :]

    image_normal = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    color = image_normal(color)

    bboxes, input_patches = split(color, mask)

    with torch.no_grad():
        output_patches = model(input_patches) # size() -- [33, 4, 128, 128]
    
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
        # output_tensor[:, 3:4, :, :] = 1.0 - find_border(input_tensor[:, 3:4, :, :])
        output_tensor[:, 3:4, :, :] = refine_mask
        # output_tensor[:, 3:4, :, :] = find_border(input_tensor[:, 3:4, :, :])

        todos.data.save_tensor([input_tensor, output_tensor], output_file)

    progress_bar.close()
    todos.model.reset_device()
