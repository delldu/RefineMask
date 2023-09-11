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
import todos
from RefineMask import refine_mask
from torchvision.transforms import Compose, ToTensor
import pdb

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

        with torch.no_grad():
            output_tensor = model(input_tensor).cpu()

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        input_tensor[:, 3:4, :, :] = 1.0
        todos.data.save_tensor([input_tensor], output_file)

    progress_bar.close()


    todos.model.reset_device()
