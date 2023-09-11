# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Tue 12 Sep 2023 12:36:45 AM CST
# ***
# ************************************************************************************/
#

import pdb
import os
import time
import torch
import RefineMask

from tqdm import tqdm

if __name__ == "__main__":
    model, device = RefineMask.get_model()

    N = 100
    B, C, H, W = 1, 3, 384, 384

    mean_time = 0
    progress_bar = tqdm(total=N)
    for count in range(N):
        progress_bar.update(1)

        x = torch.randn(B, C, H, W)
        # print("x: ", x.size())

        start_time = time.time()
        with torch.no_grad():
            y = model(x.to(device))
        torch.cuda.synchronize()
        mean_time += time.time() - start_time

    mean_time /= N
    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")
