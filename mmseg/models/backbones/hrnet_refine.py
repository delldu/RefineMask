import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

# from mmseg.ops import Upsample, resize
# from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from .hrnet import HRNet
import pdb

# xxxx1111
@BACKBONES.register_module()
class HRNetRefine(HRNet):
    """HRNet backbone (add coarse mask input).
    """
    def __init__(self, *args, **kwargs):
        super(HRNetRefine, self).__init__(*args, **kwargs)
        # args = ()
        # kwargs = {'norm_cfg': {'type': 'SyncBN', 'requires_grad': True}, 
        #             'norm_eval': False, 'extra': 
        #     {'stage1': {'num_modules': 1, 'num_branches': 1, 
        #         'block': 'BOTTLENECK', 'num_blocks': (2,), 'num_channels': (64,)}, 
        #     'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': 
        #         (2, 2), 'num_channels': (18, 36)}, 
        #     'stage3': {'num_modules': 3, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': 
        #         (2, 2, 2), 'num_channels': (18, 36, 72)}, 
        #     'stage4': {'num_modules': 2, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': 
        #         (2, 2, 2, 2), 'num_channels': (18, 36, 72, 144)}}}

        # pp self.conv_cfg -- None
        self.conv_mask = build_conv_layer(
            self.conv_cfg,
            1,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        # self.conv_mask -- Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        """Forward function."""
        img = x[:,:3,...]
        mask = x[:,3:,...]

        x = self.conv1(img) + self.conv_mask(mask)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']): # 2
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']): # 3
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']): # 4
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # len(y_list) -- 4

        # xxxx8888
        return y_list
