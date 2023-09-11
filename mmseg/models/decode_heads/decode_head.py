from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.runner import auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy
import pdb

class BaseDecodeHead(nn.Module): # xxxx8888 , metaclass=ABCMeta
    """Base class for BaseDecodeHead.
    """
    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=(0, 1, 2, 3),
                 input_transform='resize_concat',
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 # sampler=None,
                 align_corners=False,
                ):
        super(BaseDecodeHead, self).__init__()
        # in_channels = [18, 36, 72, 144]
        # channels = 270
        # num_classes = 2
        # dropout_ratio = -1
        # conv_cfg = None
        # norm_cfg = {'type': 'SyncBN', 'requires_grad': True}
        # act_cfg = {'type': 'ReLU'}
        # in_index = (0, 1, 2, 3)
        # input_transform = 'resize_concat'
        # loss_decode = {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}
        # ignore_index = 255
        # sampler = None
        # align_corners = False

        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0: # False
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        # # self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.
        """
        self.input_transform = input_transform
        self.in_index = in_index
        self.in_channels = sum(in_channels)

    def init_weights(self):
        """Initialize weights of classification layer."""
        normal_init(self.conv_seg, mean=0, std=0.01)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        """
        # self.input_transform == 'resize_concat'
        inputs = [inputs[i] for i in self.in_index]
        upsampled_inputs = [
            resize(
                input=x,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for x in inputs
        ]
        inputs = torch.cat(upsampled_inputs, dim=1)
        # ==> pdb.set_trace()
        return inputs

    # @auto_fp16()
    # @abstractmethod
    # def forward(self, inputs):
    #     """Placeholder of forward function."""
    #     pass

    # def forward_test(self, inputs, img_metas, test_cfg):
    #     """Forward function for testing.
    #     """
    #     return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output
 