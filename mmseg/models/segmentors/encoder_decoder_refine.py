import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
import pdb

@SEGMENTORS.register_module()
class EncoderDecoderRefine(EncoderDecoder):
    def __init__(self,
                 backbone,
                 decode_head,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
            ):
        super(EncoderDecoderRefine, self).__init__(
            backbone,
            decode_head,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )
        # backbone = {'type': 'HRNetRefine', 'norm_cfg': {'type': 'SyncBN', 'requires_grad': True}, 'norm_eval': False, 'extra': {'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (2,), 'num_channels': (64,)}, 'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (2, 2), 'num_channels': (18, 36)}, 'stage3': {'num_modules': 3, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (2, 2, 2), 'num_channels': (18, 36, 72)}, 'stage4': {'num_modules': 2, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (2, 2, 2, 2), 'num_channels': (18, 36, 72, 144)}}}
        # decode_head = {'type': 'FCNHead', 'in_channels': [18, 36, 72, 144], 'in_index': (0, 1, 2, 3), 'channels': 270, 'input_transform': 'resize_concat', 'kernel_size': 1, 'num_convs': 1, 'concat_input': False, 'dropout_ratio': -1, 'num_classes': 2, 'norm_cfg': {'type': 'SyncBN', 'requires_grad': True}, 'align_corners': False, 'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}}
        # auxiliary_head = None
        # train_cfg = None
        # test_cfg = {'mode': 'whole'}
        # pretrained = None

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.
        Returns:
            Tensor: The output segmentation map.
        """
        # img.size() -- [33, 4, 128, 128]
        # img_meta = [{'ori_shape': (64, 64), 'flip': False}]
        # rescale = False

        x = self.backbone(img)
        x = self.decode_head(x)
        x = resize(
            input=x,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        return F.softmax(x, dim=1) # size() -- [33, 2, 128, 128]
