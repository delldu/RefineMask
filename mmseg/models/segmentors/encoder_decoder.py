import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.ops import resize
from .. import builder
import pdb

class EncoderDecoder(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 # neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                ):
        super(EncoderDecoder, self).__init__()
        # backbone = {'type': 'HRNetRefine', 'norm_cfg': {'type': 'SyncBN', 'requires_grad': True}, 'norm_eval': False, 'extra': {'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (2,), 'num_channels': (64,)}, 'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (2, 2), 'num_channels': (18, 36)}, 'stage3': {'num_modules': 3, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (2, 2, 2), 'num_channels': (18, 36, 72)}, 'stage4': {'num_modules': 2, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (2, 2, 2, 2), 'num_channels': (18, 36, 72, 144)}}}
        # decode_head = {'type': 'FCNHead', 'in_channels': [18, 36, 72, 144], 'in_index': (0, 1, 2, 3), 'channels': 270, 'input_transform': 'resize_concat', 'kernel_size': 1, 'num_convs': 1, 'concat_input': False, 'dropout_ratio': -1, 'num_classes': 2, 'norm_cfg': {'type': 'SyncBN', 'requires_grad': True}, 'align_corners': False, 'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}}
        # neck = None
        # auxiliary_head = None
        # train_cfg = None
        # test_cfg = {'mode': 'whole'}
        # pretrained = None

        self.backbone = builder.build_backbone(backbone)
        # self.backbone -- HRNetRefine()

        self._init_decode_head(decode_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""

        self.decode_head = builder.build_head(decode_head)
        # (Pdb) self.decode_head
        # FCNHead(
        #   input_transform=resize_concat, ignore_index=255, align_corners=False
        #   (loss_decode): CrossEntropyLoss()
        #   (conv_seg): Conv2d(270, 2, kernel_size=(1, 1), stride=(1, 1))
        #   (convs): Sequential(
        #     (0): ConvModule(
        #       (conv): Conv2d(270, 270, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #       (bn): SyncBatchNorm(270, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (activate): ReLU(inplace=True)
        #     )
        #   )
        # )

        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
