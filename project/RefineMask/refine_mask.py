# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by RainbowSecret (yhyuan@pku.edu.cn)
# ------------------------------------------------------------------------------
# The following code main comes from 
# https://github.com/HRNet/HRNet-Semantic-Segmentation.git/hrnet.py
# Thanks guys !!!

from yacs.config import CfgNode as CN

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from typing import Dict, Tuple
import pdb

logger = logging.getLogger('HRNet')

class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0):
        super().__init__()
        # in_channels = 270
        # out_channels = 270

        self.conv = nn.Conv2d(in_channels, out_channels, 
            kernel_size=(kernel_size, kernel_size), stride=(stride, stride), bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1, affine=True)
        self.activate = nn.ReLU() 

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):
    def __init__(self,
                 size=None,
                 scale_factor=2.0,
                 mode='nearest',
                 align_corners=False):
        super(Upsample, self).__init__()
        # size = None
        # scale_factor = 2
        # mode = 'bilinear'
        # align_corners = False

        self.size = size
        self.scale_factor = float(scale_factor)
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size: #True
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)


class FCNHead(nn.Module):
    """Fully Convolution Networks for Semantic Segmentation."""
    def __init__(self,
                 in_channels=[18, 36, 72, 144],
                 num_convs=1,
                 kernel_size=1,
                 num_classes = 2,
                 in_index=(0, 1, 2, 3),
                 align_corners=False,
                ):
        super(FCNHead, self).__init__()

        self.in_index = in_index
        self.channels = sum(in_channels)
        self.in_index = in_index
        self.align_corners = align_corners

        self.conv_seg = nn.Conv2d(self.channels, num_classes, kernel_size=kernel_size)


        convs = []
        convs.append(
            ConvModule(
                self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2))
        # for i in range(num_convs - 1): # for num_convs == 1
        #     convs.append(
        #         ConvModule(
        #             self.channels,
        #             self.channels,
        #             kernel_size=kernel_size,
        #             padding=kernel_size // 2,
        #             ))
        self.convs = nn.Sequential(*convs)


    def transform_inputs(self, inputs):
        """Transform inputs for decoder.
        """
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

    # def forward_test(self, inputs, img_metas, test_cfg):
    #     """Forward function for testing.
    #     """
    #     return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        # if self.dropout is not None:
        #     feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output
 
    def forward(self, inputs):
        """Forward function."""
        x = self.transform_inputs(inputs)
        output = self.convs(x)
        output = self.cls_seg(output)
        return output


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        # self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        # self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, norm_layer=None):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU()

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index], norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, 
                            bias=False),
                        self.norm_layer(num_inchannels[i]),
                        Upsample(scale_factor=2**(j - i), mode='bilinear', 
                            align_corners=False)
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3),
                                nn.ReLU()))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=True
                        )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

class HighResolutionNet(nn.Module):

    def __init__(self, cfg):
        super(HighResolutionNet, self).__init__()

        self.norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = self.norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = self.norm_layer(64)
        self.relu = nn.ReLU()

        # stage 1
        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        # stage 2
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        self.conv_mask = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i],
                                  3, 1, 1, bias=False),
                        self.norm_layer(num_channels_cur_layer[i]),
                        nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        self.norm_layer(outchannels),
                        nn.ReLU()))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output,
                                     norm_layer=self.norm_layer)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels


    def forward(self, x):
        """Forward function."""
        img = x[:,:3,...]
        mask = x[:,3:,...]

        x = self.conv1(img) + self.conv_mask(mask)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage4(x_list)

        # len(y_list) -- 4
        return y_list


class RefineMask(nn.Module):
    def __init__(self):
        super(RefineMask, self).__init__()
        HRNET_18 = CN()

        HRNET_18.STAGE1 = CN()
        HRNET_18.STAGE1.NUM_MODULES = 1
        HRNET_18.STAGE1.NUM_BRANCHES = 1
        HRNET_18.STAGE1.NUM_BLOCKS = [2]
        HRNET_18.STAGE1.NUM_CHANNELS = [64]
        HRNET_18.STAGE1.BLOCK = 'BOTTLENECK'
        HRNET_18.STAGE1.FUSE_METHOD = 'SUM'

        HRNET_18.STAGE2 = CN()
        HRNET_18.STAGE2.NUM_MODULES = 1
        HRNET_18.STAGE2.NUM_BRANCHES = 2
        HRNET_18.STAGE2.NUM_BLOCKS = [2, 2]
        HRNET_18.STAGE2.NUM_CHANNELS = [18, 36]
        HRNET_18.STAGE2.BLOCK = 'BASIC'
        HRNET_18.STAGE2.FUSE_METHOD = 'SUM'

        HRNET_18.STAGE3 = CN()
        HRNET_18.STAGE3.NUM_MODULES = 3
        HRNET_18.STAGE3.NUM_BRANCHES = 3
        HRNET_18.STAGE3.NUM_BLOCKS = [2, 2, 2]
        HRNET_18.STAGE3.NUM_CHANNELS = [18, 36, 72]
        HRNET_18.STAGE3.BLOCK = 'BASIC'
        HRNET_18.STAGE3.FUSE_METHOD = 'SUM'

        HRNET_18.STAGE4 = CN()
        HRNET_18.STAGE4.NUM_MODULES = 2
        HRNET_18.STAGE4.NUM_BRANCHES = 4
        HRNET_18.STAGE4.NUM_BLOCKS = [2, 2, 2, 2]
        HRNET_18.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
        HRNET_18.STAGE4.BLOCK = 'BASIC'
        HRNET_18.STAGE4.FUSE_METHOD = 'SUM'

        self.backbone = HighResolutionNet(HRNET_18)
        self.decode_head = FCNHead()

        self.load_weights()

    def forward(self, image):
        B, C, H, W = image.size()
        x = self.backbone(image)
        x = self.decode_head(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return F.softmax(x, dim=1) # size() -- [33, 2, 128, 128]

    def load_weights(self, model_path="models/RefineMask.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        if os.path.exists(checkpoint):
            print(f"Loading weight from {checkpoint} ...")
            self.load_state_dict(torch.load(checkpoint)['state_dict'])
        else:
            print("-" * 32, "Warnning", "-" * 32)
            print(f"Weight file '{checkpoint}' not exist !!!")

# if __name__ == "__main__":
#     model = RefineMask()
#     model = model.eval()

#     # image = torch.randn(33, 4, 128, 128)
#     # with torch.no_grad():
#     #     y = model(image)

#     # print(y.size()) # size() -- [33, 2, 128, 128]

#     print(model)