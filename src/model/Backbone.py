import random
import math
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

from model.resnet import ResNet
from model.vgg import VGG
from model.mobilenetv2 import MobileNetV2
from model.swin_transformer import SwinTransformer
from model.resnet import My_ResNet
from torchvision.utils import _log_api_usage_once

def get_backbone(backbone_name):
    if 'resnet' in backbone_name:
        return ResNet(backbone_name)
        # return My_ResNet(backbone_name)
    elif 'vgg' in backbone_name:
        return VGG(backbone_name)
    elif 'mobilenetv2' in backbone_name:
        return MobileNetV2()
    elif 'swin_T' in backbone_name:
        model = SwinTransformer()
        pretrain = os.path.join(os.environ["TORCH_HOME"], 'models/swin/swin_tiny_patch4_window7_224.pth')
        sd = torch.load(pretrain)['model']
        model.load_state_dict(sd, strict=False)
        return model

def get_channels(backbone_name):
    if 'resnet' in backbone_name:
        d = {
            'resnet18': 64,
            'resnet50': 256,
        }
        channel = d[backbone_name]
        return [channel, channel, channel * 2, channel * 4, channel * 8]
    elif 'vgg' in backbone_name:
        return [64, 128, 256, 512, 512]
    elif 'mobilenetv2' in backbone_name:
        return [16, 24, 32, 96, 1280]
    elif 'swin_T' in backbone_name:
        return [int(96 * 2 ** i) for i in range(4)]
    
class MTF(nn.Module):
    def __init__(self, channel, mode='iade', kernel_size=1):
        super(MTF, self).__init__()
        assert mode in ['i', 'a', 'd', 'e', 'ia', 'id', 'ie', 'iae', 'ide', 'iad', 'iade', 'i2ade', 'iad2e', 'i2ad2e', 'i2d']
        self.mode = mode
        self.channel = channel
        self.relu = nn.ReLU(inplace=True)
        if kernel_size == 1:
            padding = 0
        elif kernel_size == 3:
            padding = 1
        if 'i2' in mode:
            self.i0 = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
            self.i1 = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
        else:
            self.conv = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
            
        if 'ad2'in mode:
            self.app = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
            self.dis = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
        else:
            self.res = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
            
        self.exchange = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
        print("MTF: mode: {} kernel_size: {}".format(self.mode, kernel_size))
        
    def forward(self, f0, f1):
        #t0 = self.conv(f0)
        #t1 = self.conv(f1)
        if 'i2' in self.mode:
            info = self.i0(f0) + self.i1(f1)
        else:
            info = self.conv(f0 + f1)
            
        if 'd' in self.mode:
            if 'ad2' in self.mode:
                disappear = self.dis(self.relu(f0 - f1))
            else:
                disappear = self.res(self.relu(f0 - f1))
        else:
            disappear = 0

        if 'a' in self.mode:
            if 'ad2' in self.mode:
                appear = self.app(self.relu(f1 - f0))
            else:
                appear = self.res(self.relu(f1 - f0))
        else:
            appear = 0

        if 'e' in self.mode:
            exchange = self.exchange(torch.max(f0, f1) - torch.min(f0, f1))
        else:
            exchange = 0

        if self.mode == 'i':
            f = info
        elif self.mode == 'a':
            f = appear
        elif self.mode == 'd':
            f = disappear
        elif self.mode == 'e':
            f = exchange
        elif self.mode == 'ia':
            f = info + 2 * appear
        elif self.mode in ['id', 'i2d']:
            f = info + 2 * disappear
        elif self.mode == 'ie':
            f = info + 2 * exchange
        elif self.mode == 'iae':
            f = info + appear + exchange
        elif self.mode == 'ide':
            f = info + disappear + exchange
        elif self.mode == 'iad':
            f = info + disappear + appear
        elif self.mode in ['iade', 'i2ade', 'iad2e', 'i2ad2e']:
            f = info + disappear + appear + exchange

        f = self.relu(f)
        return f

        
        
class MSF(nn.Module):
    def __init__(self, channels, total_f=5, fpn_channel=None, with_bn=False, mode='iade'):
        super(MSF, self).__init__()
        print("MSF: {}".format(channels))
        self.num_f = len(channels)
        self.total_f = total_f
        assert 0 < self.num_f <= self.total_f
        cf_list = []
        cf_inner = []
        cf_layer = []
        for i in range(self.num_f):
            cf_list.append(MTF(channels[i], mode, kernel_size=3))
            cf_inner.append(self._make_layer(channels[i], fpn_channel, 1, with_bn))
            cf_layer.append(self._make_layer(fpn_channel, fpn_channel, 3, with_bn))

        self.cfs = nn.ModuleList(cf_list)
        self.cf_inners = nn.ModuleList(cf_inner)
        self.cf_layers = nn.ModuleList(cf_layer)

        self.reduce = nn.Conv2d(fpn_channel * self.num_f, fpn_channel, 3, padding=1, stride=1, bias=False)
        self.bn   = nn.BatchNorm2d(fpn_channel)
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, in_channel, out_channel, kernel, with_bn):
        l = []
        if kernel == 1:
            l.append(nn.Conv2d(in_channel, out_channel, 1))
        elif kernel == 3:
            l.append(nn.Conv2d(in_channel, out_channel, 3, padding=1))
        else:
            raise ValueError(kernel)
        
        if with_bn:
            l.append(nn.BatchNorm2d(out_channel))

        l.append(nn.ReLU(inplace=True))
        return nn.Sequential(*l)

    def forward(self, t0_fs, t1_fs=None):
        cfs = []
        for i in range(self.num_f):
            k = i + self.total_f - self.num_f
            if t1_fs is None:
                cfs.append(self.cfs[i](t0_fs[k], torch.zeros_like(t0_fs[k])))
            else:
                cfs.append(self.cfs[i](t0_fs[k], t1_fs[k]))

        resize_shape = cfs[0].shape[-2:]
        final_list = []
        last_inner = None
        for i in range(self.num_f - 1, -1, -1):
            cf = self.cf_inners[i](cfs[i])
            if last_inner is None:
                last_inner = cf
            else:
                inner_top_down = F.interpolate(last_inner, size=cf.shape[-2:], mode="nearest")
                last_inner = cf + inner_top_down
            cf_layer = self.cf_layers[i](last_inner)
            final_list.append(cf_layer)

        final_list = [F.interpolate(cf_layer, resize_shape, mode='bilinear') for cf_layer in final_list]
        cf = torch.cat(final_list, dim=1)
        cf = self.relu(self.bn(self.reduce(cf)))
        return cf

class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.

    The feature maps are currently supposed to be in increasing depth
    order.

    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.

    Args:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names

    Examples::

        >>> m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]

    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        mode: str,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.mtf = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
            self.mtf.append(MTF(in_channels, mode, kernel_size=3))

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # if extra_blocks is not None:
        #     assert isinstance(extra_blocks, ExtraFPNBlock)
        # self.extra_blocks = extra_blocks

    def get_result_from_mtf_inner_blocks(self, f0, f1, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
            
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                merge = self.mtf[i](f0, f1)
                out = self.inner_blocks[i](merge)
                
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, f0, f1) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # # unpack OrderedDict into two lists for easier handling
        # names = list(x.keys())
        # x = list(x.values())
        

        last_inner = self.get_result_from_mtf_inner_blocks(f0[-1], f1[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(f0) - 2, -1, -1):
            inner_lateral = self.get_result_from_mtf_inner_blocks(f0[idx], f1[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # if self.extra_blocks is not None:
        #     results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        names = ['0', '1', '2', '3', '4']
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class UniBackbone_FPN(nn.Module):
    def __init__(self, backbone_name, combinefeature):
        super(UniBackbone_FPN, self).__init__()
        self.encoder = get_backbone(backbone_name)
        self.combinefeature = combinefeature
        
    def forward(self, img):
        out = OrderedDict()
        fs = self.encoder(img)
        out['out'] = self.combinefeature(fs)
        return out

class MTF_Backbone(nn.Module):
    def __init__(self, backbone_name, combinefeature, share_weight=False, mode='i'):
        super(MTF_Backbone, self).__init__()
        self.input_MTF = MTF(3, mode, kernel_size=3)
        self.share_weight = share_weight
        self.encoder = get_backbone(backbone_name)
        self.combinefeature = combinefeature
        
    def forward(self, img):
        out = OrderedDict()
        img_t0, img_t1 = torch.split(img,3,1)
        img = self.input_MTF(img_t0, img_t1)
        fs = self.encoder(img_t0)
        out['out'] = self.combinefeature(fs)
        return out

class Backbone_MTF_MSF(nn.Module):
    def __init__(self, backbone_name, combinefeature, share_weight=False):
        super(Backbone_MTF_MSF, self).__init__()
        self.share_weight = share_weight
        if share_weight:
            self.encoder = get_backbone(backbone_name)
        else:
            self.encoder1 = get_backbone(backbone_name)
            self.encoder2 = get_backbone(backbone_name)
        self.combinefeature = combinefeature
        
    def forward(self, img):
        out = OrderedDict()
        img_t0, img_t1 = torch.split(img,3,1)
        bs = img_t0.size(0)
        img_cat = torch.cat([img_t0, img_t1], 0)
        if self.share_weight:
            # t0_fs = self.encoder(img_t0)
            # t1_fs = self.encoder(img_t1)
            fs = self.encoder(img_cat)
            t0_fs = [fs[i][:bs] for i in range(len(fs))]
            t1_fs = [fs[i][bs:] for i in range(len(fs))]
        else:
            t0_fs = self.encoder1(img_t0)
            t1_fs = self.encoder2(img_t1)
        out = self.combinefeature(t0_fs, t1_fs)
        return out

class Backbone_MSF_MTF(Backbone_MTF_MSF):
    def __init__(self, backbone_name, combinefeature, share_weight=False, mode='iade'):
        super(Backbone_MSF_MTF, self).__init__(backbone_name, combinefeature, share_weight)
        self.MTF = MTF(512, mode, kernel_size=3)
        
    def forward(self, img):
        out = OrderedDict()
        img_t0, img_t1 = torch.split(img,3,1)
        if self.share_weight:
            t0_fs = self.encoder(img_t0)
            t1_fs = self.encoder(img_t1)
        else:
            t0_fs = self.encoder1(img_t0)
            t1_fs = self.encoder2(img_t1)

        t0_out = self.combinefeature(t0_fs)
        t1_out = self.combinefeature(t1_fs)
        out['out'] = self.MTF(t0_out, t1_out)
        return out

class Backbone_MSF(Backbone_MTF_MSF):
    def __init__(self, backbone_name, combinefeature, share_weight=False):
        super(Backbone_MSF, self).__init__(backbone_name, combinefeature, share_weight)
        
    def forward(self, img):
        out = OrderedDict()
        img_t0, img_t1 = torch.split(img,3,1)
        if self.share_weight:
            t0_fs = self.encoder(img_t0)
            t1_fs = self.encoder(img_t1)
        else:
            t0_fs = self.encoder1(img_t0)
            t1_fs = self.encoder2(img_t1)

        t0_out = self.combinefeature(t0_fs)
        t1_out = self.combinefeature(t1_fs)
        out['t0_out'] = t0_out
        out['t1_out'] = t1_out
        return out


def unibackbone_fpn(backbone_name, fpn_num=4):
    # COCO pretrained model
    channels = get_channels(backbone_name)
    combinefeature = MSF(channels[-fpn_num:], total_f=len(channels), fpn_channel=512, with_bn=False, mode='i')
    model = UniBackbone_FPN(backbone_name, combinefeature)
    return model

def mtf_backbone_msf(backbone_name, fpn_num=4, mode='iade'):
    # T -> B -> S -> H 
    channels = get_channels(backbone_name)
    combinefeature = MSF(channels[-fpn_num:], total_f=len(channels), fpn_channel=512, with_bn=False, mode='i')
    model = MTF_Backbone(backbone_name, combinefeature, mode=mode)
    return model

def backbone_mtf_msf(backbone_name, fpn_num=4, mode='iade'):
    # B -> T -> S -> H 
    channels = get_channels(backbone_name)
    combinefeature = MSF(channels[-fpn_num:], total_f=len(channels), fpn_channel=512, with_bn=False, mode=mode)
    model = Backbone_MTF_MSF(backbone_name, combinefeature, share_weight=True)
    return model

def backbone_mtf_fpn(backbone_name, fpn_num=4, mode='iade'):
    # B -> T -> S -> H 
    channels = get_channels(backbone_name)
    in_channels = [256, 512, 1024, 2048]
    combinefeature = FeaturePyramidNetwork(in_channels, 256, mode=mode)
    model = Backbone_MTF_MSF(backbone_name, combinefeature, share_weight=True)
    return model

def bibackbone_mtf_msf(backbone_name, fpn_num=4, mode='iade'):
    # B -> T -> S -> H 
    channels = get_channels(backbone_name)
    combinefeature = MSF(channels[-fpn_num:], total_f=len(channels), fpn_channel=512, with_bn=False, mode=mode)
    model = Backbone_MTF_MSF(backbone_name, combinefeature, share_weight=False)
    return model

def backbone_msf_mtf(backbone_name, fpn_num=4, mode='iade'):
    # B -> S -> T -> H 
    channels = get_channels(backbone_name)
    combinefeature = MSF(channels[-fpn_num:], total_f=len(channels), fpn_channel=512, with_bn=False, mode='i')
    model = Backbone_MSF_MTF(backbone_name, combinefeature, share_weight=True, mode=mode)
    return model

def backbone_msf(backbone_name, fpn_num=4):
    # B -> S -> H -> T
    channels = get_channels(backbone_name)
    combinefeature = MSF(channels[-fpn_num:], total_f=len(channels), fpn_channel=512, with_bn=False, mode='i')
    model = Backbone_MSF(backbone_name, combinefeature, share_weight=True)
    return model


