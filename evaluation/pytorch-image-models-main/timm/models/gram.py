"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman
"""
from typing import List, Optional
import os

import math
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, model_entrypoint

# Import GRAM
import sys
from gram_export.gram.utils.architecture import Architecture

__all__ = ['GRAM_Wrapper']  
# model_registry will add each entrypoint fn to this

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'gram224': _cfg(
        url=None,
        interpolation='bicubic', crop_pct=0.95),
}

def init_weights(m):
    if isinstance(m, nn.Linear):
        # fan_out = m.weight.size(0)  # fan-out
        init_range = 0.01
        m.weight.data.normal_(0, init_range)
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        # This fixes the TensorFlow prototxts, where Convolution layers are used to represent FC for efficiency.
        try:
            if m.dense_initializer:
                init_range = 0.01
                m.weight.data.normal_(0, init_range)
                m.bias.data.zero_()
            else:
                torch.nn.init.kaiming_normal_(m.weight, a=math.sqrt(5.0))
                # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels / m.groups
                # m.weight.data.normal_(0, math.sqrt(1.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
        except Exception:
            # Otherwise, weights are initialized as usual.
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels / m.groups
            m.weight.data.normal_(0, math.sqrt(1.0 / (3 * fan_out))) #  = sqrt(5.0)
            if m.bias is not None:
                m.bias.data.zero_()

    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()

# A wrapper to wrap gram code.
class GRAM_Wrapper(nn.Module):
    def __init__(
        self,
        prototxt=None,
        num_classes: int = 1000,
        in_chans: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.bn_momentum = kwargs["bn_momentum"] if "bn_momentum" in kwargs else 0.1 
        self.bn_epsilon = kwargs["bn_eps"] if "bn_eps" in kwargs else 1e-5
        self.drop_path_rate = kwargs["drop_path_rate"] if "drop_path_rate" in kwargs else 0.0 
        if prototxt is not None:
            self.model = Architecture(
                prototxt, 
                bn_momentum=self.bn_momentum, 
                bn_epsilon=self.bn_epsilon, 
                drop_connect_rate=self.drop_path_rate,
            )
            self.model.configure_drop_path_with_linear(self.drop_path_rate)
        else:
            self.model = None
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)
    
    def overrride_prototxt(self, prototxt=None):
        if prototxt is not None:
            del self.model
            self.model = Architecture(
                prototxt, 
                bn_momentum=self.bn_momentum, 
                bn_epsilon=self.bn_epsilon, 
                drop_connect_rate=self.drop_path_rate,
            )
            self.model.configure_drop_path_with_linear(self.drop_path_rate)
        else:
            pass

    def init_weights(self):
        if self.model is not None:
            self.model.apply(init_weights)

def _create_gram(variant, pretrained=False, **kwargs):
    assert not pretrained, NotImplementedError("Pretrained model not Available!") 
    return build_model_with_cfg(GRAM_Wrapper, variant, pretrained, **kwargs)

@register_model
def gram224(pretrained=False, **kwargs):
    """Constructs a GRAM model, with 224x224 inputs.
    """
    model_args = dict(num_classes=1000)
    return _create_gram("gram224", pretrained, **dict(model_args, **kwargs))
