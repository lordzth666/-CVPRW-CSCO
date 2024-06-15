"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman
"""
from typing import List
import os

import math
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, create_attn, get_attn, \
    get_act_layer, get_norm_layer, create_classifier
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, model_entrypoint

__all__ = ['ConvCoder', 'ConvCoderBottleneck']  
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
    'convcoder50': _cfg(
        url=None,
        interpolation='bicubic', crop_pct=0.95),
    'convcoder101': _cfg(
        url=None,
        interpolation='bicubic', crop_pct=0.95),
}

# ---------------ConvCoder Utilities------------------- #
# ---------------- START ------------------------------ #
def unit_matrix(in_planes: int, out_planes: int, kernel_size: int) -> torch.Tensor:
    matrix = np.zeros([out_planes, in_planes, kernel_size, kernel_size], dtype=np.float32)
    planes: int = min(in_planes, out_planes)
    k_center: int = kernel_size // 2
    id_kernel_matrix = np.zeros([kernel_size, kernel_size], dtype=np.float32)
    id_kernel_matrix[k_center, k_center] = 1.0
    matrix[np.arange(planes), np.arange(planes), :, :] = id_kernel_matrix
    return torch.from_numpy(matrix)


def kaiming_normal_matrix(in_planes, out_planes, kernel_size):
    weight = torch.zeros([out_planes, in_planes, kernel_size, kernel_size], dtype=torch.float)
    torch.nn.init.kaiming_normal_(weight.data, a=math.sqrt(5.0))
    return weight


def zero_matrix(in_planes: int, out_planes: int, kernel_size: int) -> torch.Tensor:
    matrix = np.zeros([out_planes, in_planes, kernel_size, kernel_size], dtype=np.float32)
    return torch.from_numpy(matrix)


def load_matrix_pair(in_planes, gen_planes, kernel_size, model_dir, inst="normal") -> torch.Tensor:
    if inst == "normal":
        conv1_save_name = os.path.join(model_dir,
            "plain_{}_{}.conv1.npy".format(in_planes, gen_planes))
        conv2_save_name = os.path.join(model_dir,
            "plain_{}_{}.conv2.npy".format(in_planes, gen_planes))
        print("Loading plain_{}_{}.conv1/2.npy".format(in_planes, gen_planes))
        conv1_mat = np.load(conv1_save_name)
        conv2_mat = np.load(conv2_save_name)
    elif inst == "id":
        print("Use id matrix.")
        conv1_mat = unit_matrix(in_planes, gen_planes, kernel_size)
        conv2_mat = unit_matrix(gen_planes, in_planes, kernel_size)
    elif inst == "kaiming":
        print("Warning: use Kaiming as loaded kernels...")
        conv1_mat = kaiming_normal_matrix(in_planes, gen_planes, kernel_size)
        conv2_mat = kaiming_normal_matrix(gen_planes, in_planes, kernel_size)
    return torch.from_numpy(conv1_mat), torch.from_numpy(conv2_mat)
# ---------------- END-- ------------------------------ #


# ---------------ConvCoder Main Files------------------ #
# ---------------- START ------------------------------ #
def unset_requires_grad(m):
    for param in m.parameters():
        param.requires_grad = False

def set_requires_grad(m):
    for param in m.parameters():
        param.requires_grad = True


EXPANSION_FAC = 4
class ConvCoderBottleneck(nn.Module):
    def __init__(
            self,
            in_planes,
            out_planes,
            strides,
            dropout = 0.0,
            drop_path: float = 0.0,
            bot_ratio: float = 4. / 8.,
            mode: str = 'rep',
            use_bn: bool = True,
            use_gen: bool = True,
    ):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.strides = strides
        self.dropout = dropout
        self.drop_path = drop_path
        self.mode = mode
        self.use_bn = use_bn
        self.use_gen = use_gen
        self.bot_ratio = bot_ratio

        # Rep planes for 3x3 layer.
        self.rep_planes_3x3 = int(self.out_planes * self.bot_ratio // EXPANSION_FAC) 
        self.gen_planes_3x3 = int(self.out_planes * (1. - self.bot_ratio) // EXPANSION_FAC)

        self.squeezed_planes = int(out_planes / EXPANSION_FAC)

        # Conv 1
        # [squeeze_planes, in_planes, ksize, ksize]
        self.conv1 = nn.Conv2d(in_planes, self.squeezed_planes, 1, stride=1, padding=0, bias=not self.use_bn)
        self.relu1 = nn.ReLU(True)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(self.squeezed_planes, eps=1e-5, momentum=0.1)
            self.bn2 = nn.BatchNorm2d(self.squeezed_planes, eps=1e-5, momentum=0.1)
        else:
            self.bn1 = None
            self.bn2 = None

        # Conv 2
        self.conv2 = nn.Conv2d(
            self.squeezed_planes, self.squeezed_planes, 3, stride=self.strides, padding=1, bias=not self.use_bn)
        self.relu2 = nn.ReLU(True)


        # cONV 3
        self.conv3 = nn.Conv2d(
            self.squeezed_planes, out_planes, 1, stride=1, padding=0, bias=not (self.use_bn and not self.use_gen))
        if self.use_bn and not self.use_gen:
            self.bn3 = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.1)
        else:
            self.bn3 = None
        self.relu3 = nn.ReLU(True)

        # Global training steps.
        self.global_steps = 0
        # Initialize gradient masking.
        self.conv1_grad_mask = None
        self.conv2_grad_mask = None
        self.conv3_grad_mask = None
        self.conv1_mask = None
        self.conv2_mask = None

    def _block_forward(self, x):
        out = self.conv1(x)        
        if self.bn1 is not None:
            if not self.use_gen:
                out = self.bn1(out)
            else:
                out = self.bn1(out) * self.conv1_mask + out * (1. - self.conv1_mask)
        out = self.relu1(out)

        # Apply dropout.
        if self.dropout != 0.0:
            out = self.dropout_nn(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            if not self.use_gen:
                out = self.bn2(out)
            else:
                out = self.bn2(out) * self.conv2_mask + out * (1. - self.conv2_mask)
        out = self.relu2(out)

        out = self.conv3(out)
        if self.bn3 is not None:
            out = self.bn3(out)
        out = self.relu3(out)
        return out

    def forward(self, x):
        if self.training:
            self.global_steps += 1
        if not self.training or abs(self.drop_path) < 1e-5 or self.strides != 1:
            return self._block_forward(x)
        else:
            out = self._block_forward(x)
            drop_path_mask = np.random.uniform(size=[out.size(0), 1, 1, 1])
            drop_path_mask = torch.from_numpy(np.floor(drop_path_mask + self.drop_path).astype(np.float32)).to(out.device)
            return x * drop_path_mask + out * (1 - drop_path_mask)

    def init_weights(self):
        if self.use_gen:
            # Load regular weight matrix
            conv1_normal_mat = kaiming_normal_matrix(self.in_planes, self.rep_planes_3x3, 1)
            # [rep_3x3, squeezed, 3, 3]
            conv2_normal_mat = kaiming_normal_matrix(self.squeezed_planes, self.rep_planes_3x3, 3)
            conv3_normal_mat = kaiming_normal_matrix(self.rep_planes_3x3, self.out_planes, 1)
            # Load fixed weight matrix
            k_path = "./convcoder_export/imagenet_1x1"

            # Loading weights. Conv1/3 use pretrained kernels. Conv2 uses all 1s.
            conv1_unit_mat, conv3_unit_mat = load_matrix_pair(self.in_planes, self.gen_planes_3x3, 1, k_path)
            
            conv2_unit_mat = unit_matrix(self.gen_planes_3x3, self.gen_planes_3x3, 3)
            conv2_zeros_mat = zero_matrix(self.rep_planes_3x3, self.gen_planes_3x3, 3)

            # [gen_planes_3x3, squeezed, 3, 3]

            conv1_weight = torch.cat([conv1_normal_mat, conv1_unit_mat], dim=0).float()
            self.conv1.weight.data = conv1_weight

            conv2_gen_mat = torch.cat([conv2_zeros_mat, conv2_unit_mat], dim=1) # [gen_planes, squeezed, 3, 3]

            conv2_weight = torch.cat([conv2_normal_mat, conv2_gen_mat], dim=0).float()  # [squeezed, squeezed, 3, 3]
            self.conv2.weight.data = conv2_weight

            conv3_weight = torch.cat([conv3_normal_mat, conv3_unit_mat], dim=1).float()
            self.conv3.weight.data = conv3_weight
            
            # Gradient
            conv1_grad_mask = np.asarray([1.0] * self.rep_planes_3x3 + [0.0] * self.gen_planes_3x3)
            conv1_grad_mask = conv1_grad_mask.reshape([conv1_grad_mask.shape[0], 1, 1, 1]).astype(np.float32)
            self.conv1_grad_mask = torch.nn.parameter.Parameter(
                torch.from_numpy(conv1_grad_mask), requires_grad=False)
            self.conv1_mask = torch.nn.parameter.Parameter(
                torch.from_numpy(conv1_grad_mask.transpose(1, 0, 2, 3)), requires_grad=False)
            
            conv2_grad_mask = np.asarray([1.0] * self.rep_planes_3x3 + [0.0] * self.gen_planes_3x3)
            conv2_grad_mask = conv2_grad_mask.reshape([conv2_grad_mask.shape[0], 1, 1, 1]).astype(np.float32)
            self.conv2_grad_mask = torch.nn.parameter.Parameter(
                torch.from_numpy(conv2_grad_mask), requires_grad=False)
            self.conv2_mask = torch.nn.parameter.Parameter(
                torch.from_numpy(conv2_grad_mask.transpose(1, 0, 2, 3)), requires_grad=False)

            conv3_grad_mask = np.asarray([1.0] * self.rep_planes_3x3 + [0.0] * self.gen_planes_3x3)
            conv3_grad_mask = conv3_grad_mask.reshape([1, conv3_grad_mask.shape[0], 1, 1]).astype(np.float32)
            self.conv3_grad_mask = torch.nn.parameter.Parameter(
                torch.from_numpy(conv3_grad_mask), requires_grad=False)
            torch.nn.init.zeros_(self.conv3.bias)
        else:
            torch.nn.init.kaiming_normal_(self.conv1.weight, a=math.sqrt(5.0))
            torch.nn.init.kaiming_normal_(self.conv2.weight, a=math.sqrt(5.0))
            torch.nn.init.kaiming_normal_(self.conv3.weight, a=math.sqrt(5.0))

    def cuda(self, *args, **kwargs):
        self = super().cuda(*args, **kwargs)


class ConvCoder(nn.Module):
    def __init__(
            self,
            num_blocks: List[int],
            stem_strides: int = 1,
            stem_planes: int = 64,
            strides: List[int] = [1, 2, 2],
            planes: List[int] = [16, 32, 64],
            channel_multiplier: float = 2.0,
            num_classes: int = 10,
            dropout: float = 0.0,
            drop_path: float = 0.0,
            mode: str = 'rep',
            use_bn: bool = True,
            use_finegrained_head: bool = False,
            finegrained_head_dropout: float = 0.0,
            head_planes: int = 1280,
            bot_ratio: float = 1. - 8. / 16.,
            **kwargs,
    ):
        super().__init__()
        self.strides = strides
        self.num_blocks = num_blocks
        self.stem_strides = stem_strides
        assert len(self.strides) == len(self.num_blocks)
        self.stem_planes = stem_planes
        self.in_planes = stem_planes
        self.planes = planes
        self.channel_multiplier = channel_multiplier
        self.dropout = dropout
        self.drop_path = drop_path
        self.mode = mode
        self.use_bn = use_bn
        self.bot_ratio = bot_ratio
        self.num_classes = num_classes
        # Stem Layer configuration.
        assert stem_strides == 2, "Stem strides must be 2 for ImageNet!"
        self.conv1 = nn.Conv2d(
            3, self.stem_planes, kernel_size=7, stride=self.stem_strides, padding=3, bias=not self.use_bn)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(self.stem_planes, eps=1e-5, momentum=0.1)
        else:
            self.bn1 = None
        self.relu1 = nn.ReLU(True)

        self.ds1 = nn.MaxPool2d(3, 2, padding=1)

        self.layers = nn.ModuleList([])    
        self.in_planes = self.stem_planes
        num_stages = len(strides)
        for idx in range(num_stages):
            layer = self._make_layer(self.planes[idx], num_blocks[idx], strides[idx], bot_ratio=self.bot_ratio)
            self.layers.append(layer)
        # Initialize heads.
        self.use_finegrained_head = use_finegrained_head
        self.head_planes = head_planes
        self.finegrained_head_dropout = finegrained_head_dropout
        if self.use_finegrained_head:
            self.finegrained_head = nn.Conv2d(
                self.planes[-1], self.head_planes, 1, 1, 0, bias=not self.use_bn)
            self.head_bn = nn.BatchNorm2d(self.head_planes, eps=1e-5, momentum=0.1) if self.use_bn else None
            self.head_dropout = nn.Dropout(
                self.finegrained_head_dropout) if self.finegrained_head_dropout != 0 else None
            self.head_relu = nn.ReLU(True)
            out_planes = self.head_planes
        else:
            self.finegrained_head, self.head_bn, self.head_relu = None, None, None
            out_planes = self.planes[-1]

        self.linear = nn.Linear(out_planes, num_classes)
        # Intialize global steps.
        self.global_steps = 0
        # Initialize weights if needed.
        self.init_weights()

    @staticmethod
    def gen_planes(num_blocks, candidate_planes):
        planes = []
        for idx in range(num_blocks):
            planes.append(candidate_planes[(idx+len(candidate_planes)-1) % len(candidate_planes)])
        return planes

    def init_weights(self):
        # Stem Conv.
        print("Init weights from priors...")
        torch.nn.init.kaiming_normal_(self.conv1.weight, a=math.sqrt(5.0))
        # Body.
        def init_weights_per_layer(layer):
            for block in layer:
                block.init_weights()
        for layer in self.layers:
            init_weights_per_layer(layer)
        if self.use_finegrained_head:
            torch.nn.init.kaiming_normal_(self.finegrained_head.weight, a=math.sqrt(5.0))
            if not self.use_bn:
                torch.nn.init.zeros_(self.finegrained_head.bias)
        # Head Linear.
        torch.nn.init.normal_(self.linear.weight, 0.0, 0.01)
        torch.nn.init.zeros_(self.linear.bias)

    def _make_layer(self, planes, num_blocks, stride, bot_ratio: float = 4. / 8.):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for idx, s in enumerate(strides):
            layers.append(ConvCoderBottleneck(
                self.in_planes,
                planes,
                s,
                dropout=self.dropout if idx != 0 else 0.0,
                drop_path=self.drop_path,
                bot_ratio=bot_ratio,
                mode=self.mode,
                use_bn=self.use_bn,
                use_gen=(idx != 0)))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            self.global_steps += 1
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu1(out)
        # Keep original Max Pool.
        out = self.ds1(out)
        # Body
        for layer in self.layers:
            out = layer(out)
        if self.use_finegrained_head:
            out = self.finegrained_head(out)
            if self.use_bn:
                out = self.head_bn(out)
            out = self.head_relu(out)
            if self.finegrained_head_dropout != 0.0:
                out = self.head_dropout(out)
        out = out.mean(3).mean(2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def mask_arch_grads(self):
        def _mask_arch_grads_per_layer(layer):
            # Apply gradient on the architecture part.
            for idx in range(len(layer)):
                # Disable learning from data.
                if layer[idx].use_gen:
                    if self.mode == 'rep':
                        layer[idx].conv1.weight.grad.data = layer[idx].conv1.weight.grad.data * layer[idx].conv1_grad_mask
                        layer[idx].conv2.weight.grad.data = layer[idx].conv2.weight.grad.data * layer[idx].conv2_grad_mask
                        layer[idx].conv3.weight.grad.data = layer[idx].conv3.weight.grad.data * layer[idx].conv3_grad_mask
                    else:
                        raise NotImplementedError("Mode {} not supported!".format(self.mode))
        for layer in self.layers:
            _mask_arch_grads_per_layer(layer)

    def cuda(self, *args, **kwargs):
        for _, m in self.named_modules():
            if m != self:
                m = m.cuda(*args, **kwargs)
        return self

    def configure_drop_path(self, drop_path_rate):
        def _configure_drop_path_per_layer(layer, dp_rate):
            for idx in range(len(layer)):
                layer[idx].drop_path = dp_rate if idx != 0 else 0.0
        self.drop_path = drop_path_rate
        # Configure all sub-layers.
        for layer in self.layers:
            _configure_drop_path_per_layer(layer, drop_path_rate)


def _create_convcoder(variant, pretrained=False, **kwargs):
    assert not pretrained, NotImplementedError("Pretrained model not Available!") 
    return build_model_with_cfg(ConvCoder, variant, pretrained, **kwargs)


@register_model
def convcoder50(pretrained=False, **kwargs):
    """Constructs a ConvCoder-50 model.
    """
    model_args = dict(stem_strides=2, strides=[1, 2, 2, 2],
            num_blocks=[3, 4, 6, 3], planes=[256, 512, 1024, 2048], 
            stem_planes=64, num_classes=1000)
    return _create_convcoder("convcoder50", pretrained, **dict(model_args, **kwargs))


@register_model
def convcoder101(pretrained=False, **kwargs):
    """Constructs a ConvCoder-101 model.
    """
    model_args = dict(stem_strides=2, strides=[1, 2, 2, 2],
            num_blocks=[3, 4, 23, 3], planes=[256, 512, 1024, 2048], 
            stem_planes=64, num_classes=1000)
    return _create_convcoder("convcoder101", pretrained, **dict(model_args, **kwargs))



@register_model
def convcoder152(pretrained=False, **kwargs):
    """Constructs a ConvCoder-152 model.
    """
    model_args = dict(stem_strides=2, strides=[1, 2, 2, 2],
            num_blocks=[3, 8, 36, 3], planes=[256, 512, 1024, 2048], 
            stem_planes=64, num_classes=1000)
    return _create_convcoder("convcoder152", pretrained, **dict(model_args, **kwargs))
