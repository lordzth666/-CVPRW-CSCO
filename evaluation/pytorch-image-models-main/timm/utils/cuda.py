""" CUDA / AMP utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch

try:
    from apex import amp
    has_apex = True
except ImportError:
    amp = None
    has_apex = False

from .clip_grad import dispatch_clip_grad

from timm.models.convcoder import ConvCoder
from timm.models.swin_transformer_coder import SwinTransformerCoder

class ApexScalerConvCoder:
    state_dict_key = "amp"

    def __call__(self, loss, model, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        assert isinstance(model, ConvCoder), TypeError("Instance must be a ConvCoder")
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(create_graph=create_graph)
        if clip_grad is not None:
            dispatch_clip_grad(amp.master_params(optimizer), clip_grad, mode=clip_mode)
        # Important: mask gradients in ConvCoder instance.
        model.mask_arch_grads()        
        optimizer.step()

    def state_dict(self):
        if 'state_dict' in amp.__dict__:
            return amp.state_dict()

    def load_state_dict(self, state_dict):
        if 'load_state_dict' in amp.__dict__:
            amp.load_state_dict(state_dict)


class ApexScaler:
    state_dict_key = "amp"

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(create_graph=create_graph)
        if clip_grad is not None:
            dispatch_clip_grad(amp.master_params(optimizer), clip_grad, mode=clip_mode)
        optimizer.step()

    def state_dict(self):
        if 'state_dict' in amp.__dict__:
            return amp.state_dict()

    def load_state_dict(self, state_dict):
        if 'load_state_dict' in amp.__dict__:
            amp.load_state_dict(state_dict)


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

class NativeScalerConvCoder:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, model, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        assert isinstance(model.module, ConvCoder) or isinstance(model.module, SwinTransformerCoder), TypeError("Instance must be a ConvCoder")
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        # Important: mask gradients in ConvCoder instance.
        else:
            self._scaler.unscale_(optimizer)
        model.module.mask_arch_grads()
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

