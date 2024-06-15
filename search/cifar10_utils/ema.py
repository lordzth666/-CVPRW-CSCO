from typing import List

import torch
from torch import nn

from copy import deepcopy
from collections import OrderedDict
from sys import stderr

# for type hint
from torch import Tensor


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.training:
            return self.model(inputs)
        else:
            return self.shadow(inputs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if self.training:
            return self.model.state_dict()
        else:
            print("Saving state_dict for shadow weights...")
            return self.shadow.state_dict()
        
    def set_drop_connect_rate(self, drop_connect_rate: float = 0.0):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.set_drop_connect_rate(drop_connect_rate)
        else:
            self.model.set_drop_connect_rate(drop_connect_rate)
            
    def forward_with_names(self, x: torch.Tensor, list_of_names: List[str]):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.forward_with_names(x, list_of_names)
        else:
            self.model.forward_with_names(x, list_of_names)