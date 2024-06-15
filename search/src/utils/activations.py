import torch.nn as nn
import torch.nn.functional as F


function_wrapper = {
    "relu": lambda x: F.relu(x, inplace=False),
    "leaky": lambda x: F.leaky_relu(x, inplace=False),
    "linear": lambda x: x,
    "sigmoid": lambda x: F.sigmoid(x),
    "hard_swish": lambda x: F.hardswish(x, inplace=False),
    "hswish": lambda x: F.hardswish(x, inplace=False),
    "swish": lambda x: F.silu(x, inplace=False),
    "hard_sigmoid": lambda x: F.hardsigmoid((x, True)),
}

def apply_activation(x, name):
    """
    Apply activation given an input tensor.
	:param x: input tensor.
	:param name: activation name. Must be one of 'relu', 'leaky' and 'linear'.
	:return:
	"""
    func = function_wrapper[name]
    return func(x)
