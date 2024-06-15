import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import apply_activation
from .drop_connect import drop_connect_impl

def get_kwarg_with_default(kwargs, key, default_value):
    try:
        value = kwargs[key]
        if isinstance(default_value, bool):
            if value == "True" or value == True:
                value = True
            else:
                value = False
    except Exception:
        value = default_value
    return value

class Convolution(nn.Module):
    def __init__(self, **kwargs):
        super(Convolution, self).__init__()
        self.kernel_size = get_kwarg_with_default(kwargs, 'kernel_size', 3)
        self.strides = get_kwarg_with_default(kwargs, 'strides', 1)
        self.filters = get_kwarg_with_default(kwargs, 'filters', 32)
        self.regularizer = get_kwarg_with_default(kwargs, 'regularizer', 1e-5)
        self.batchnorm = get_kwarg_with_default(kwargs, 'batchnorm', True)
        self.use_bias = get_kwarg_with_default(kwargs, 'use_bias', False)
        self.trainable = get_kwarg_with_default(kwargs, 'trainable', True)
        self.padding = get_kwarg_with_default(kwargs, 'padding', 'SAME')
        self.activation = get_kwarg_with_default(kwargs, 'activation', 'relu')
        self.bn_momentum = get_kwarg_with_default(kwargs, 'bn_momentum', 0.1)
        self.bn_epsilon = get_kwarg_with_default(kwargs, 'bn_epsilon', 1e-5)

        # Additional arguments has to be input.
        self.in_channels = get_kwarg_with_default(kwargs, 'in_channels', -1)

        # Define the padding way
        if self.padding == 'SAME':
            padded_value = self.kernel_size // 2
        else:
            padded_value = 0

        self.conv = nn.Conv2d(
            kernel_size=self.kernel_size,
            stride=self.strides,
            in_channels=self.in_channels,
            out_channels=self.filters,
            padding=padded_value,
            bias=self.use_bias,
            dilation=1,
            groups=1
        )
        self.conv.dense_initializer = get_kwarg_with_default(
            kwargs, 'use_dense_initializer', False)
        # define the batchnorm way.
        if self.batchnorm:
            self.batchnorm_layer = nn.BatchNorm2d(
                eps=self.bn_epsilon, 
                momentum=self.bn_momentum,
                num_features=self.filters,
            )
        else:
            self.batchnorm_layer = None


    def forward(self, x):
        out = self.conv(x)
        if self.batchnorm:
            out = self.batchnorm_layer(out)
        out = apply_activation(out, self.activation)
        return out

class DepthwiseConvolution(nn.Module):
    
    def __init__(self, **kwargs):
        super(DepthwiseConvolution, self).__init__()
        self.kernel_size = get_kwarg_with_default(kwargs, 'kernel_size', 3)
        self.strides = get_kwarg_with_default(kwargs, 'strides', 1)
        # self.filters = get_kwarg_with_default(kwargs, 'filters', 32)
        self.regularizer = get_kwarg_with_default(kwargs, 'regularizer', 1e-5)
        self.batchnorm = get_kwarg_with_default(kwargs, 'batchnorm', True)
        self.use_bias = get_kwarg_with_default(kwargs, 'use_bias', False)
        self.trainable = get_kwarg_with_default(kwargs, 'trainable', True)
        self.padding = get_kwarg_with_default(kwargs, 'padding', 'SAME')
        self.activation = get_kwarg_with_default(kwargs, 'activation', 'relu')
        self.depthwise_multiplier = get_kwarg_with_default(kwargs, 'depthwise_multiplier', -1)
        self.bn_momentum = get_kwarg_with_default(kwargs, 'bn_momentum', 0.1)
        self.bn_epsilon = get_kwarg_with_default(kwargs, 'bn_epsilon', 1e-5)

        # Additional arguments has to be input.
        self.in_channels = get_kwarg_with_default(kwargs, 'in_channels', -1)

        # Define the padding way
        if self.padding == 'SAME':
            padded_value = self.kernel_size // 2
        else:
            padded_value = 0

        self.filters = self.in_channels * self.depthwise_multiplier
        self.conv = nn.Conv2d(
            kernel_size=self.kernel_size,
            stride=self.strides,
            in_channels=self.in_channels,
            out_channels=self.filters,
            padding=padded_value,
            bias=self.use_bias,
            dilation=1,
            groups=self.in_channels
        )
        # define the batchnorm way.
        if self.batchnorm:
            self.batchnorm_layer = nn.BatchNorm2d(
                num_features=self.filters, 
                eps=self.bn_epsilon,
                momentum=self.bn_momentum
            )
        else:
            self.batchnorm_layer = None

    def forward(self, x):
        out = self.conv(x)
        if self.batchnorm:
            out = self.batchnorm_layer(out)
        out = apply_activation(out, self.activation)
        return out

class SeparableConvolution(nn.Module):
    def __init__(self, **kwargs):
        super(SeparableConvolution, self).__init__()
        self.kernel_size = get_kwarg_with_default(kwargs, 'kernel_size', 3)
        self.strides = get_kwarg_with_default(kwargs, 'strides', 1)
        self.filters = get_kwarg_with_default(kwargs, 'filters', 32)
        self.regularizer = get_kwarg_with_default(kwargs, 'regularizer', 1e-5)
        self.batchnorm = get_kwarg_with_default(kwargs, 'batchnorm', True)
        self.use_bias = get_kwarg_with_default(kwargs, 'use_bias', False)
        self.trainable = get_kwarg_with_default(kwargs, 'trainable', True)
        self.padding = get_kwarg_with_default(kwargs, 'padding', 'SAME')
        self.activation = get_kwarg_with_default(kwargs, 'activation', 'relu')
        self.depth_multiplier = get_kwarg_with_default(kwargs, 'depth_multiplier', -1)
        assert self.depth_multiplier != -1, ValueError("Missing depth_multiplier in layer %s" %(self.name))

        # Additional arguments has to be input.
        self.in_channels = get_kwarg_with_default(kwargs, 'in_channels', -1)

        self.bn_momentum = get_kwarg_with_default(kwargs, 'bn_momentum', 0.1)
        self.bn_epsilon = get_kwarg_with_default(kwargs, 'bn_epsilon', 1e-5)

        # Define the padding way
        if self.padding == 'SAME':
            padded_value = self.kernel_size // 2
        else:
            padded_value = 0

        self._dw_filters = self.depth_multiplier * self.in_channels

        self.depthwise_conv = nn.Conv2d(
            kernel_size=self.kernel_size,
            stride=self.strides,
            in_channels=self.in_channels,
            out_channels=self._dw_filters,
            padding=padded_value,
            bias=True,
            dilation=1,
            groups=self.in_channels
        )
        self.pointwise_conv = nn.Conv2d(
            kernel_size=1,
            stride=1,
            in_channels=self._dw_filters,
            out_channels=self.filters,
            padding=0,
            bias=self.use_bias,
            dilation=1
        )
        # define the batchnorm way.
        if self.batchnorm:
            self.batchnorm_layer = nn.BatchNorm2d(
                num_features=self.filters, 
                eps=self.bn_epsilon,
                momentum=self.bn_momentum
            )
        else:
            self.batchnorm_layer = None


    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        if self.batchnorm:
            out = self.batchnorm_layer(out)
        out = apply_activation(out, self.activation)
        return out

class Dense(nn.Module):
    def __init__(self, **kwargs):
        super(Dense, self).__init__()
        self.units = get_kwarg_with_default(kwargs, 'units', 6)
        self.regularizer = get_kwarg_with_default(kwargs, 'regularizer', 1e-5)
        self.batchnorm = get_kwarg_with_default(kwargs, 'batchnorm', True)
        self.use_bias = get_kwarg_with_default(kwargs, 'use_bias', False)
        self.trainable = get_kwarg_with_default(kwargs, 'trainable', True)
        self.activation = get_kwarg_with_default(kwargs, 'activation', 'relu')

        # Additional arguments has to be input.
        self.in_features = get_kwarg_with_default(kwargs, 'in_features', 1024)
        self.bn_momentum = get_kwarg_with_default(kwargs, 'bn_momentum', 0.1)
        self.bn_epsilon = get_kwarg_with_default(kwargs, 'bn_epsilon', 1e-5)
        
        self.dense = nn.Linear(
            in_features=self.in_features,
            out_features=self.units
        )
        # define the batchnorm way.
        if self.batchnorm:
            self.batchnorm_layer = nn.BatchNorm1d(
                num_features=self.units, 
                momentum=self.bn_momentum,
                eps=self.bn_epsilon
            )
        else:
            self.batchnorm_layer = None


    def forward(self, x):
        out = self.dense(x)
        if self.batchnorm:
            out = self.batchnorm_layer(out)
        out = apply_activation(out, self.activation)
        return out

class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Concat(nn.Module):
    def __init__(self, **kwargs):
        super(Concat, self).__init__()

    def forward(self, x):
        return torch.cat(x, dim=1)

class Add(nn.Module):
    def __init__(self, **kwargs):
        super(Add, self).__init__()
        # Create an identity mapping if needed.
        self.op1_channel = get_kwarg_with_default(kwargs, 'op1_channel', -1)
        self.op2_channel = get_kwarg_with_default(kwargs, 'op2_channel', -1)
        self.strides = get_kwarg_with_default(kwargs, 'inferred_strides', 1)
        self.activation = get_kwarg_with_default(kwargs, 'activation', 'linear')
        self.bn_momentum = get_kwarg_with_default(kwargs, 'bn_momentum', 0.1)
        self.bn_epsilon = get_kwarg_with_default(kwargs, 'bn_epsilon', 1e-5)
        self.drop_connect_rate = get_kwarg_with_default(kwargs, 'drop_connect_rate', 1e-5)
        # Use the batchnorm in identity mapping by default.
        self.batchnorm = True
        if self.op1_channel != self.op2_channel or self.strides != 1:
            self.id_conv = nn.Conv2d(
                kernel_size=1,
                in_channels=self.op1_channel,
                out_channels=self.op2_channel,
                stride=self.strides,
                bias=False,
                groups=1,
                dilation=1
            )
            self.batchnorm = nn.BatchNorm2d(
                num_features=self.op2_channel, 
                eps=self.bn_epsilon,
                momentum=self.bn_momentum
            )
        else:
            self.id_conv = None
            self.batchnorm = None

    def forward(self, x):
        # Normal Residual Block.
        if self.id_conv is not None:
            out0 = self.id_conv(x[0])
            out0 = self.batchnorm(out0)
            if self.drop_connect_rate != 0:
                out = drop_connect_impl(
                    x[1], self.drop_connect_rate, self.training) + out0
            else:
                out = x[1] + out0
        else:
            if self.drop_connect_rate != 0:
                out = drop_connect_impl(
                    x[1], self.drop_connect_rate, self.training) + x[0]
            else:
                out = x[1] + x[0]
        return apply_activation(out, self.activation)

    def set_drop_connect_rate(self, drop_connect_rate):
        self.drop_connect_rate = drop_connect_rate

class GlobalAveragePooling(nn.Module):
    def __init__(self, **kwargs):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, 1)

class Flatten(nn.Module):
    def __init__(self, **kwargs):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.squeeze(3).squeeze(2)

class Dropout(nn.Module):
    def __init__(self, **kwargs):
        self.p = get_kwarg_with_default(kwargs, 'dropout', 0.0)
        super(Dropout, self).__init__()
        self._apply_dropout = nn.Dropout(p=self.p)

    def forward(self, x):
        return self._apply_dropout(x)

class MaxPooling(nn.Module):
    def __init__(self, **kwargs):
        super(MaxPooling, self).__init__()
        self.pool_size = get_kwarg_with_default(kwargs, 'pool_size', 2)
        self.strides = get_kwarg_with_default(kwargs, 'strides', 2)
        self.F = nn.MaxPool2d(
            self.pool_size, 
            self.strides, 
            padding=(self.pool_size - self.strides + 1) // 2)

    def forward(self, x):
        return self.F(x)

class SE(nn.Module):
    def __init__(self, **kwargs):
        super(SE, self).__init__()
        self.se_ratio = get_kwarg_with_default(kwargs, 'se_ratio', -1)
        # Additional arguments has to be input.
        self.in_channels = get_kwarg_with_default(kwargs, 'in_channels', -1)
        self.se_intra_activation_fn = get_kwarg_with_default(
            kwargs, "se_intra_activation_fn", "relu")
        self.se_output_activation_fn = get_kwarg_with_default(
            kwargs, "se_output_activation_fn", "relu")
        
        num_squeezed_channels = max(1, int(self.in_channels * self.se_ratio))
        self._se_reduce = nn.Conv2d(in_channels=self.in_channels,
                                    out_channels=num_squeezed_channels,
                                    kernel_size=1)
        self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels,
                                    out_channels=self.in_channels, kernel_size=1)

    def forward(self, x):
        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = apply_activation(
            x_squeezed, self.se_intra_activation_fn)
        x_squeezed = self._se_expand(x_squeezed)
        out = apply_activation(
            x_squeezed, self.se_output_activation_fn) * x
        return out
    

layer_mapper = {
    'Convolutional': Convolution,
    'DepthwiseConv': DepthwiseConvolution,
    'SeparableConv': SeparableConvolution,
    'MaxPool': MaxPooling,
    'GlobalAvgPool': GlobalAveragePooling,
    'Identity': Identity,
    'Add': Add,
    'SE': SE,
    'Concat': Concat,
    'Dense': Dense,
    'Flatten': Flatten,
    'Dropout': Dropout
}