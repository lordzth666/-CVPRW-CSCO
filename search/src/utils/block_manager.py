from src.utils.basic_layers import layer_mapper, get_kwarg_with_default

def channel_block_manager(
        op_type, 
        block_proto, 
        in_channels, 
        out_channels=-1,
        inferred_strides=1, 
        bn_momentum: float = 0.1,
        bn_epsilon: float = 1e-5, 
        drop_connect_rate: float = 0.0,
        dropout_rate: float = 0.0,
        se_intra_activation_fn: str = "relu",
        se_output_activation_fn: str = "sigmoid",
    ):
    """
    A block manager which creates an op according to the input prototxts.
    :param op_type:
    :param block_proto:
    :param in_channels:
    :return:
    """
    op = layer_mapper[op_type](
        kernel_size=get_kwarg_with_default(block_proto, 'kernel_size', 3),
        in_channels=in_channels,
        in_features=in_channels,
        # Generic settings
        bn_momentum=bn_momentum,
        bn_epsilon=bn_epsilon,
        drop_connect_rate=drop_connect_rate,
        dropout_rate=dropout_rate,
        se_intra_activation_fn=se_intra_activation_fn,
        se_output_activation_fn=se_output_activation_fn,
        # CONV
        strides=get_kwarg_with_default(block_proto, 'strides', 1),
        padding=get_kwarg_with_default(block_proto, 'padding', 'SAME'),
        activation=get_kwarg_with_default(block_proto, 'activation', 'relu'),
        filters=get_kwarg_with_default(block_proto, 'filters', 32),
        dropout=get_kwarg_with_default(block_proto, 'dropout', 0.0),
        batchnorm=get_kwarg_with_default(block_proto, 'batchnorm', True),
        regularizer_strength=get_kwarg_with_default(block_proto, 'regularizer_strength', 1e-05),
        use_bias=get_kwarg_with_default(block_proto, 'use_bias', False),
        pool_size=get_kwarg_with_default(block_proto, 'pool_size', 2),
        trainable=get_kwarg_with_default(block_proto, 'trainable', True),
        units=get_kwarg_with_default(block_proto, 'units', 10),
        # Add
        op1_channel=in_channels,
        op2_channel=out_channels,
        inferred_strides=inferred_strides,
        # Depthwise convolution
        depth_multiplier=get_kwarg_with_default(block_proto, 'depth_multiplier', -1),
        depthwise_multiplier=get_kwarg_with_default(block_proto, 'depthwise_multiplier', -1),
        # SE
        se_ratio=get_kwarg_with_default(block_proto, 'se_ratio', -1),
        # MISC
        use_dense_initializer=get_kwarg_with_default(block_proto, 'use_dense_initializer', False)
    )
    return op

def get_output_channels(op_type, input_names, 
                        input_channels_list, output_channels,
                        depth_multiplier=1, depthwise_multiplier=1):
    # print(op_type)
    if op_type == 'Identity' or op_type == 'GlobalAvgPool' or op_type == 'Flatten' or op_type == 'MaxPool' or \
            op_type == "Dropout" or op_type == 'SE':
        return input_channels_list[input_names]
    if op_type == 'Add':
        # For addition, all types of input tensors has the same output dim.
        return input_channels_list[input_names[1]]
    elif op_type == 'Concat':
        sum_channels = 0
        for name in input_names:
            sum_channels += input_channels_list[name]
        return sum_channels
    elif op_type == 'DepthwiseConv':
        return input_channels_list[input_names] * depthwise_multiplier
    else:
        # For other operations, return the output channels
        return output_channels
