from multiprocessing import pool
import numpy as np
from sympy import Not

from src.proto.layer import *

def add_to_name_scope(
        name,
        scope=None):
    """
    Add name to scope
    :param name:
    :param scope:
    :return:
    """
    if scope is None:
        return name
    else:
        return scope + "/" + name


def make_divisible(filters, divisible_by=4):
    filters = max(round(filters / divisible_by), 1) * divisible_by
    return filters


def get_leaf_nodes(mask):
    """
    Collect the leaf nodes in the mask. Remove activation function for these layers.
    :param mask:
    :return:
    """
    nodes = []
    n = np.shape(mask)[0]
    for i in range(n):
        has_connection = False
        for j in range(n):
            if mask[i][j]:
                has_connection = True
                break
        if not has_connection:
            nodes.append(i)
    return nodes


def convert_mask_to_proto(mask,
                          ops_def,
                          input,
                          num_channels,
                          scope=None,
                          strides=1,
                          activation="relu",
                          use_bias=False,
                          batchnorm=True,
                          use_residual=False,
                          bottleneck_factor=4,
                          use_se=False,
                          node_coefficient=None,
                          proj_dim_per_cell: bool = False,
                          proj_dim_multiplier: float = 1.0,
                          pool_style: str = "max-pool"):

    nodes = np.shape(mask)[0]
    if node_coefficient is None:
        node_coefficient = np.ones([nodes])

    leafs = get_leaf_nodes(mask)

    out_filters = int(num_channels)
    out_filters = make_divisible(out_filters)
    if not use_residual:
        base_filters = out_filters
    else:
        base_filters = out_filters // bottleneck_factor

    identity_name = add_to_name_scope("node_0", scope)
    # first op is always identity
    if strides == 1:
        ret = Identity_proto(name=identity_name,
                   input=input)
    else:
        if pool_style == "max-pool":                
            ret = MaxPool_proto(
                name=identity_name,
                input=input,
                strides=strides,
                pool_size=strides+1,
            )
        if pool_style == "irb":                
            ret = Convolutional_Proto(
                name=f"{identity_name}/irb-expand-conv",
                input=input,
                filters=int(base_filters * 4),  # Fixed
                kernel_size=1,
                strides=1,
                padding='SAME',
                activation=activation,
                batchnorm=batchnorm,
                use_bias=use_bias,
            )
            ret += DepthwiseConv_Proto(
                name=f"{identity_name}/irb-dw-conv",
                input=f"{identity_name}/irb-expand-conv",
                kernel_size=3,
                strides=strides,
                activation=activation,
                padding='SAME',
                batchnorm=batchnorm,
                use_bias=use_bias,
            )
            ret += Convolutional_Proto(
                name=identity_name,
                input=f"{identity_name}/irb-dw-conv",
                filters=base_filters,    # Fixed
                kernel_size=1,
                strides=1,
                padding='SAME',
                activation="linear",
                batchnorm=batchnorm,
                use_bias=use_bias,
            )
            return ret, identity_name
        elif pool_style == "depthwise-conv":
            ret = DepthwiseConv_Proto(
                name=identity_name,
                input=input,
                kernel_size=strides+1,
                strides=strides,
                activation=activation,
                padding='SAME',
                batchnorm=batchnorm,
                use_bias=use_bias,
            )
        else:
            raise NotImplementedError("downsample block {} not found!")
    concat_cnt = 0
    nodes_queue = [0]
    for i in range(nodes):
        if i in leafs and use_residual and not proj_dim_per_cell:
            activation_ = 'linear'
        else:
            activation_ = activation
        ltype = ops_def[i]['type']
        connected = []
        for j in range(i):
            if not j in nodes_queue:
                continue
            if mask[j][i] == 1:
                node_name = add_to_name_scope("node_%d" %j, scope)
                connected.append(node_name)
                nodes_queue.append(i)
        if len(connected) == 0:
            continue
        # maybe concat or identity
        if len(connected) > 1:
            concat_name = add_to_name_scope("concat_%d" %concat_cnt, scope)
            ret += Concat_proto(name=concat_name,
                             input=connected)
            ifstream = concat_name
            concat_cnt += 1
        else:
            ifstream = connected[0]

        ofstream = add_to_name_scope("node_%d" %i, scope=scope)
        if ltype == "Convolutional":
            # enable stride for only the first op
            ret += Convolutional_Proto(name=ofstream,
                                       input=ifstream,
                                       filters=int(base_filters * node_coefficient[i]),
                                       kernel_size=ops_def[i]['kernel_size'],
                                       strides=1,
                                       padding='SAME',
                                       activation=activation_,
                                       batchnorm=batchnorm,
                                       use_bias=use_bias)
        elif ltype == "SeparableConv":
            ret += SeparableConv_Proto(name=ofstream,
                                       input=ifstream,
                                       filters=int(base_filters * node_coefficient[i]),
                                       kernel_size=ops_def[i]['kernel_size'],
                                       depth_multiplier=node_coefficient[i],
                                       strides=1,
                                       padding='SAME',
                                       activation=activation_,
                                       batchnorm=batchnorm,
                                       use_bias=use_bias)
        elif ltype == "DepthwiseConv":
            if node_coefficient[i] != 1:
                print("INFO: Node coefficient is applied to depth multiplier.")
            ret += DepthwiseConv_Proto(name=ofstream,
                                       input=ifstream,
                                       kernel_size=ops_def[i]['kernel_size'],
                                       strides=1,
                                       depthwise_multiplier=node_coefficient[i],
                                       padding='SAME',
                                       activation=activation_,
                                       batchnorm=batchnorm,
                                       use_bias=use_bias)
        elif ltype == "MaxPool":
            ret += MaxPool_proto(name=ofstream,
                                 input=ifstream,
                                 pool_size=ops_def[i]['pool_size'],
                                 strides=ops_def[i]['strides'])
        else:
            print(ltype)
            raise NotImplementedError

    # Inspect leaf nodes and concat
    out_degree = np.zeros(nodes)
    in_degree = np.zeros(nodes)
    in_degree[0] = 1
    for i in range(nodes):
        for j in range(i+1, nodes):
            if mask[i][j] == 1 and i in nodes_queue and j in nodes_queue:
                out_degree[i] += 1
                in_degree[j] += 1

    # Create concat_pool list
    concat_pool_names = []
    for i in range(nodes):
        if out_degree[i] == 0 and in_degree[i] != 0:
            concat_pool_names.append(add_to_name_scope("node_%d" %i, scope))
    # Write Concat Pool list
    concat_name = add_to_name_scope("concat_pool", scope)
    if len(concat_pool_names) != 1:
        ret += Concat_proto(name=concat_name,
                     input=concat_pool_names)
    else:
        ret += Identity_proto(name=concat_name,
                              input=concat_pool_names[0])

    # SE first on rich features.
    if use_se:
        se_name = add_to_name_scope("se", scope)
        ret += SE_Proto(name=se_name,
                        input=concat_name,
                        se_ratio=0.25)
        concat_name = se_name

    # Add a final projection layer if needed.
    if proj_dim_per_cell:
        concat_name_proj = add_to_name_scope("concat_pool_proj", scope)
        ret += Convolutional_Proto(name=concat_name_proj,
                                   input=concat_name,
                                   filters=base_filters * proj_dim_multiplier,
                                   kernel_size=1,
                                   strides=1,
                                   padding='SAME',
                                   activation="linear",
                                   batchnorm=batchnorm,
                                   use_bias=use_bias)
        concat_name = concat_name_proj

    if use_residual:
        out_name = concat_name
        # Then add residual connections together, id mapping is fused into add.
        residual_output_name = add_to_name_scope("residual_out", scope)
        residual_activation = "linear"
        ret += Add_proto(residual_output_name,
                         input=[identity_name, out_name],
                         activation=residual_activation)
        output_node_name = residual_output_name
    else:
        output_node_name = concat_name

    return ret, output_node_name

def inspect_num_edges(mask):
    nodes_queue = [0]
    nodes = np.shape(mask)[0]
    edges = 0
    for i in range(nodes):
        if not i in nodes_queue:
            continue
        for j in range(i+1, nodes):
            if mask[i][j] == 1:
                nodes_queue.append(j)
                edges += 1
    return edges
