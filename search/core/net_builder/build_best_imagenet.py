import os, sys
from tokenize import maybe
cwd = os.getcwd()
sys.path.append(cwd)

import pickle
import argparse
import time
import copy
import numpy as np

from src.metagraph_v3.metagraph import MetaGraph_v3
from src.proto.writer import ProtoWriter
from src.proto.contrib import *
from src.estimator.imagenet_estimator import ImageNetEstimator
from src.estimator.cifar10_estimator import CIFAR10Estimator
from src.yaml_utils.yaml_parser import load_and_apply_yaml_config  # noqa: E402
from nasflow.io_utils.base_io import maybe_load_json_file


ps_settings_imagenet_monotone = {
    'strides': [2, 2, 2, 1, 2, 1], 
    'num_channels': [24, 32, 64, 96, 160, 320],
    'num_repeats': [2, 3, 4, 3, 3, 1],
    'num_stem_conv': 16, 'stem_activation': "relu",
    'num_head_conv': 1280, "head_activation": "relu", 
    'num_stages': 6, 'num_nodes': 13,
}

# 1, [1-3], [2,4], [3-5], [1-3], [2,4], [1-2]
def make_divisible(value, divided_by=8):
    return int(round(value / divided_by) * divided_by)

def scale_up_ps(ps, width_multiplier):
    new_ps = copy.deepcopy(ps)
    for idx in range(len(new_ps['num_channels'])):
        new_ps['num_channels'][idx] = make_divisible(new_ps['num_channels'][idx] * width_multiplier)
    new_ps['num_stem_conv'] = make_divisible(new_ps['num_stem_conv'] * width_multiplier)
    return new_ps

def main(args):
    config = load_and_apply_yaml_config(args.yaml_cfg)
    if not os.path.exists(args.proto_dir):
        os.makedirs(args.proto_dir)
    ps = ps_settings_imagenet_monotone
    ps = scale_up_ps(ps, args.width_multiplier)
    if os.path.exists(args.metagraph_path):
        print("Loaded meta-graphs!")
        with open(args.metagraph_path, 'rb') as fp:
            meta_graph = pickle.load(fp)
        assert isinstance(meta_graph, MetaGraph_v3)
    else:
        raise NotImplementedError("Metagraph path must be specified!")
    
    best_json_file = maybe_load_json_file(args.model_json_config)
    best_hash = best_json_file["hash"]
    best_mask = np.asarray(list(best_hash), dtype=np.int32).reshape([ps['num_stages'], ps['num_nodes'], ps['num_nodes']])
    meta_graph.mask = copy.deepcopy(best_mask)

    writer = ProtoWriter("hybnas-imagenet-wm{}-se{}".format(args.width_multiplier, int(args.use_se)))
    writer.add_header("imagenet")
    writer.add(Convolutional_Proto(name="conv_stem",
                                   input="input",
                                   kernel_size=3,
                                   strides=2,
                                   filters=ps['num_stem_conv'],
                                   activation=ps['stem_activation'],
                                   regularizer_strength=1e-5,
                                   batchnorm=True,
                                   use_bias=False,
                                   trainable=True,
                                   use_dense_initializer=False))
    meta_graph.ofstream = "conv_stem"
    # Add 1 depthwise layer for 1st stride=2 block on ImageNet.
    writer.add(DepthwiseConv_Proto(name="conv_stem_dw3x3",
                                   input="conv_stem",
                                   kernel_size=3,
                                   strides=1,
                                   depthwise_multiplier=1,
                                   activation=ps['stem_activation'],
                                   regularizer_strength=1e-5,
                                   batchnorm=True,
                                   use_bias=False,
                                   trainable=True,))
    meta_graph.ofstream = "conv_stem_dw3x3"

    # Add 1 depthwise layer for 1st stride=2 block on ImageNet.
    writer.add(Convolutional_Proto(name="conv_stem_1x1",
                                   input="conv_stem_dw3x3",
                                   kernel_size=1,
                                   strides=1,
                                   filters=ps['num_stem_conv'],
                                   activation="linear",
                                   regularizer_strength=1e-5,
                                   batchnorm=True,
                                   use_bias=False,
                                   trainable=True,))

    meta_graph.ofstream = "conv_stem_1x1"
    meta_graph.add_to_proto(writer,
                            num_repeats=ps['num_repeats'],
                            strides=ps['strides'],
                            num_channels=ps['num_channels'],
                            use_bias=False,
                            use_residual=True,
                            activation="relu",
                            proj_dim_per_cell=True,
                            pool_style="irb",
                            use_se=args.use_se
                        )
    writer.in_name = meta_graph.ofstream
    if ps['num_head_conv'] != 0:
        writer.add(Convolutional_Proto(name='conv1x1',
                                       input=writer.in_name,
                                       kernel_size=1,
                                       filters=ps['num_head_conv'],
                                       strides=1,
                                       activation=ps['head_activation'],
                                       batchnorm=True,
                                       regularizer_strength=1e-5,
                                       use_bias=False,
                                       trainable=True,
                                       use_dense_initializer=False,))
        out_name = "conv1x1"
    else:
        out_name = writer.in_name
    # Add the average pool layer.
    writer.add(GlobalAvgPool_proto(name="avg_1k",
                                   input=out_name))

    out_name = "avg_1k"

    # Add Dropout
    writer.add(Dropout_proto(name="fc_dropout",
                             input=out_name,
                             dropout=args.dropout))

    writer.finalized("imagenet", out_name=out_name)
    writer.set_global_regularization(config.weight_decay)

    prototxt = os.path.join(args.proto_dir, "hybnas-imagenet-wm{}-se{}.prototxt".format(args.width_multiplier, int(args.use_se)))
    with open(prototxt, 'w') as fp:
        writer.dump(fp)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metagraph_path", type=str, default=None,
                        help="Metagraph path.")
    parser.add_argument("--width_multiplier", type=float, default=1.0,
                        help="Width multiplier.")
    parser.add_argument("--proto_dir", type=str, default=None,
                        help='Root directory to store best prototxts.')
    parser.add_argument("--yaml_cfg", type=str ,default=None,
                        help="Search config.")
    parser.add_argument("--model_json_config", type=str, default=None,
                        help="Best model json config.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout ratio.")
    parser.add_argument("--use_se", default=False,
                        action="store_true")
    args = parser.parse_args()
    main(args)
