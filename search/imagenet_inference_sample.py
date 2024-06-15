import os, sys
cwd = os.getcwd()
sys.path.append(cwd)

import pickle
import argparse
import torch
from src.metagraph_v3.metagraph import MetaGraph_v3
from src.proto.writer import ProtoWriter
from src.proto.contrib import *
from src.estimator.imagenet_estimator import ImageNetEstimator
from src.estimator.cifar10_estimator import CIFAR10Estimator
from src.yaml_utils.yaml_parser import load_and_apply_yaml_config

from src.metagraph_v3.util_graph import random_connected_graph


import numpy as np


ps_settings_imagenet_monotone = {
    'strides': [2, 2, 2, 1, 2, 1], 
    'num_channels': [24, 32, 64, 96, 160, 320],
    'num_repeats': [2, 3, 4, 3, 3, 1],
    'num_stem_conv': 32, 'stem_activation': "relu",
    'num_head_conv': 640, "head_activation": "relu", 
    'num_stages': 6, 'num_nodes': 13,
}
ps_settings_imagenet_monotone = {
    'strides': [2, 2, 2, 1, 2, 1], 
    'num_channels': [24, 32, 64, 96, 160, 320],
    'num_repeats': [2, 3, 4, 3, 3, 1],
    'num_stem_conv': 32, 'stem_activation': "relu",
    'num_head_conv': 640, "head_activation": "relu", 
    'num_stages': 6, 'num_nodes': 13,
}
ps_settings_imagenet_monotone_inference_final = {
    'strides': [2, 2, 2, 1, 2, 1], 
    'num_channels': [24, 32, 64, 96, 160, 320],
    'num_repeats': [2, 3, 4, 3, 3, 1],
    'num_stem_conv': 32, 'stem_activation': "relu",
    'num_head_conv': 1280, "head_activation": "relu", 
    'num_stages': 6, 'num_nodes': 13,
}

# Unfold downsample 
ps_settings_cifar10_monotone = {
    'strides': [1, 2, 1, 2, 1], 'num_channels': [16, 32, 32, 64, 64],
    'num_repeats': [3, 1, 2, 1, 2], 'num_stem_conv': 16, 'stem_activation': "relu",
    'num_head_conv': 0, "head_activation": "relu", 'num_stages': 5, 'num_nodes': 13
}

def main(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    config = load_and_apply_yaml_config(args.yaml_cfg)
    os.makedirs(args.model_dir, exist_ok=True)
    if args.task in ["cifar10"]:
        ps = ps_settings_cifar10_monotone
    elif args.task in ["imagenet"]:
        ps = ps_settings_imagenet_monotone
    else:
        raise NotImplementedError

    # Use to train a FLOPS predictor.
    ps = ps_settings_imagenet_monotone_inference_final
    config.image_size = 224
    ps['num_head_conv'] = 1280

    if os.path.exists(os.path.join(args.model_dir, "hybnas-{}.metagraph".format(args.task))):
        print("Loaded meta-graphs!")
        with open(os.path.join(args.model_dir, "hybnas-{}.metagraph".format(args.task)), 'rb') as fp:
            meta_graph = pickle.load(fp)
        assert isinstance(meta_graph, MetaGraph_v3)
    else:
        meta_graph = MetaGraph_v3(nodes=ps['num_nodes'],
                                  depth=ps['num_stages'],
                                  protocol=args.protocol
                                )
        # Save graph first.
        with open(os.path.join(args.model_dir, "hybnas-{}.metagraph".format(args.task)), 'wb') as fp:
            pickle.dump(meta_graph, fp)

    cfg_path = os.path.join(args.model_dir, "search.config")
    with open(cfg_path, 'w') as fp:
        fp.write("%s" %(vars(args)))

    sharded_record_dir = os.path.join(args.model_dir, "shard-{}".format(args.worker_id))
    os.makedirs(sharded_record_dir, exist_ok=True)

    record_path = os.path.join(sharded_record_dir, "{}.records".format(args.task))
    try:
        with open(record_path, 'rb') as fp:
            records = pickle.load(fp)
        print("Loaded {} records!".format(len(records)))
    except Exception:
        print("Searching from scratch!")
        records = []

    proto_dir = "temp/temp_search_{}_profile_shard_{}".format(args.task, args.worker_id)
    if not os.path.exists(proto_dir):
        os.makedirs(proto_dir)

    if args.task in ["imagenet"]:
        estimator = ImageNetEstimator()
    elif args.task in ["cifar10"]:
        estimator = CIFAR10Estimator()
    else:
        raise NotImplementedError

    i = 0
    while i < args.budget:
        meta_graph.random_sampling(random_connected_graph,
            max_edges=args.max_num_edges)
        mask_list = meta_graph.mask.astype(np.int64).flatten().tolist()
        mask_list = [str(x) for x in mask_list]
        mask_hash = "".join(mask_list)
        # Refresh all records present and query the mask hash.
        all_records = os.listdir(args.model_dir)
        flag = False
        if not args.profile_only:
            for file_dir in all_records:
                if file_dir.startswith("shard"):
                    record_path_tmp = os.path.join(args.model_dir, file_dir, "{}.records".format(args.task))
                    if os.path.exists(record_path_tmp):
                        with open(record_path_tmp, 'rb') as fp:
                            tmp_record = pickle.load(fp)
                            all_keys = [x['hash'] for x in tmp_record]
                            if mask_hash in all_keys:
                                flag = True
                                break
            if flag:
                continue

        writer = ProtoWriter(os.path.join(proto_dir, "{}-candidates-{}.prototxt".format(args.task, i)))
        writer.add_header(args.task)
        writer.add(Convolutional_Proto(name="conv_stem",
                                      input="input",
                                      kernel_size=3,
                                      strides=2 if args.task == "imagenet" else 1,
                                      filters=ps['num_stem_conv'],
                                      activation=ps['stem_activation'],
                                      regularizer_strength=1e-5,
                                      batchnorm=True,
                                      use_bias=False,
                                      trainable=True,
                                      use_dense_initializer=False)
                                )
        meta_graph.ofstream = "conv_stem"
        if args.task in ["imagenet"]:
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
                                      trainable=True,)
                                )
            meta_graph.ofstream = "conv_stem_dw3x3"
            # Add 1 depthwise layer for 1st stride=2 block on ImageNet.
            writer.add(Convolutional_Proto(name="conv_stem_1x1",
                                      input="conv_stem_dw3x3",
                                      kernel_size=1,
                                      strides=1,
                                      filters=max(16, ps['num_stem_conv'] // 2),
                                      activation="linear",
                                      regularizer_strength=1e-5,
                                      batchnorm=True,
                                      use_bias=False,
                                      trainable=True,)
                                )
            meta_graph.ofstream = "conv_stem_1x1"
        # Body
        meta_graph.add_to_proto(writer,
                                num_repeats=ps['num_repeats'],
                                strides=ps['strides'],
                                num_channels=ps['num_channels'],
                                use_bias=False,
                                use_residual=True,
                                activation="relu",
                                proj_dim_per_cell=True,
                                pool_style="irb"
                                )
        writer.in_name = meta_graph.ofstream

        # Head
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
                                          use_dense_initializer=False)
                                )
            out_name = "conv1x1"
        else:
            out_name = writer.in_name

        # Add the average pool layer.
        writer.add(GlobalAvgPool_proto(name="avg_1k",
                                      input=out_name))
        out_name = "avg_1k"
        writer.finalized(args.task, out_name=out_name)
        writer.set_global_regularization(1e-5)
        prototxt = os.path.join(proto_dir, "{}-candidates-{}.prototxt".format(args.task, i))
        with open(prototxt, 'w') as fp:
            writer.dump(fp)

        config.prototxt = prototxt
        metrics = estimator.train_and_evaluate(config, max_model_mac=args.max_mac_limit, profile_only=args.profile_only)
        if metrics is not None:
            metrics.update({"hash": mask_hash})
            records.append(metrics)
            if i % 100 == 0 or i == args.budget - 1:
                print("Dumping to {}".format(record_path))
                with open(record_path, 'wb') as fp:
                    pickle.dump(records, fp)
            i += 1
        else:
            print("Skipping records with no metrics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None,
        help='Root directory to store records.')
    parser.add_argument("--yaml_cfg", type=str ,default=None,
        help="Search config.")
    parser.add_argument("--budget", type=int, nargs='?',
        help="Maximum sampled archs.", default=2000)
    parser.add_argument("--task", type=str, nargs='?',
        help="Task.", default=None,
        choices=["cifar10", "imagenet"])
    parser.add_argument("--profile_only", action='store_true',
        default=False)
    parser.add_argument("--worker_id", 
        type=int, default=0,
        help="Worker ID.")
    parser.add_argument("--protocol", 
        type=str, nargs='?',
        help="Protocol to use", default='new')
    parser.add_argument(
        "--max_mac_limit", 
        type=int, default=600, help="Maximum MAC limit for the model.")
    parser.add_argument(
        "--max_num_edges", 
        type=int, default=15, help="Maximum number of edges in an architecture.")
    # Constraint model macs. If exceed the maximum budget, stop training immediately.
    args = parser.parse_args()
    main(args)
