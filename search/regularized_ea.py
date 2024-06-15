from audioop import add
import os
import sys

from py import process
from pytest import param

sys.path.append(os.getcwd())

import pickle
import copy
from math import ceil

import multiprocessing as mp
import argparse
import json
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from joblib import Parallel, delayed, parallel_backend
from ptflops import get_model_complexity_info
import numpy as np

import torch
import torch.nn as nn

# Import native packages.
from src.utils.architecture import Architecture
from src.yaml_utils.yaml_parser import load_and_apply_yaml_config
from src.metagraph_v3.metagraph import MetaGraph_v3
from src.metagraph_v3.util_graph import random_connected_graph
from src.proto.writer import ProtoWriter
from src.proto.contrib import *
from core.predictor.predictor_utils import get_predictor

ps_settings_imagenet_monotone = {
    'strides': [2, 2, 2, 2], 'num_channels': [24, 48, 96, 192],
    'num_repeats': [1, 5, 5, 5], 'num_stem_conv': 32, 'stem_activation': "relu",
    'num_head_conv': 1280, "head_activation": "relu", 'num_stages': 4, 'num_nodes': 17
}

ps_settings_cifar10_monotone = {
    'strides': [1, 2, 2], 'num_channels': [16, 32, 64],
    'num_repeats': [3, 3, 3], 'num_stem_conv': 16, 'stem_activation': "relu",
    'num_head_conv': 0, "head_activation": "relu", 'num_stages': 3, 'num_nodes': 17
}

_task_in_dims = {
    'cifar10': 17*17*3,
    'cifar10-flops': 17*17*3,
    'imagenet': 17*17*4,
    'imagenet-flops': 17*17*4,
}

def maybe_copy_to_gpu(tensor_or_model, to_gpu: bool = False):
    return tensor_or_model if to_gpu else tensor_or_model.cuda()

def maybe_copy_to_cpu(tensor_or_model, to_cpu: bool = True):
    return tensor_or_model.cpu() if to_cpu else tensor_or_model

class MutateSpec:
    def add_edge_to_leaf_nodes(mask, min_edges, max_edges):
        n = mask.shape[0]
        mutated_mask = np.copy(mask)
        leaf_indices = np.arange(n)[(np.sum(mutated_mask,  1) == 0) & (np.sum(mutated_mask,  0) != 0)]
        all_indices = np.arange(n)[(np.sum(mutated_mask,  1) != 0) | (np.sum(mutated_mask,  0) != 0)]

        leaf_chosen = np.random.choice(leaf_indices)
        if np.sum(mask[:, leaf_chosen]) == leaf_chosen:
            mutated_mask, _ =  random_connected_graph(n, min_edges, max_edges)
            return mutated_mask
        rand_node_chosen = np.random.choice(all_indices)
        if rand_node_chosen < leaf_chosen:
            if mutated_mask[rand_node_chosen, leaf_chosen] == 0:
                mutated_mask[rand_node_chosen, leaf_chosen] = 1
                return mutated_mask
        elif rand_node_chosen > leaf_chosen:
            if mutated_mask[leaf_chosen, rand_node_chosen] == 0:
                mutated_mask[leaf_chosen, rand_node_chosen] = 1
                return mutated_mask            
        mutated_mask, _ =  random_connected_graph(n, min_edges, max_edges)
        return mutated_mask

    def delete_edge_to_leaf_nodes(mask, min_edges, max_edges):
        n = mask.shape[0]
        mutated_mask = np.copy(mask)
        leaf_indices = np.arange(n)[(np.sum(mutated_mask,  1) == 0) & (np.sum(mutated_mask,  0) != 0)]
        leaf_chosen = np.random.choice(leaf_indices)
        connected_indices = np.arange(n)[(mask[:, leaf_chosen] == 1)]
        rand_node_chosen = np.random.choice(connected_indices)
        if rand_node_chosen < leaf_chosen:
            if mutated_mask[rand_node_chosen, leaf_chosen] == 1:
                mutated_mask[rand_node_chosen, leaf_chosen] = 0
        else:
            print("Please check adjacency matrix.")
            mutated_mask, _ =  random_connected_graph(n, min_edges, max_edges)
            return mutated_mask
        # Return original if fails.
        return mutated_mask

    def random_mask(mask, min_edges, max_edges):
        n = mask.shape[0]
        mutated_mask, _ = random_connected_graph(n, min_edges, max_edges)
        return mutated_mask

def mutate_mask(mask, min_edges, max_edges):
    if np.sum(mask) <= min_edges:
        mutate_fn = np.random.choice([MutateSpec.add_edge_to_leaf_nodes, MutateSpec.random_mask])
    elif np.sum(mask) >= max_edges:
        mutate_fn = np.random.choice([MutateSpec.delete_edge_to_leaf_nodes, MutateSpec.random_mask])
    else:
        mutate_fn = np.random.choice([MutateSpec.add_edge_to_leaf_nodes, MutateSpec.delete_edge_to_leaf_nodes, MutateSpec.random_mask])
    
    return mutate_fn(mask, min_edges, max_edges)

def random_sample_from_metagraph(metagraph, best_mask, ps, max_edges):
    metagraph.mask = np.reshape(best_mask, [ps['num_stages'], ps['num_nodes'], ps['num_nodes']])
    idx = np.random.randint(0, metagraph.depth)
    metagraph.random_sampling_with_stage(idx, random_connected_graph, max_edges=max_edges)
    return metagraph.mask.flatten()

def mutate_from_metagraph(metagraph, best_mask, ps, max_edges, num_mutations):
    best_mask_3d = np.reshape(best_mask, [ps['num_stages'], ps['num_nodes'], ps['num_nodes']])
    mutated_mask = np.copy(best_mask_3d)
    for _ in range(num_mutations):
        idx = np.random.randint(0, metagraph.depth)
        mutated_idx_mask = np.copy(best_mask_3d[idx])
        mutated_idx_mask = mutate_mask(mutated_idx_mask, 2, max_edges)
        mutated_mask[idx] = np.copy(mutated_idx_mask)
    return mutated_mask.flatten()

def mask2str(mask):
    return "".join([str(x) for x in mask.flatten()])
    

@torch.no_grad()
def main(args):
    np.random.seed(args.random_seed)
    config = load_and_apply_yaml_config(args.yaml_cfg)
    with open(args.metagraph_path, 'rb') as fp:
        metagraph = pickle.load(fp)
    assert isinstance(metagraph, MetaGraph_v3)

    # get a simple mask to determine the dimension.
    dim = len(metagraph.mask.flatten())

    # Load both the accuracy model and latency model.
    # Load accuracy predictor
    accuracy_predictor = get_predictor(
        args.accuracy_predictor_nn_arch,
        None,
        _task_in_dims[args.task],
        None,
        None,
        args.accuracy_predictor_loss_fn_name,
        None,
        1,
    )
    accuracy_predictor.load_weights(args.accuracy_pretrain_ckpt_path)
    accuracy_predictor.eval()
    accuracy_predictor = maybe_copy_to_gpu(accuracy_predictor, args.use_gpu)

    if args.latency_pretrain_ckpt_path is not None:
        latency_predictor = get_predictor(
            args.latency_predictor_nn_arch,
            None,
            _task_in_dims[args.task],
            None,
            None,
            args.latency_predictor_loss_fn_name,
            None,
            1,
        )
        latency_predictor.load_weights(args.latency_pretrain_ckpt_path)    
        latency_predictor.eval()
        latency_predictor = maybe_copy_to_gpu(latency_predictor, args.use_gpu)
    else:
        latency_predictor = None
        input("Warning: No latency model. Press [Enter] to continue.")

    if args.task == "cifar10":
        ps = ps_settings_cifar10_monotone
    elif args.task == "imagenet":
        ps = ps_settings_imagenet_monotone
    else:
        raise NotImplementedError

    # Now, start the burn-in stage.
    all_populations_masks = []

    print("Start collecting initial population!")
    for i in tqdm(range(args.init_population)):
        metagraph.random_sampling(random_connected_graph, max_edges=args.max_num_edges)
        all_populations_masks.append(metagraph.mask.flatten())

    all_populations_masks = np.asarray(all_populations_masks).astype(np.float32)

    history_best_masks = []
    history_best_scores = []
    
    history = {'round':[], 'best_acc':[]}

    best_score = 0

    for i in tqdm(range(args.n_generations)):
        # Select parent architecture.
        sampled_indices = np.random.choice(
            len(all_populations_masks),
            size=int(len(all_populations_masks) * args.sample_ratio), 
            replace=False)
        sampled_population_masks = all_populations_masks[sampled_indices]
        sampled_population_pred_accs = accuracy_predictor.predict(maybe_copy_to_gpu(
                torch.from_numpy(sampled_population_masks), (args.use_gpu == 1))).cpu().numpy().flatten()

        if latency_predictor is not None:
            sampled_population_pred_lats = latency_predictor.predict(maybe_copy_to_gpu(
                torch.from_numpy(sampled_population_masks), (args.use_gpu == 1))).cpu().numpy().flatten()
            # lat_penalty = np.minimum(1.0, np.power(pred_lats / args.constraint, args.alpha))
            lat_penalty = (sampled_population_pred_lats <= args.constraint).astype(np.float32)
            scores = sampled_population_pred_accs * lat_penalty
        else:
            scores = sampled_population_pred_accs

        parent_arch_idx = np.argmax(scores)
        parent_mask = np.copy(sampled_population_masks[parent_arch_idx])
        parent_arch_score = np.copy(scores[parent_arch_idx])
        
        # Now, generate child architecture.
        # num_mutations = ceil(3 * (1 - i / args.num_rounds))
        num_mutations = 1
        mask_count = 0
        child_masks = []
        
        while mask_count < args.n_childs:
            mutated_mask = mutate_from_metagraph(
                metagraph,
                np.copy(parent_mask),
                ps, args.max_num_edges, num_mutations)
            child_masks.append(mutated_mask)
            mask_count += 1
        
        child_masks = np.asarray(child_masks).astype(np.float32)
        child_pred_accs = accuracy_predictor.predict(maybe_copy_to_gpu(
            torch.from_numpy(child_masks), (args.use_gpu == 1))).cpu().numpy().flatten()

        if latency_predictor is not None:
            child_pred_lats = latency_predictor.predict(maybe_copy_to_gpu(
                torch.from_numpy(child_masks), (args.use_gpu == 1))).cpu().numpy().flatten()
            # lat_penalty = np.minimum(1.0, np.power(pred_lats / args.constraint, args.alpha))
            child_lat_penalty = (child_pred_lats <= args.constraint).astype(np.float32)
            child_scores = child_pred_accs * child_lat_penalty
        else:
            child_scores = child_pred_accs

        # Add child masks to the whole population.
        all_populations_masks = np.concatenate([all_populations_masks, child_masks], axis=0)
        all_populations_masks = all_populations_masks[len(child_masks):]

        # Copy child masks
        sorted_child_masks_indices = np.argsort(child_scores)
        sorted_child_masks_indices = sorted_child_masks_indices[::-1]

        for idx in range(min(len(child_masks), 5)):
            history_best_masks.append(np.copy(child_masks[sorted_child_masks_indices[idx]]))
            history_best_scores.append(np.copy(child_masks[sorted_child_masks_indices[idx]]))

        #print(best_arch_score)
        history['best_acc'].append(np.max(child_scores).item())
        history['round'].append(i)
        if np.max(child_scores) > best_score:
            best_score = np.max(child_scores)
        if i % 20 == 0:
            print("Round {}: Best Score: {}".format(i, best_score))

    history_best_masks = np.asarray(history_best_masks)

    _, unique_indices = np.unique(history_best_masks, axis=0, return_index=True)
    history_best_masks = history_best_masks[unique_indices]

    if latency_predictor is not None:
        history_best_latency = latency_predictor.predict(
            maybe_copy_to_gpu(torch.from_numpy(history_best_masks), (args.use_gpu == 1))).cpu().numpy().flatten()
    else:
        history_best_latency = None

    history_best_accs = accuracy_predictor.predict(
        maybe_copy_to_gpu(torch.from_numpy(history_best_masks), (args.use_gpu == 1))).cpu().numpy().flatten()
        
    sorted_args = np.argsort(history_best_accs)
    sorted_args = sorted_args[::-1]

    history_best_accs = history_best_accs[sorted_args]
    history_best_masks = history_best_masks[sorted_args]
    if history_best_latency is not None:
        history_best_latency = history_best_latency[sorted_args]

    if not os.path.exists(args.proto_dir):
        os.makedirs(args.proto_dir)
        
    json_path = os.path.join(args.proto_dir, "history.json")
    with open(json_path, 'w') as fp:
        json.dump(history, fp, indent=0)
    result_dict = {}

    counter = 0
    idx = 0

    all_flops_sampled = []

    while counter < args.top_k and idx < len(history_best_masks):
        metagraph.mask = np.reshape(history_best_masks[idx],
                                    [ps['num_stages'], ps['num_nodes'], ps['num_nodes']])
        writer = ProtoWriter("{}-candidates-{}.prototxt".format(args.task, idx))
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
                                      use_dense_initializer=False))
        metagraph.ofstream = "conv_stem"
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
                                      trainable=True,))
            metagraph.ofstream = "conv_stem_dw3x3"

            # Add 1 depthwise layer for 1st stride=2 block on ImageNet.
            writer.add(Convolutional_Proto(name="conv_stem_1x1",
                                      input="conv_stem_dw3x3",
                                      kernel_size=1,
                                      strides=1,
                                      filters=max(16, ps['num_stem_conv']),
                                      activation="relu",
                                      regularizer_strength=1e-5,
                                      batchnorm=True,
                                      use_bias=False,
                                      trainable=True,))
            metagraph.ofstream = "conv_stem_1x1"
        
        # Body
        metagraph.add_to_proto(writer,
                               num_repeats=ps['num_repeats'],
                               strides=ps['strides'],
                               num_channels=ps['num_channels'],
                               use_bias=False,
                               use_residual=True,
                               activation="relu",
                               linear_act_with_proj_1x1=True,
                               proj_dim_per_cell=True,
                               pool_style="max-pool"
                )
        writer.in_name = metagraph.ofstream
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
                                          use_dense_initializer=False))
            out_name = "conv1x1"
        else:
            out_name = writer.in_name
        # Add the average pool layer.
        writer.add(GlobalAvgPool_proto(name="avg_1k",
                                      input=out_name))

        out_name = "avg_1k"
        writer.finalized(args.task, out_name=out_name)
        writer.set_global_regularization(1e-5)

        proto_name = 'micronas-{}-candidate-{}.prototxt'.format(args.task, idx)
        prototxt = os.path.join(args.proto_dir, proto_name)
        with open(prototxt, 'w') as fp:
            writer.dump(fp)
        # Measure flops.
        model = Architecture(prototxt)
        # Testing model latency.
        flops, params = get_model_complexity_info(model, (3, config.image_size, config.image_size),
                                                  as_strings=False, verbose=False,
                                                  print_per_layer_stat=False, ignore_modules=[nn.ReLU, nn.BatchNorm2d])
        print("FLOPS: {} M".format(flops / 1e6))
        print("Params: {} M".format(params / 1e6))
        if params / 1e6 >= args.max_params or flops in all_flops_sampled:
            idx += 1
            os.remove(prototxt)
            continue
        all_flops_sampled.append(flops)
        counter += 1
        mask_list = metagraph.mask.astype(np.int).flatten().tolist()
        mask_list = [str(x) for x in mask_list]
        mask_hash = "".join(mask_list)
        res = {
            proto_name: {
            'FLOPS(M):': float(flops / 1e6),
            'Params(M)': float(params / 1e6),
            'pred_acc': float(history_best_accs[idx]),
            'hash': mask_hash}
        }
        if history_best_latency is not None:
            res[proto_name].update({'pred_latency': float(history_best_latency[idx])})
        result_dict.update(res)
        idx += 1

    json_path = os.path.join(args.proto_dir, "summary.json")
    with open(json_path, 'w') as fp:
        json.dump(result_dict, fp, indent=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="imagenet",
                        help="Task to choose.")
    parser.add_argument("--metagraph_path", type=str, default=None,
                        help="Path to the meta-graph.")
    parser.add_argument("--init_population", type=int, default=10000,
                        help="Initial Population.")
    parser.add_argument("--n_childs", type=int, default=50,
                        help="Number of childs for each generation.")
    parser.add_argument("--n_generations", type=int, default=10000,
                        help="Number of generations for Regularized EA.")
    parser.add_argument("--sample_ratio", type=float, default=0.75,
                        help="Sample ratio for regularized EA.")
    parser.add_argument("--constraint", type=float, default=0.18,
                        help="Constraint for the latency model.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-k models to keep.")
    parser.add_argument("--proto_dir", type=str, default=None,
                        help="Directory to output the model.")
    parser.add_argument("--yaml_cfg", type=str, default=None,
                        help="Yaml config for measurements.")
    parser.add_argument("--max_params", type=float, default=999.999,
                        help="Maximum parameter count.")
    parser.add_argument("--alpha", type=float, default=-0.18,
                        help="Alpha to trade-off acc and latency.")
    parser.add_argument("--max_num_edges", type=int, default=15, 
                        help="Maximum number of edges in an architecture.")
    parser.add_argument("--accuracy_predictor_nn_arch", type=str, default="dense-nn",
                        help="NN architecture to train the accuracy predictor",
                        choices=[
                                'embedding-nn',
                                'dense-nn',
                                'dense-sparse-nn',
                            ]
    )
    parser.add_argument("--random_seed", type=int, default=None, help="Random Seed.")
    parser.add_argument(
        '--accuracy_predictor_loss_fn_name', type=str, default='mse-loss',
        help="Accuracy Predictor loss function name to choose from."
    )
    parser.add_argument("--accuracy_pretrain_ckpt_path", type=str,
                        default=None,
                        help="Path to the accuracy model checkpoint.")
    parser.add_argument("--latency_pretrain_ckpt_path", type=str,
                        default=None,
                        help="Path to the latency model checkpoint.")

    parser.add_argument(
        '--latency_predictor_nn_arch', type=str, default="dense-nn",
        help="NN architecture to train the latency predictor",
        choices=[
            'embedding-nn',
            'dense-nn',
            'dense-sparse-nn',
        ]
    )
    parser.add_argument(
        '--latency_predictor_loss_fn_name', type=str, default='mse-loss',
        help="Latency Predictor loss function name to choose from."
    )
    parser.add_argument("--use_gpu", type=int, default=1, help="Whether use GPU or not.")
    args = parser.parse_args()
    main(args)
