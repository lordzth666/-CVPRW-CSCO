import os
import sys
sys.path.append(".")
from tqdm import tqdm

import argparse
from typing import (
    Any,
    Optional
)
from matplotlib import pyplot as plt
from itertools import combinations
import numpy as np
import torch
# Train the aug records. Using predictors.
from scipy.stats import pearsonr, kendalltau
from nasflow.dataset.nas_dataset import NASDataSet
from core.predictor.predictor_utils import get_predictor
from tqdm.contrib.concurrent import process_map

@torch.no_grad()
def eval_corrleation_with_predictor(dataset_iterator, predictor):
    all_outputs = []
    all_labels = []
    predictor.eval()
    for _, batch in enumerate(dataset_iterator):
        inputs, labels = [x[0] for x in batch], [x[1] for x in batch]
        inputs, labels = torch.tensor(inputs), torch.tensor(labels)
        if predictor.use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = predictor.predict(inputs)
        all_outputs.append(outputs)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, 0)
    all_labels = torch.cat(all_labels, 0)
    all_outputs = all_outputs.cpu().numpy().flatten()
    all_labels = all_labels.cpu().numpy()
    pearson_r, _ = pearsonr(all_outputs.flatten(), all_labels)
    kendall_tau, _ = kendalltau(all_outputs.flatten(), all_labels)
    mse_loss = np.mean(np.square(all_outputs.flatten() - all_labels))
    return pearson_r, kendall_tau, mse_loss


def train_and_evaluate_per_predictor(
        nas_dataset,
        num_inputs: int,
        in_dims: int,
        args: Any,
        verbose: bool = False,  
        num_epochs: int = 150,
        batch_size: int = 64,
        **kwargs):
    """
    Train & Evaluate a predictor using different hyperparameters.
    """
    print(kwargs)
    predictor = get_predictor(
        args.predictor_nn_arch,
        nas_dataset,
        in_dims,
        num_epochs,
        num_inputs,
        args.predictor_loss_fn_name,
        args.ranking_loss_fn_name,
        batch_size=batch_size,
        **kwargs)
    # print(predictor.core_ml_arch)
    predictor.load_weights(args.pretrain_ckpt_path, args.pretrain_exclude_ckpt_keys)
    predictor.fit(verbose)
    predictor.save_weights(args.save_ckpt_path)
    predictor.eval()    
    train_dataset_iterator = nas_dataset.iter_map_and_batch(
        split='train', shuffle=False, 
        drop_last_batch=False, batch_size=128)
    test_dataset_iterator = nas_dataset.iter_map_and_batch(
        split='test', shuffle=False, 
        drop_last_batch=False, batch_size=128)
    train_pearson, train_kendall, train_loss = eval_corrleation_with_predictor(
        train_dataset_iterator, predictor)
    test_pearson, test_kendall, test_loss = eval_corrleation_with_predictor(
        test_dataset_iterator, predictor)
    print("-----------------------------------------------------------------------")
    print("Evaluating config: lr={:.8f}, wd={:.8f}, margin={:.8f}, ranking_coef={:.8f}".format(
        kwargs['learning_rate'], kwargs['weight_decay'], kwargs['margin'], kwargs['ranking_loss_coef']))
    print("Training Pearson: {:.5f}, Training Kendall Tau: {:.5f}".format(train_pearson, train_kendall))
    print("Testing Pearson: {:.5f}, Testing Kendall Tau: {:.5f}".format(test_pearson, test_kendall))
    print("Training Loss: {:.5f}, Testing Loss: {:.5f}".format(train_loss, test_loss))
    print("-----------------------------------------------------------------------")
    # Objective should be negative kendall_tau.
    if np.isnan(test_kendall):
        return 99999
    else:
        return test_loss


def get_mask_with_triu(mask):
    assert len(np.shape(mask)) == 1, "Mask should be 1D flatten."
    cascaded_masks = np.reshape(mask, [-1, 10, 10])
    new_masks = []
    for idx in range(cascaded_masks.shape[0]):
        mask_to_convert = cascaded_masks[idx]
        mask_triu_indices = np.triu_indices(10)
        new_masks.append(mask_to_convert[mask_triu_indices])
    new_masks = np.asarray(new_masks)
    new_masks = new_masks.flatten().tolist()
    return new_masks


class MapFn(object):
    @staticmethod
    def cifar10_map_fn(item, normalize=True, minval=70.48, maxval=79.51):
        mask = [float(x) for x in item['hash']]
        if normalize:
            acc = (item['best_acc1'] - minval) / (maxval - minval)
        else:
            acc = item['best_acc1'] / 100
        return mask, acc

    @staticmethod
    def cifar10_flops_map_fn(item, normalize=True, minval=10.88, maxval=99.27):
        mask = [float(x) for x in item['hash']]
        if normalize:
            macs = (item['MACs(M)'] - minval) / (maxval - minval)
        else:
            macs = item['MACs(M)'] / 100
        # mask = get_mask_with_triu(mask)
        return mask, macs

    @staticmethod
    def cifar10_params_map_fn(item, normalize=True, minval=0.0551, maxval=0.4318):
        mask = [float(x) for x in item['hash']]
        if normalize:
            params = (item['params(M)'] - minval) / (maxval - minval)
        else:
            params = item['params(M)'] / 100
        # mask = get_mask_with_triu(mask)
        return mask, params
    
    @staticmethod
    def imagenet_flops_map_fn(item, normalize=True, minval=156.02, maxval=437.90):
        mask = [float(x) for x in item['hash']]
        if normalize:
            macs = (item['MACs(M)'] - minval) / (maxval - minval)
        else:
            macs = item['MACs(M)'] / 1000
        return mask, macs


    @staticmethod
    def imagenet_lat_map_fn(item, normalize=True, minval=24.24, maxval=70.63):
        mask = [float(x) for x in item['hash']]
        if normalize:
            lat = (item['Mean_Lat(ms)'] - minval).item() / (maxval - minval)
        else:
            lat = item['Mean_Lat(ms)'].item() / 1000
        return mask, lat


    @staticmethod
    def imagenet_map_fn(item, normalize=True, minval=72.19, maxval=87.59):
        mask = [float(x) for x in item['hash']]
        if normalize:
            acc = (item['best_acc1'] - minval) / (maxval - minval)
        else:
            acc = item['best_acc1'] / 100
        return mask, acc


_task_map_fn = {
    'cifar10': MapFn.cifar10_map_fn,
    'cifar10-flops': MapFn.cifar10_flops_map_fn,
    'cifar10-params': MapFn.cifar10_params_map_fn,
    "imagenet": MapFn.imagenet_map_fn,
    "imagenet-flops": MapFn.imagenet_flops_map_fn,
    "imagenet-lats": MapFn.imagenet_lat_map_fn,
}


num_nodes = 13
_task_in_dims = {
    'cifar10': num_nodes * num_nodes * 3,
    'cifar10-flops': num_nodes * num_nodes * 3,
    'cifar10-params': num_nodes * num_nodes * 3,
    'imagenet': num_nodes * num_nodes * 6,
    'imagenet-flops': num_nodes * num_nodes * 6,
    'imagenet-lats': num_nodes * num_nodes * 6,
}


def main(args):
    # Initialize dataset.
    test_split = .01 if args.all_in else 0.15
    print("Testing split: {}".format(test_split))
    record_name_to_rec_dict = {
        "cifar10": "cifar10",
        "cifar10-flops": "cifar10",
        "cifar10-params": "cifar10",
        "imagenet": "imagenet",
        "imagenet-flops": "imagenet",
        "imagenet-lats": "imagenet",
    }
    record_name = "{}.records".format(record_name_to_rec_dict[args.task])
    nas_dataset = NASDataSet(
        args.record_dir, pattern="shard-*",
        record_name=record_name,
        map_fn=_task_map_fn[args.task], max_records=10000000,
        cache=True, test_size=test_split
    )
    # Set torch manual seed.
    torch.manual_seed(args.seed)
    train_and_evaluate_per_predictor(
        nas_dataset, None, _task_in_dims[args.task], args, verbose=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        margin=args.ranking_loss_margin,
        ranking_loss_coef=args.ranking_loss_coef,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_dir", default=None, help="Directory for records.")
    parser.add_argument("--all_in", action='store_true', default=False,
        help="Whether all in all records or not.")
    parser.add_argument("--task", type=str, default=None,
                        help="Task to train acc predictor.")
    # Predictor Path handlers.
    parser.add_argument("--pretrain_ckpt_path", default=None, 
                        type=str, help="Pretraining checkpoint path.")
    parser.add_argument("--pretrain_exclude_ckpt_keys", type=str, nargs='*',
        default=None)
    parser.add_argument("--save_ckpt_path", default=None, 
                        type=str, help="Predictor save path.")
    # Predictor loss fn.
    # Predictor Arch Choice.
    parser.add_argument(
        '--predictor_nn_arch', type=str, default="dense-nn",
        help="NN architecture to train the predictor",
        choices=[
            'embedding-nn',
            'dense-nn',
            'dense-sparse-nn',
        ],
    )
    parser.add_argument(
        '--predictor_loss_fn_name', type=str, default="mse-loss",
        help="Predictor loss function name to choose from.",
        choices=["mse-loss", "smoothed-l1-loss"],
    )
    # Predictor Hyperparameters.
    parser.add_argument(
        '--ranking_loss_fn_name', type=str, default='margin-ranking-loss',
        help="Ranking loss function name for predictor training."
    )
    parser.add_argument("--ranking_loss_coef", type=float, default=0.0,
        help="Margin for the margin ranking loss.")
    parser.add_argument("--ranking_loss_margin", type=float, default=0.0,
        help="Margin for the margin ranking loss.")
    parser.add_argument("--learning_rate", type=float, default=0.01,
        help="Learning rate for the predictor training.")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
        help="Weight decay of the predictor.")
    parser.add_argument("--batch_size", type=int, default=128,
        help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=100,
        help="Number of epochs.")
    # Random Seed.
    parser.add_argument("--seed", type=int, default=233,
        help="Random Seed.")
    args = parser.parse_args()
    main(args)
