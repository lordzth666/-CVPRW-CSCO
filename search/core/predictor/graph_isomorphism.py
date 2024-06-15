import os
import sys
import copy
import pickle
import numpy as np

from tqdm import tqdm

def augment_mask(mask, op_len: int = 3, max_tries_each_windows=10, verbose: bool = False):
    n = mask.shape[0]
    for j in range(1, op_len+1):
        # First, get all isomorphic nodes.
        all_iso_nodes = []
        node_idx = j
        while node_idx < n:
            all_iso_nodes.append(node_idx)
            node_idx += op_len
        if len(all_iso_nodes) < 2:
            continue
        for i in range(max_tries_each_windows):
            chosen_nodes = np.random.choice(all_iso_nodes, 2, replace=False)
            node1, node2 = min(chosen_nodes[0], chosen_nodes[1]), max(chosen_nodes[0], chosen_nodes[1])
            out_degree_edges = np.sum(mask[node1, node1:node2])
            in_degree_edges = np.sum(mask[node1:node2, node2])
            # Lack constraint: node must have an in-degree (i.e., in the graph)
            in_degree_node1 = np.sum(mask[:node1, node1])
            in_degree_node2 = np.sum(mask[:node2, node2])
            if in_degree_edges == 0 and out_degree_edges == 0 and mask[node1, node2] == 0 and in_degree_node1 != 0 and in_degree_node2 != 0:
                if verbose: 
                    print("Sucess! Change node {}, and node {}!".format(node1, node2))
                mutated_mask = np.copy(mask)
                mutated_mask[:, node1] = np.copy(mask[:, node2])
                mutated_mask[:, node2] = np.copy(mask[:, node1])
                mutated_mask[node1, :] = np.copy(mask[node2, :])
                mutated_mask[node2, :] = np.copy(mask[node1, :])
                return mutated_mask

    if verbose: print("Failed!")
    return mask

def gather_all_records_in_dir(exp_dir, task="cifar10"):
    all_records_path = os.listdir(exp_dir)
    all_recs = []
    for rec in all_records_path:
        if rec.startswith("shard-"):
            with open(os.path.join(exp_dir, rec, "{}.records".format(task)), 'rb') as fp:
                records = pickle.load(fp)
                all_recs += records
    return all_recs

def mask2str(mask):
    return "".join([str(x) for x in mask.flatten()])

def main(args):
    num_nodes = 13
    task = "cifar10"
    exp_dir = "./exps-0213/hybnas-{}-search/".format(task)
    exp_aug_dir = "./exps-0213/hybnas-{}-search-augmented/".format(task)
    os.makedirs(exp_aug_dir, exist_ok=True)

    num_rounds = 2000
    aug_steps = 5000

    for idx in range(num_rounds):
        print("Round {} ...".format(idx))
        all_new_records_hashes = []
        # Parse all args before-hand to avoid duplication.
        for shard_name in os.listdir(exp_dir):
            if not shard_name.startswith("shard-"):
                continue
            with open(os.path.join(exp_dir, shard_name, "{}.records".format(task)), "rb") as fp:
                all_records = pickle.load(fp)
            all_new_records_hashes += [x['hash'] for x in all_records]
        print("Loaded {} records in hash from {}!".format(len(all_new_records_hashes), exp_dir))
        for shard_name in os.listdir(exp_dir):
            if not shard_name.startswith("shard-"):
                continue
            with open(os.path.join(exp_dir, shard_name, "{}.records".format(task)), "rb") as fp:
                all_records = pickle.load(fp)
            all_new_records = all_records
            num_records = len(all_records)
            for _ in tqdm(range(aug_steps)):
                record_idx = np.random.choice(num_records)
                all_records[record_idx]['hash'] = all_records[record_idx]['hash']
                record_mask = [int(x) for x in all_records[record_idx]['hash']]
                record_mask = np.reshape(record_mask, [-1, num_nodes, num_nodes])
                # Select a index to perform augmentation.
                stage_idx = np.random.choice(record_mask.shape[0])
                new_mask = augment_mask(record_mask[stage_idx], verbose=False, op_len=4)
                if mask2str(new_mask) != mask2str(record_mask[stage_idx]):
                    new_record_mask = np.copy(record_mask)
                    new_record_mask[stage_idx] = new_mask.copy()
                    new_mask_hash = mask2str(new_record_mask)
                    if new_mask_hash not in all_new_records_hashes:
                        all_new_records_hashes.append(new_mask_hash)
                        new_record_item = copy.deepcopy(all_records[record_idx])
                        new_record_item['hash'] = new_mask_hash
                        all_new_records.append(copy.deepcopy(new_record_item))
            print(len(all_new_records))
            os.makedirs(os.path.join(exp_aug_dir, shard_name), exist_ok=True)
            with open(os.path.join(exp_aug_dir, shard_name, "{}.records".format(task)), "wb") as fp:
                pickle.dump(all_new_records, fp)

        # Move exp_dir to exp_aug_dir for recursive augmentation.
        exp_dir = exp_aug_dir

if __name__ == "__main__":
    main(None)
