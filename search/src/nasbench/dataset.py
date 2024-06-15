import numpy as np
from tqdm import tqdm
from itertools import combinations

# Latency definition
RASP_LAT = 1
PIXEL_LAT = 2
GPU_LAT = 3

def generate_hash_kernel(size):
    return np.random.normal(loc=0, scale=1, size=size)

def hash_func(np_array, kernel):
    return np.sum(np.abs(np_array * kernel), axis=(1,2,3))

def get_unique_masks(masks, return_idx=True):
    if not isinstance(masks, np.ndarray):
        masks = np.asarray(masks)
    hash_kernel = generate_hash_kernel(masks[0].shape)
    hashed_masks = hash_func(masks, hash_kernel)
    _, indices = np.unique(hashed_masks, return_index=True)
    if return_idx:
        return masks[indices].copy(), indices
    else:
        return masks[indices].copy()
    pass

def print_mask_info(masks):
    num_samples = masks.shape[0]
    num_stages = masks.shape[1]
    n = masks.shape[2]

    for i in range(num_samples):
        print("Sample %d:" %i)
        for j in range(num_stages):
            print("Stage %d" %j)
            for l1 in range(n-1):
                for l2 in range(l1+1, n):
                    if masks[i][j][l1][l2] == 1:
                        print("%d --> %d" %(l1, l2))

# Dataset for benchmarking NAS search process.
class Dataset:
    def __init__(self, ops_def, nodes, num_stages, name="Dataset_V1"):
        self.nodes = nodes
        self.ops_def = ops_def
        self.masks = []
        self.num_stages = num_stages
        self.name = name
        # Add the performance metrics checklist
        self.accs = []
        self.macs = []
        self.latencies = []
        self.rasp_latencies = []
        self.pixel_latencies = []
        self.gpu_latencies = []
        # self.pixel_latency = []
        # Add iteration pointers
        self.num_examples = 0
        self.ptr = 0
        # Add configuration points
        self.cfg = {}
        #Augmentation Flag
        self.augmented=False

    def _add_cfg_arg(self, key, value):
        self.cfg[key] = value

    def add_params(self, scale, pool, replicate):
        """
        Add scale, pool and replicate parameters into the NASbench dataset.
        :param scale:
        :param pool:
        :param replicate:
        :return:
        """
        self._add_cfg_arg('scale', scale)
        self._add_cfg_arg('pool', pool)
        self._add_cfg_arg('replicate', replicate)

    def add_data(self, masks, acc, mac, latency):
        """
        Input masks
        :param masks: Input mask of shape [depth, width, nodes, nodes] or [depth, nodes, nodes].
        Due to deprecation, the 4-D mask shape will be shrinked to 3-D after input.
        :param acc: []
        :param mac: []
        :param latency: []
        :return:
        """
        if len(np.shape(masks)) == 4:
            masks_ = np.squeeze(masks, axis=1)
        else:
            masks_ = masks.copy()

        # First, we will do an query of the existing mask.
        ret1, ret2, ret3 = self.query(masks_)
        if ret1 == -1:
            self.masks.append(masks_.copy())
            self.accs.append(acc)
            self.macs.append(mac)
            self.latencies.append(latency)
        else:
            self.update(masks_, mac, acc, latency)
        print(self.accs)

    def _is_equal_mask(self, mask1, mask2):
        return np.array_equal(mask1, mask2)

    def query(self, mask, device_type=0):
        """
        Query an input mask. The input mask should have shape (num_stages, 1).
        :param mask:
        :return:
        """
        num_cases = np.shape(self.masks)[0]
        for i in range(num_cases):
            if self._is_equal_mask(mask, self.masks[i]):
                if device_type == PIXEL_LAT:
                    latency = self.pixel_latencies[i]
                elif device_type == RASP_LAT:
                    latency = self.rasp_latencies[i]
                elif device_type == GPU_LAT:
                    latency = self.gpu_latencies[i]
                else:
                    latency = self.latencies[i]
                return self.macs[i], self.accs[i], latency
        print("Query failed! No such masking is found.")
        return -1, -1, -1

    def update(self, mask, mac, acc, latency, moving_average=0.5):
        """
        Update the current mask by using an moving average of 0.5.
        As collisions are not likely, moving average of 0.5 can possibly lead to good effects.
        :param mask:
        :param mac:
        :param acc:
        :param latency:
        :param moving_average:
        :return:
        """
        num_cases = np.shape(self.masks)[0]
        for i in range(num_cases):
            if self._is_equal_mask(mask, self.masks[i]):
                self.macs[i] = self.macs[i] * moving_average + (1-moving_average) * mac
                self.accs[i] = self.accs[i] * moving_average + (1-moving_average) * acc
                # This is a dummy update.
                self.latencies[i] = self.latencies[i] * moving_average + (1-moving_average) * latency
                print("Updated masking data.")
                break

    def reset(self):
        # Reset the dataset for iteration.
        self.ptr = 0
        self.num_examples = len(self.accs)

    def shuffle_one_example(self, device_type=0):
        self.num_examples = len(self.accs)
        indice = np.random.randint(0, self.num_examples)
        if device_type == PIXEL_LAT:
            latency = self.pixel_latencies[indice]
        elif device_type == RASP_LAT:
            latency = self.rasp_latencies[indice]
        elif device_type == GPU_LAT:
            latency = self.gpu_latencies[indice]
        else:
            latency = self.latencies[indice]
        return self.masks[indice], self.accs[indice], self.macs[indice], latency

    def get_next(self, device_type=0):
        mask, acc, mac = self.masks[self.ptr], self.accs[self.ptr], self.macs[self.ptr]
        try:
            if device_type == PIXEL_LAT:
                latency = self.pixel_latencies[self.ptr]
            elif device_type == RASP_LAT:
                latency = self.rasp_latencies[self.ptr]
            elif device_type == GPU_LAT:
                latency = self.gpu_latencies[self.ptr]
            else:
                latency = self.latencies[self.ptr]
        except Exception:
            print("Warning: Use fake latency.")
            latency = 0
        self.ptr = (self.ptr + 1) % self.num_examples
        return mask, acc, mac, latency

    def concat(self, dataset_):
        """
        Concatenate the data in the target dataset 'dataset_'.
        :param dataset_:
        :return:
        """
        assert isinstance(dataset_, Dataset)
        self.masks = np.concatenate([self.masks, dataset_.masks], axis=0)
        self.macs = np.concatenate([self.macs, dataset_.macs], axis=0)
        self.accs = np.concatenate([self.accs, dataset_.accs], axis=0)
        self.latencies = np.concatenate([self.latencies, dataset_.latencies], axis=0)
        try:
            self.rasp_latencies = np.concatenate([self.rasp_latencies, dataset_.rasp_latencies], axis=0)
            self.gpu_latencies = np.concatenate([self.gpu_latencies, dataset_.gpu_latencies], axis=0)
            self.pixel_latencies = np.concatenate([self.pixel_latencies, dataset_.pixel_latencies], axis=0)
        except Exception:
            print("Warning: No device latencies detected.")
        self.num_examples = len(self.accs)

    def assign_latency(self, net_id, value, device_type):
        # Shiyu: Please use this function to assign latency. Please call this initailizer before assigning latency.
        if device_type == PIXEL_LAT:
            self.pixel_latencies[net_id] = value
        elif device_type == RASP_LAT:
            self.rasp_latencies[net_id] = value
        elif device_type == GPU_LAT:
            self.gpu_latencies[net_id] = value
        else:
            raise NotImplementedError("Current latency device is not supported.")
        pass

    def init_device_latency(self, device_type):
        # Please call this initailizer before assigning latency.
        if device_type == PIXEL_LAT:
            self.pixel_latencies = np.zeros_like(self.latencies, dtype=np.float32)
        elif device_type == RASP_LAT:
            self.rasp_latencies = np.zeros_like(self.latencies, dtype=np.float32)
        elif device_type == GPU_LAT:
            self.gpu_latencies = np.zeros_like(self.latencies, dtype=np.float32)
        else:
            raise NotImplementedError("Bad device definition!")

    def shuffle_examples(self, num_examples, device_type):
        self.num_examples = len(self.accs)
        if num_examples > self.num_examples:
            print("Number of examples exceeeds the volume of the NASBench dataset ...")
            print("Using the full dataset ...")
            num_examples = self.num_examples
        indices = np.random.choice(np.arange(self.num_examples), size=(num_examples), replace=False)
        try:
            if device_type == PIXEL_LAT:
                latency = np.asarray(self.pixel_latencies)[indices]
            elif device_type == RASP_LAT:
                latency = np.asarray(self.rasp_latencies)[indices]
            elif device_type == GPU_LAT:
                latency = np.asarray(self.gpu_latencies)[indices]
            else:
                latency = np.asarray(self.latencies)[indices]
        except Exception:
            print("Warning: Use fake latency.")
            latency = np.zeros_like(indices)
        return np.asarray(self.masks)[indices], np.asarray(self.accs)[indices], np.asarray(self.macs)[indices], latency


    def cleanup_nasbench(self):
        self.num_examples = len(self.accs)
        counter = 0
        blacklist_indices = []
        for i in tqdm(range(self.num_examples)):
            flag = False
            for j in range(i):
                if self._is_equal_mask(self.masks[i], self.masks[j]):
                    flag = True
                    print("Mask %d is identical to mask %d! Accuracy: %f vs. %f" %(i, j, self.accs[i], self.accs[j]))
            if flag:
                blacklist_indices.append(i)
                counter += 1
        if counter == 0:
            return
        print("Removing %d blacklist ops ..." %counter)
        print(blacklist_indices)
        blacklist_indices = np.asarray(blacklist_indices)

        arr = np.ones(self.num_examples)
        arr[blacklist_indices] = 0
        indices = (arr == 1)

        self.masks = self.masks[indices]
        self.accs = self.accs[indices]
        self.macs = self.macs[indices]
        self.latencies = self.latencies[indices]
        try:
            self.rasp_latencies = self.rasp_latencies[indices]
            self.pixel_latencies = self.pixel_latencies[indices]
            self.gpu_latencies = self.gpu_latencies[indices]
        except Exception:
            pass
        self.num_examples = len(self.accs)
        print("Done. New NASbench has %d examples." %self.num_examples)

    def graph_aug(self, mult=None, srate=1.0, force=False):
        '''
        Augmente the dataset by adding all the isomorphic (defined by operation rather than topology)
        matrices into the current dataset.
        Current Algorithm is under the assumption that each unique operation will appear at most twice
        in the operation list.

        :param: Force: Conduct augmentation on already augmented dataset
        '''
        if hasattr(self, 'augmented') and self.augmented:
            if force:
                print("Identical samples might be generated! USE WITH CAUTION")
            else:
                print("This dataset has been augmented!")
                return
        if not isinstance(self.masks, np.ndarray):
            self.masks = np.asarray(self.masks)
        n_stages = self.masks.shape[1]
        n_nodes = self.masks.shape[2]

        #Compute the hash of each operation
        hash_func = lambda x: hash(frozenset(x.items()))
        hashv = np.vectorize(hash_func)

        n_ops = np.unique(hashv(self.ops_def[0][0])).shape[0]
        permutations = []

        for stage in range(n_stages):
            current_ops = hashv(self.ops_def[stage][0])
            _, indices = np.unique(current_ops, return_inverse=True)
            perm_list = [np.arange(0, n_nodes),]
            for op in range(n_ops):
                #print(len(perm_list))
                pair = np.nonzero(indices==op)[0]
                pairs = list(combinations(pair, 2))

                new_perm_list = []
                #print("OP#",op)
                #print(pairs)
                pid = 0
                for perm in perm_list:
                    for p in pairs:
                        if 0 not in p:
                            new_perm = perm.copy()
                            new_perm[perm==p[0]] = p[1]
                            new_perm[perm==p[1]] = p[0]
                            new_perm_list.append(new_perm)
                        # else:
                        #     print("P_ITEM:", pid, "Pair:", p)
                        #     print("Skipped.")
                    pid += 1

                perm_list += new_perm_list
                #print(len(new_perm_list), len(perm_list))
            print("Stage %d, %d permuations."%(stage, len(perm_list)))
            permutations.append(perm_list)
        permutations = np.array(permutations)
        new_masks = []
        new_accs = []
        new_macs = []

        for mask_idx in tqdm(range(self.masks.shape[0])):
            local_new = [self.masks[mask_idx, :, :, :].copy(),]
            for stage in range(n_stages):
                stage_new = []
                for new_mask in local_new:
                    for perm in permutations[stage, np.random.choice(
                            permutations.shape[1], int(np.rint(permutations.shape[1] * srate))), :] :
                        l_mask = new_mask.copy()
                        permutation_matrix = np.zeros([perm.shape[0], perm.shape[0]])
                        permutation_matrix[np.arange(perm.shape[0]), perm] = 1
                        l_mask[stage, :, :] = np.matmul(l_mask[stage, :, :], permutation_matrix)
                        if np.sum(np.tril(l_mask[stage, :, :], -1)) == 0:
                            stage_new.append(l_mask)

                local_new += stage_new
                #print("Stage %d, %d new valid masks."%(stage, len(stage_new)))
            local_new = np.array(local_new)
            # raise NotImplementedError
            local_new = np.unique(local_new, axis=0).tolist()

            if mult is not None and len(local_new) > mult:
                local_new = np.array(local_new)
                local_new = local_new[np.random.choice(len(local_new), mult)]
                local_new = local_new.tolist()
            new_masks += local_new
            new_accs += [self.accs[mask_idx]] * len(local_new)
            new_macs += [self.macs[mask_idx]] * len(local_new)
            #print("%d new valid masks added." %(len(local_new)-1))

        print("Augmented dataset has %d records." % len(new_masks))
        self.masks, _uidx = np.unique(new_masks, return_index=True, axis=0)
        self.accs = np.array(new_accs)[_uidx]
        self.macs = np.array(new_macs)[_uidx]
        print("After eliminate duplicated masks, %d records generated." % len(self.masks))


    def inspect_nasbench(self):
        self.num_examples = len(self.accs)
        counter = 0
        for i in tqdm(range(self.num_examples)):
            for j in range(i):
                if self._is_equal_mask(self.masks[i], self.masks[j]):
                    counter += 1
                    print("Mask %d is identical to mask %d! Accuracy: %f vs. %f" %(i, j, self.accs[i], self.accs[j]))
        if counter != 0:
            raise ValueError("Check failed for NASbench! Please use self.cleanup_nasbench() before proceeding.")
        else:
            print("Inspection success! No data corruption/duplication in NASBench.")