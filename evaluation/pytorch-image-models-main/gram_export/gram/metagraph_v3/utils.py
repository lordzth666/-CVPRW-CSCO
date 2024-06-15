import numpy as np


def get_hash_kernel(size):
    return np.random.normal(size=size)



def select_unique(masks, kernel):
    res = np.sum(masks * kernel, axis=(1,2))
    _, args = np.unique(res, return_index=True)
    return masks[args]

def get_mask_pool(sorted_candidate_masks, top_k=50):
    """
    Get the best 'top_k' masks for each stage in the candidate pool.
    :param sorted_candidate_masks: Sorted masks from high score to low score.
    :param top_k: 50
    :return: A [num_stage, ?, n, n] array indicating all of the candidate masks.
    """
    num_stages = np.shape(sorted_candidate_masks)[1]
    n = np.shape(sorted_candidate_masks)[2]

    hash_kernel = get_hash_kernel(size=[n, n])

    ret_masks = []
    for i in range(num_stages):
        masks = sorted_candidate_masks[:, i]
        unique_masks = select_unique(masks, hash_kernel)

        ret_masks.append(unique_masks[:top_k])

    return np.asarray(ret_masks)


def sample_mask_from_pool(sorted_candidate_masks, stage=0):
    arg = np.random.choice(len(sorted_candidate_masks[stage]))
    return sorted_candidate_masks[stage][arg]