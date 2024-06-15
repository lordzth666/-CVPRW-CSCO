import numpy as np


def create_mapping_using_ops_def(ops_def, num_filters):
    def _get_budget(op, num_filters):
        if op['type'] == "SeparableConv":
            ksize = op['kernel_size']
            return 1 + ksize * ksize / num_filters
        elif op['type'] == "Convolutional":
            ksize = op['kernel_size']
            return ksize * ksize

    mapping = np.zeros(len(ops_def))
    for i in range(len(ops_def)):
        mapping[i] = _get_budget(ops_def[i], num_filters)
    return mapping




class MACEstimator:
    """
    We need a one-shot MAC predictor here. One-shot methods are quite unstable.
    """
    def __init__(self, ops_def, replicate, scale, pool, input_size, init_filters, n, depth, width):
        self.ops_def = ops_def
        self.replicate = replicate
        self.scale = scale
        self.pool_size = pool
        self.input_size = input_size
        self.mac_cost = None
        self.n = n
        self.depth = depth
        self.width = width
        self.init_filters = init_filters
        self._create_logistics()

    def _create_logistics(self):
        assert self.width == 1, "Width of more than 1 is not supported."
        num_stages = len(self.replicate)
        input_size = self.input_size
        self.mapping = np.zeros(shape=(self.depth, self.width, self.n, self.n), dtype=np.float32)
        for i in range(self.depth):
            if self.pool_size[i] != 1:
                input_size = input_size // self.pool_size[i]
            for j in range(self.width):
                base_filters = self.init_filters * self.scale[i]
                mapping_ = create_mapping_using_ops_def(self.ops_def[i][j], num_filters=base_filters)
                mapping_ = np.array([mapping_] * self.n)
                # Now, scale up for real setting.
                mapping_ = mapping_ * input_size * input_size * base_filters * base_filters * (self.replicate[i] + 0.5) / 1e6
                self.mapping[i,j] = mapping_.copy()

        print("Finalized mapping!")
        print(self.mapping)
        print(self.mapping.shape)

    def estimate_mask(self, mask):
        """
        :param mask: (batch_size, num_stages, n, n)
        :return:
        """
        assert isinstance(mask, np.ndarray)
        if mask.ndim < 4:
            print("Mask should have ndim of at least 4.")
            mask_ = mask
        elif mask.ndim == 4:
            mask_ = np.expand_dims(mask, axis=2)
            ret = np.sum(mask_ * self.mapping, axis=(1,2,3,4))
            return ret
        else:
            raise NotImplementedError

