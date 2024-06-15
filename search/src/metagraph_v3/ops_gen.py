
import numpy as np

class ops_gen_v3:
    def __init__(self):
            self.ops_def = np.asarray([{'type': 'SeparableConv', 'filters': 16, 'kernel_size': 3, 'strides': 1},
                                       {'type': 'SeparableConv', 'filters': 16, 'kernel_size': 5, 'strides': 1},
                                       {'type': 'SeparableConv', 'filters': 16, 'kernel_size': 7, 'strides': 1},
                                       {'type': 'Convolutional', 'filters': 16, 'kernel_size': 1, 'strides': 1},
                                       {'type': 'Convolutional', 'filters': 16, 'kernel_size': 3, 'strides': 1},
                                       {'type': 'Convolutional', 'filters': 16, 'kernel_size': 5, 'strides': 1}])

    def seed_ops(self):
        id = np.random.randint(0, len(self.ops_def))
        print(id)
        return self.ops_def[id]

    def seed_group_ops(self, n):
        """
        Seed a group of ops
        :param n:
        :return:
        """
        rand_list = []
        m = len(self.ops_def)
        for i in range(n):
            rand_list.append(i % m)
        np.random.shuffle(rand_list)
        rand_list = np.asarray(rand_list)
        print(self.ops_def[rand_list])
        return self.ops_def[rand_list]

    def fixed_group_ops(self, n):
        """
        Seed a group of ops
        :param n:
        :return:
        """
        rand_list = []
        m = len(self.ops_def)
        for i in range(n):
            rand_list.append(i % m)
        rand_list = np.asarray(rand_list)
        print(self.ops_def[rand_list])
        return self.ops_def[rand_list]

class ops_gen_new:
    def __init__(self):
        self.ops_def = np.asarray([{'type': 'DepthwiseConv', 'filters': 16, 'kernel_size': 3, 'strides': 1},
                                    {'type': 'DepthwiseConv', 'filters': 16, 'kernel_size': 5, 'strides': 1},
                                    {'type': 'DepthwiseConv', 'filters': 16, 'kernel_size': 7, 'strides': 1},
                                    {'type': 'Convolutional', 'filters': 16, 'kernel_size': 1, 'strides': 1},
                                    ])

    def seed_ops(self):
        id = np.random.randint(0, len(self.ops_def))
        print(id)
        return self.ops_def[id]

    def seed_group_ops(self, n):
        """
        Seed a group of ops
        :param n:
        :return:
        """
        rand_list = []
        m = len(self.ops_def)
        for i in range(n):
            rand_list.append(i % m)
        np.random.shuffle(rand_list)
        rand_list = np.asarray(rand_list)
        print(self.ops_def[rand_list])
        return self.ops_def[rand_list]

    def fixed_group_ops(self, n):
        """
        Seed a group of ops
        :param n:
        :return:
        """
        rand_list = []
        m = len(self.ops_def)
        for i in range(n):
            rand_list.append(i % m)
        rand_list = np.asarray(rand_list)
        print(self.ops_def[rand_list])
        return self.ops_def[rand_list]
