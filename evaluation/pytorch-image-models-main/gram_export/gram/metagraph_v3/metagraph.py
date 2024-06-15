from multiprocessing import pool
import os
import pickle

from sklearn.model_selection import train_test_split

from gram_export.gram.metagraph_v3 import util_concat as util
from gram_export.gram.metagraph_v3 import ops_gen
from gram_export.gram.metagraph_v3.algorithm import *
from gram_export.gram.metagraph_v3.util_graph import random_connected_graph
from gram_export.gram.nasbench.dataset import PIXEL_LAT

from gram_export.gram.metagraph_v3.predictor import *

from gram_export.gram.metagraph_v3.utils import get_mask_pool, sample_mask_from_pool
from gram_export.gram.metagraph_v3.mac_predictor import MACEstimator

from gram_export.gram.metagraph_v3.algorithm import sat_max_edge_limit, sat_min_edge_limit


class MetaGraph_v3:
    def __init__(self, nodes=6, depth=3, p=1.0, protocol="v3"):
        self.nodes = nodes
        self.depth = depth
        self.edges = 0
        print("Initialize graph with {} nodes!".format(self.nodes))
        # Add node coefficient.
        raw_mask = np.zeros(shape=[nodes, nodes])
        nodes_queue = [0]
        for i in range(nodes):
            if not i in nodes_queue:
                continue
            for j in range(i + 1, nodes):
                if np.random.rand() <= p:
                    raw_mask[i][j] = 1
                    nodes_queue.append(j)
        self.mask = []
        for i in range(self.depth):
            self.mask.append(raw_mask.copy())

        self.mask = np.asarray(self.mask)
        # Generate weight matrix for graph progagation
        self.edge_connections = np.zeros_like(self.mask)
        if protocol == "v3":
            self.protocol = ops_gen.ops_gen_v3()
        elif protocol == "new":
            self.protocol = ops_gen.ops_gen_new()
        else:
            raise NotImplementedError
        self.ops_def = []
        for i in range(depth):
            ops_def_ = self.protocol.fixed_group_ops(self.nodes)
            self.ops_def.append(ops_def_)
        self.raw_mask = self.mask.copy()

    def add_to_proto(self, protoWriter, num_repeats=None, num_channels=None, strides=None,
                     activation='relu', use_bias=True,
                     bottleneck_factor=1, use_se=False, use_residual=True,
                     pool_style: str = "max-pool",
                     proj_dim_per_cell: bool = False):
        """
        Add to protoWriter
        :param protoWriter:
        :param replicate:
        :return:
        """
        ifstream = self.ofstream
        for i in range(self.depth):
            for k in range(num_repeats[i]):
                if k == 0:
                    strides_ = strides[i]
                else:
                    strides_ = 1
                scope = "DAG_%d/replica_%d" % (i, k)
                proto, ofstream = util.convert_mask_to_proto(self.mask[i],
                                                             self.ops_def[i],
                                                             ifstream,
                                                             num_channels=num_channels[i],
                                                             scope=scope,
                                                             strides=strides_,
                                                             use_bias=use_bias,
                                                             use_residual=use_residual,
                                                             activation=activation,
                                                             bottleneck_factor=bottleneck_factor,
                                                             use_se=(use_se),
                                                             pool_style=pool_style,
                                                             proj_dim_multiplier=2 if k == num_repeats[i]-1 else 1,
                                                             proj_dim_per_cell=proj_dim_per_cell)
                protoWriter.add(proto)
                ifstream = ofstream

        self.ofstream = ifstream

    def set_depth(self, new_depth):
        self.depth = new_depth

    def random_sampling(self, sampler=None, max_edges=9):
        if sampler is None:
            sampler_ = random_connected_graph
        else:
            sampler_ = sampler

        while 1:
            for i in range(self.depth):
                mask, _ = sampler_(
                    self.nodes, max_edges=max_edges)
                self.mask[i] = mask.copy()
            if sat_min_edge_limit(self.mask, 2) and sat_max_edge_limit(self.mask, max_edges):
                return

    def random_sampling_with_stage(self, idx, sampler=None, max_edges=9):
        if sampler is None:
            sampler_ = random_connected_graph
        else:
            sampler_ = sampler
        while 1:
            mask, nodes_queue = sampler_(self.nodes, max_edges=max_edges)
            self.mask[idx] = mask.copy()
            if sat_min_edge_limit(self.mask, 2) and sat_max_edge_limit(self.mask, max_edges):
                return
