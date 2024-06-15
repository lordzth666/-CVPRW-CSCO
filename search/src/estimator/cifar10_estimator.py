import os
import numpy as np
from ptflops.flops_counter import get_model_complexity_info

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets

from cifar10_utils.train_cifar10 import train, test
from imagenet.image_folder import ImageFolder
from src.utils.architecture import Architecture

from cifar10_utils.train_utils import *
from imagenet.folder2lmdb import ImageFolderLMDB
from imagenet.utils.cross_entropy_with_label_smoothing import LabelSmoothingCrossEntropy
from imagenet.ema import EMA
import torchvision
from cifar10_utils.cifar10_dataset import CIFAR10

class CIFAR10Estimator:
    def __init__(self):
        pass

    def train_and_evaluate(
            self, 
            config, 
            max_model_mac: float = 99999.999,
            validate_last_only: bool = False,
            profile_only: bool = False):
        print("=> Creating model from '{}':".format(config.prototxt))
        model = Architecture(config.prototxt).cuda()
        model_name = model.model_name
        print("Created model %s" % (model_name))

        flops, params = get_model_complexity_info(model, (3, config.image_size, config.image_size),
                                                  as_strings=False, verbose=False,
                                                  print_per_layer_stat=False, ignore_modules=[nn.ReLU, nn.BatchNorm2d])

        print("FLOPS: %.5f (M)" % (flops / 1e6))
        print("Params: %.5f (M)" % (params / 1e6))

        if profile_only:
            return {
                'params(M)': params / 1e6,
                'MACs(M)': flops / 1e6
            }

        if max_model_mac is not None:
            mac_limit = max_model_mac * config.image_size * config.image_size / 32 / 32
            if flops / 1e6 > mac_limit:
                print("Warning: FLOPS exceeds the maximum flops! Skipping the training and evaluation...")
                return None

        best_acc = 0  # best test accuracy
        best_loss = 99999.999

        # Data
        print('==> Preparing data..')
        transform_train = build_transform(True, config.augmentation)
        transform_test = build_transform(False, config.augmentation)

        train_dataset = CIFAR10(root='./data', train=True, download=True,
                                transform=transform_train, downsample_ratio=config.downsample_ratio)

        train_sampler = None

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size,
                                                  shuffle=not train_sampler,
                                                  num_workers=config.workers, sampler=train_sampler, pin_memory=True,
                                                  drop_last=True)

        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform_test,
                               downsample_ratio=config.downsample_ratio)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.val_batch_size,
                                                 shuffle=False, num_workers=config.workers, pin_memory=True)

        # Model
        print('==> Building model..')
        # model = Architecture("target/resmodel20-cifar-base.prototxt")
        model = Architecture(config.prototxt)
        model.apply(init_weights)
        print(model)
        model_root_path = model.save_path

        if config.multiprocessing_distributed:
            print("Use DDP...")
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()

        if config.EMA != 0:
            print("Wrapping model with EMA and decay={}".format(config.EMA))
            model = EMA(model, decay=config.EMA)

        model = model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer_with_lr(model, config.lr, config.optimizer)
        scheduler = get_lr_scheduler(optimizer, config.lr_schedule, config.epochs)

        for epoch in range(config.start_epochs, config.epochs):
            lr = scheduler.get_last_lr()
            print("Learning rate: %s" % lr)
            train(trainloader, model, criterion, optimizer, epoch, config, warmup=False)

            scheduler.step()

            if not validate_last_only or (validate_last_only and epoch == config.epochs-1):
                acc, loss = test(testloader, model, criterion, optimizer, epoch, config)
                if acc > best_acc:
                    best_acc = acc
                    best_loss = loss

        return {
            'best_acc1': best_acc,
            'best_loss': best_loss,
            'params(M)': params / 1e6,
            'MACs(M)': flops / 1e6
        }
