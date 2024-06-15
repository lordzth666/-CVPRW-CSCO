import profile
import numpy as np

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torchvision.datasets as datasets

from imagenet.train_imagenet import train, validate
from imagenet.image_folder import ImageFolder
from src.utils.architecture import Architecture
from ptflops.flops_counter import get_model_complexity_info
from imagenet.utils.train_utils import *
from imagenet.folder2lmdb import ImageFolderLMDB
from torch.utils.tensorboard import SummaryWriter
from imagenet.utils.cross_entropy_with_label_smoothing import LabelSmoothingCrossEntropy
from imagenet.ema import EMA

import time

class ImageNetEstimator:
	def __init__(self):
		pass

	def train_and_evaluate(
			self,
			config,
			max_model_mac: float = 999999.99,
			validate_last_only: bool = False,
			profile_only: bool = False
   ):
		print("=> Creating model from '{}':".format(config.prototxt))
		model = Architecture(config.prototxt).cuda()  # noqa: E999
		model_name = model.model_name
		print("Created model %s" % (model_name))

		flops, params = get_model_complexity_info(model, (3, config.image_size, config.image_size),
													as_strings=False, verbose=False,
													print_per_layer_stat=False, ignore_modules=[nn.ReLU, nn.BatchNorm2d])

		print("FLOPS: %.5f (M)" % (flops / 1e6))
		print("Params: %.5f (M)" % (params / 1e6))

		# Get latency.
		mean_lat, std_lat = self.get_latency_v2(model)
		print("Latency: {} +/- {} ms".format(
			mean_lat, std_lat))

		if profile_only:
			return {
				'params(M)': params / 1e6,
				'MACs(M)': flops / 1e6,
				"Mean_Lat(ms)": mean_lat,
				"Std_Lat(ms)": std_lat,
			}

		if max_model_mac is not None:
			mac_limit = max_model_mac * config.image_size * config.image_size / 224 / 224
			if flops / 1e6 > mac_limit:
				print("Warning: FLOPS exceeds the maximum flops! Skipping the training and evaluation...")
				return None

		# print(model)
		print(vars(config))

		model.apply(init_weights)
		# model = torch.compile(model)
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

		# Finally, use EMA to stablize training.
		if config.EMA != 0:
			print("Wrapping model with EMA and decay={}".format(config.EMA))
			model = EMA(model, decay=config.EMA)

		# define loss function (criterion) and optimizer
		criterion = LabelSmoothingCrossEntropy(config.label_smoothing).cuda()

		optimizer = get_optimizer_with_lr(model, config.lr, config.optimizer)

		lr_scheduler = get_lr_scheduler(optimizer, config.lr_schedule, config.epochs)

		# Build the dataset. Whether LMDB or not.
		if config.lmdb:
			traindir = os.path.join(config.data, 'train.lmdb')
			valdir = os.path.join(config.data, 'val.lmdb')
			train_transforms = build_transform(config.augmentation, training=True)
			train_dataset = ImageFolderLMDB(traindir, train_transforms)
			val_transforms = build_transform(config.augmentation, training=False)
			val_dataset = ImageFolderLMDB(valdir, val_transforms)
		else:
			traindir = os.path.join(config.data, 'train')
			valdir = os.path.join(config.data, 'val')
			train_transforms = build_transform(config.augmentation, training=True)
			train_dataset = ImageFolder(traindir, train_transforms, max_classes=config.num_classes,
										max_samples_ratio=config.downsample_ratio)
			val_transforms = build_transform(config.augmentation, training=False)
			val_dataset = ImageFolder(valdir, val_transforms, max_classes=config.num_classes,
										max_samples_ratio=config.downsample_ratio)

		if config.multiprocessing_distributed:
			train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
		else:
			train_sampler = None

		print("Number of workers: {}".format(config.workers))

		train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=config.train_batch_size, shuffle=(train_sampler is None),
			num_workers=config.workers, pin_memory=True, sampler=train_sampler,
			drop_last=True)

		val_loader = torch.utils.data.DataLoader(
			val_dataset, batch_size=config.val_batch_size, shuffle=False,
			num_workers=config.workers, pin_memory=True)

		best_acc1 = 0
		best_acc5 = 0
		best_loss = 99999

		lr_scheduler.step(config.start_epochs)

		for epoch in range(config.start_epochs, config.epochs):
			print("Epoch %d/%d" % (epoch, config.epochs))
			print("Learning rate: %s" % lr_scheduler.get_last_lr())
			if config.multiprocessing_distributed:
				train_sampler.set_epoch(epoch)
			# train for one epoch
			acc1, acc5, loss = train(train_loader, model, criterion, optimizer, epoch, config, warmup=True)
			tr_acc1, tr_acc5, tr_loss = acc1, acc5, loss
			# evaluate on validation set
			if not validate_last_only or (validate_last_only and epoch == config.epochs-1):
				acc1, acc5, loss = validate(val_loader, model, criterion, config)
				val_acc1, val_acc5, val_loss = acc1, acc5, loss

				# remember best acc@1 and save checkpoint
				best_acc1 = max(acc1, best_acc1)
				best_acc5 = max(acc5, best_acc5)
				best_loss = min(loss, best_loss)

			lr_scheduler.step()

		return {
			'best_acc1': best_acc1,
			'best_acc5': best_acc5,
			'best_loss': best_loss,
			'params(M)': params / 1e6,
			'MACs(M)': flops / 1e6,
			"Mean_Lat(ms)": mean_lat,
			"Std_Lat(ms)": std_lat,
		}


	def get_latency_v2(self, model, num_trials=100, warmup_steps=25, batch_size=128, image_size=224):
		model.eval()
		batch_time = []
		start_cuda_event = torch.cuda.Event(enable_timing=True,  blocking=True)
		end_cuda_event = torch.cuda.Event(enable_timing=True, blocking=True)

		torch_tensor = torch.ones(
			size=[batch_size, 3, image_size, image_size], requires_grad=False).cuda()
		torch.cuda.synchronize()
		with torch.no_grad():
			for i in range(num_trials):
				start_cuda_event.record()
				# compute output
				_ = model(torch_tensor)
				end_cuda_event.record()
				torch.cuda.synchronize()
				elasped_time = float(start_cuda_event.elapsed_time(end_cuda_event))
				if i > warmup_steps:
					batch_time.append(elasped_time)
		mean_time = np.mean(batch_time)
		std_time = np.std(batch_time)
		return mean_time, std_time

