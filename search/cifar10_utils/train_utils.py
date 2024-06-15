import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from cifar10_utils.autoaugment import CIFAR10Policy
from cifar10_utils.cosine_with_warmup import CosineAnnealingLRWarmup
import math

from cifar10_utils.rmsprop_tf import RMSpropTF

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length, prob=1.0, n_holes=1):
        self.n_holes = n_holes
        self.length = length
        self.prob = prob

    def __call__(self, img):
        """
        Args:
            prob: probability of applying cutout.
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if np.random.random() > self.prob:
            return img

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

def get_optimizer_with_lr(model, lr, optimizer='sgd'):
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=0.9,
                                    weight_decay=0)
    elif optimizer == 'sgd++':  
        optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=0.875,
                                    weight_decay=0,
                                    nesterov=True)
    elif optimizer == 'rmsprop-tf':
        # This is in-consistent with TF-Rmsprop. eps should be tuned.
        optimizer = RMSpropTF(model.parameters(), lr, eps=0.001, momentum=.9)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.5, .999), eps=1e-8)
    else:
        raise NotImplementedError

    return optimizer

def get_lr_scheduler(optimizer, lr_scheduler='cosine', num_epochs=90):
    if lr_scheduler == "none":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [num_epochs*10, num_epochs*20])
    elif lr_scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(num_epochs * 0.5), int(num_epochs * .75), int(num_epochs * .90)],
                                                            gamma=0.1)
    elif lr_scheduler == "cosine":
        lr_scheduler = CosineAnnealingLRWarmup(optimizer, T_max=num_epochs, last_epoch=-1)
    elif lr_scheduler == "cosine-no-warmup":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, last_epoch=-1)
    else:
        raise NotImplementedError("Learning rate scheduler {} is not supported!".format(lr_scheduler))

    return lr_scheduler

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2434, 0.2616]

def build_transform(training=True, augmentation="normal-no-cutout"):
    if not training:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        return transform
    else:
        if augmentation == "normal-no-cutout":
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        elif augmentation == "normal":
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                Cutout(16),
            ])
        elif augmentation == 'autoaugment':
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                Cutout(16)
            ])
        else:
            raise NotImplementedError("Augmentation must be one of ['normal-no-cutout', 'normal', 'autoaugment'].")

        return transform

def get_l2_loss(model, weight_decay, regularize_depthwise: bool = False):
    assert isinstance(model, torch.nn.Module)
    l2_loss = []
    for m in model.parameters():
        if m.requires_grad and len(m.size()) > 1:
            if m.size(1) == 1 and regularize_depthwise:
                continue
            else:
                l2_loss.append(torch.square(torch.norm(m, 2)))
    l2_loss = torch.stack(l2_loss)
    return torch.sum(l2_loss) * weight_decay


from math import sqrt
def get_l1_loss(model, weight_decay):
    assert isinstance(model, torch.nn.Module)
    l1_loss = None
    for m in model.parameters():
        if m.requires_grad and len(m.size()) > 1:
            if l1_loss is None:
                l1_loss = torch.sum(torch.abs(m))
            else:
                l1_loss += torch.sum(torch.abs(m))
    l1_loss = torch.mul(l1_loss, torch.tensor(weight_decay))
    return l1_loss

def get_huber_loss(model, weight_decay):
    def _huber_loss(weight, delta):
        neg_masks = (torch.abs(weight) < delta).float()
        pos_masks = torch.ones_like(neg_masks) - neg_masks
        return torch.sum(neg_masks * torch.square(weight) + \
            pos_masks * 2 * delta * (torch.abs(weight) - 0.5 * delta))
    assert isinstance(model, torch.nn.Module)
    huber_loss = None
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            delta = 3 * torch.std(m.weight.data)
            if huber_loss is None:
                huber_loss = _huber_loss(m.weight, delta)
            else:
                huber_loss += _huber_loss(m.weight, delta)
    if _DEVICE_CFG:
        huber_loss = torch.mul(huber_loss, torch.tensor(weight_decay).cuda())
    else:
        huber_loss = torch.mul(huber_loss, torch.tensor(weight_decay))
    return huber_loss


from math import sqrt
def init_weights(m):
    if isinstance(m, nn.Linear):
        # fan_out = m.weight.size(0)  # fan-out
        init_range = 0.01
        m.weight.data.normal_(0, init_range)
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        # This fixes the TensorFlow prototxts, where Convolution layers are used to represent FC for efficiency.
        try:
            if m.dense_initializer:
                init_range = 0.01
                m.weight.data.normal_(0, init_range)
                m.bias.data.zero_()
            else:
                # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels / m.groups
                # m.weight.data.normal_(0, math.sqrt(1.0 / fan_out))
                torch.nn.init.kaiming_normal_(m.weight, a=sqrt(5.0))
                if m.bias is not None:
                    m.bias.data.zero_()
        except Exception:
            # Otherwise, weights are initialized as usual.
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels / m.groups
            m.weight.data.normal_(0, math.sqrt(1.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
