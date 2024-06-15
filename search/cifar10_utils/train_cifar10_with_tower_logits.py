import os, sys

sys.path.append(os.getcwd())
import argparse
import time

import numpy as np

import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from torch.cuda.amp import autocast

# Suppress Titan XP errors
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from cifar10_utils.train_utils import *
from cifar10_utils.ema import EMA
from cifar10_utils.cifar10_dataset import CIFAR10
from torch.utils.tensorboard import SummaryWriter
from src.yaml_utils.yaml_parser import load_and_apply_yaml_config
from src.utils.model_builder import build_model_with_yaml_config
from ptflops import get_model_complexity_info


global global_step_counter
global_step_counter = 0

TOWER_NAME = "DAG_2/replica_0/node_0"

class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU()
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0),-1))
        return x


# Training
def train(
    trainloader, model, criterion, 
    optimizer, epoch, config, 
    warmup=True, writer=None,
    auxillary_weight: float = 1.0,
    ):
    global global_step_counter
    model.train()
    train_loss = 0.
    aux_loss = 0.
    correct = 0
    total = 0
    scaler = torch.cuda.amp.GradScaler()
    print("Epoch %d: " %epoch)
    num_warmup_train_batches = 5 * len(trainloader)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if epoch < 5 and config.lr_schedule != "cosine-no-warmup" and warmup:
            wm_lr = global_step_counter * config.lr / num_warmup_train_batches
            for g in optimizer.param_groups:
                g['lr'] = wm_lr

        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        start = time.time()
        with autocast(enabled=(config.amp == 1)):
            #NOTE: logits should be the last output.
            mid_layer_output, logit = model.forward_with_names(inputs, [TOWER_NAME, "logits"])
            mid_layer_logit = model.aux_head(mid_layer_output)
            loss = criterion(logit, targets)
            loss_mid_layer = criterion(mid_layer_logit, targets)
            l2_loss = get_l2_loss(model, config.weight_decay)
            total_loss = loss + l2_loss + loss_mid_layer * auxillary_weight

        if config.amp == 1:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
        else:
            total_loss.backward()

        # Clip gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        if config.amp == 1:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Update EMA weights if possible.
        if config.EMA != 0:
            model.update()

        end = time.time()
        train_loss += loss.item()
        aux_loss += loss_mid_layer.item()
        _, predicted = logit.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        global_step_counter += 1

        if batch_idx % config.report_freq == 0:
            train_loss_avg = train_loss / (batch_idx+1)
            aux_loss_avg = aux_loss / (batch_idx+1)
            train_acc = 100.*correct / total
            print("Loss: {:.3f} | L2_loss: {:.3f} | Time: {:.3f} | Acc: {:.3f}".format(
                train_loss_avg, l2_loss.item(), end-start, train_acc))
            if writer is not None:
                writer.add_scalar("Loss/train", train_loss_avg, global_step_counter)
                writer.add_scalar("Loss/Aux_train", aux_loss_avg, global_step_counter)
                writer.add_scalar("Acc/train", train_acc, global_step_counter)
                writer.add_scalar("L2 Loss/train", l2_loss.item(), global_step_counter)

def test(testloader, model, criterion, optimizer, epoch, config, writer=None):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % config.report_freq == 0:
                print("Loss: {:.3f} | Acc: {:.3f}".format(test_loss / (batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    loss = test_loss / batch_idx
    print("Test Loss: {:.3f}, Test Acc: {:.3f}".format(loss, acc))
    
    if writer is not None:
        writer.add_scalar("Loss/test", loss, epoch)
        writer.add_scalar("Acc/test", acc, epoch)

    return acc, loss

def main(args):
    print(vars(args))
    config = load_and_apply_yaml_config(args.yaml_cfg)
    if args.prototxt is not None:
        config.prototxt = args.prototxt
    print(config)

    best_acc = 0  # best test accuracy

    # Data
    print('==> Preparing data..')
    transform_train = build_transform(True, config.augmentation)
    transform_test = build_transform(False, config.augmentation)

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    train_sampler = None

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=not train_sampler,
                                               num_workers=config.workers, sampler=train_sampler, pin_memory=True,
                                               drop_last=True)

    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.val_batch_size,
                                             shuffle=False, num_workers=config.workers, pin_memory=True)
    # Model
    print('==> Building model..')
    # The old way of parsing prototxts has been deprecated.
    model = build_model_with_yaml_config(config)
    # Override model name.
    if args.model_name is not None:
        print("Replacing model name...")
        model.model_name = args.model_name
        model.save_path = os.path.join("models", args.model_name)

    model.apply(init_weights)

    model_root_path = model.save_path
    tb_path = os.path.join(model_root_path, "training_log")
    writer = SummaryWriter(tb_path)
    rand_input = torch.rand((1, 3, 32, 32))
    writer.add_graph(model, rand_input)


    flops, params = get_model_complexity_info(model, (3, config.image_size, config.image_size),
                                              as_strings=False, verbose=False,
                                              print_per_layer_stat=False, ignore_modules=[nn.ReLU, nn.BatchNorm2d])
    print("FLOPS: %.5f (M)" % (flops / 1e6))
    print("Params: %.5f (M)" % (params / 1e6))
    
    # Get auxillary head num counts.
    with torch.no_grad():
        output_aux_head, _ = model.forward_with_names(rand_input, [TOWER_NAME, "logits"])
        num_aux_head = output_aux_head.size(1)
    
    # Adding auxillary heads.
    model.aux_head = AuxiliaryHeadCIFAR(num_aux_head, 10)
    model.aux_head.apply(init_weights)
    # Reset bn momentum and epsilon.
    for m in model.aux_head.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = float(config.bn_momentum)
            m.eps = float(config.bn_epsilon)
        
    model = model.cuda()
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer_with_lr(model, config.lr, config.optimizer)
    scheduler = get_lr_scheduler(optimizer, config.lr_schedule, config.epochs)
    base_drop_connect_rate = float(config.drop_connect)

    if config.EMA != 0:
        print("Wrapping model with EMA and decay={}".format(config.EMA))
        model = EMA(model, decay=config.EMA)
    # Check if best checkpoint exists, or evaluate force the weight loading.
    best_model_path = os.path.join(model_root_path, "checkpoint.pth")
    if os.path.exists(best_model_path) or args.evaluate:
        print("Loading checkpoint...")
        if isinstance(model, EMA):
            checkpoint = torch.load(best_model_path)
            # Both model and shadow load the same weights.
            model.model.load_state_dict(checkpoint['model'])
            model.shadow.load_state_dict(checkpoint['model'])
        else:
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model'])
        print("Done!")
        if args.evaluate:
            acc, _ = test(testloader, model, criterion, optimizer, -1, config, writer=writer)
            print("Accuracy: {}".format(acc))
            exit(0)

    for epoch in range(config.start_epochs, config.epochs):
        lr = scheduler.get_last_lr()
        writer.add_scalar("Hparams/LR", lr[0], epoch)
        print("Learning rate: %s" %lr)
        # Drop_path is changed accordingly.
        current_drop_connect_rate = base_drop_connect_rate * epoch / config.epochs
        model.set_drop_connect_rate((current_drop_connect_rate))
        train(trainloader, model, criterion, optimizer, epoch, config, writer=writer,
              auxillary_weight=args.auxillary_weight)
        scheduler.step()

        acc, _ = test(testloader, model, criterion, optimizer, epoch, config, writer=writer)
        if acc > best_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(model_root_path, "checkpoint.pth"))
            best_acc = acc

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument("--yaml_cfg", type=str, default=None,
                        help="YAML config for training.")
    parser.add_argument("--prototxt", type=str, default=None,
                        help="Prototxt. If specified, overrides the original prototxt in file.")
    parser.add_argument("--evaluate", action='store_true', default=False,
                        help="Whether doing evaluation or not.")
    parser.add_argument("--model_name", type=str, default=None, 
                        help="Overrides model name in CIFAR-10 config. Used for HPO.")
    parser.add_argument("--auxillary_weight", type=float, default=0.4, 
                        help="Weight to auxillary tower.")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="True to enable torch.compie()")
    args = parser.parse_args()

    main(args)
