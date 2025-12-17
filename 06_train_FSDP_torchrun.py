# 06_train_FSDP.py
# FSDP example, properly wrapped with torchrun
# Run with:
#   torchrun --nproc_per_node=2 06_train_FSDP_torchrun.py

import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)

#######################
# Distributed setup
#######################

def setup():
    """
    Initialize process group using env vars set by torchrun,
    and set the current CUDA device.
    """
    dist.init_process_group(backend="nccl")  # env:// by default
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()

#######################
# Model
#######################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

#######################
# Train / test
#######################

def train(args, model, device, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2, device=device)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f'Train Epoch: {epoch} \tLoss: {ddp_loss[0] / ddp_loss[1]:.6f}')

def test(model, device, rank, world_size, test_loader):
    model.eval()
    ddp_loss = torch.zeros(3, device=device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        acc = 100. * ddp_loss[1] / ddp_loss[2]
        print(
            f'Test set: Average loss: {test_loss:.4f}, '
            f'Accuracy: {int(ddp_loss[1])}/{int(ddp_loss[2])} ({acc:.2f}%)\n'
        )

#######################
# Main FSDP entry
#######################

def fsdp_main(args):
    rank, world_size, local_rank = setup()
    device = torch.device("cuda", local_rank)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Rank-0-only download to avoid race; then barrier
    if rank == 0:
        datasets.MNIST('../data', train=True, download=True, transform=transform)
        datasets.MNIST('../data', train=False, download=True, transform=transform)
    dist.barrier()

    dataset1 = datasets.MNIST('../data', train=True, download=False, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, download=False, transform=transform)

    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=5000
    )

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = Net().to(device)
    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        test(model, device, rank, world_size, test_loader)
        scheduler.step()
    init_end_event.record()

    dist.barrier() # halt until all processes have called this function (allowing rank 0 to finish)
    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(model)
        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")

    cleanup()

#######################
# Script entry point
#######################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST FSDP Example (torchrun)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    fsdp_main(args)