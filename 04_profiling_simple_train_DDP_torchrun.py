# Profiling script 
# Run with `nsys profile -t cuda,nvtx torchrun --nnodes 1 --nproc_per_node 2 04_profiling.py`

import os
import torch
from socket import gethostname
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.cuda.nvtx as nvtx
import torch.cuda.profiler as profiler

class ToyDataset(Dataset):
    def __init__(self, inputs, labels):
        assert len(inputs) == len(labels)
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (self.inputs[idx], self.labels[idx])

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(8192, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        with nvtx.range(f"fc1"):
            x = self.fc1(x)
        with nvtx.range(f"fc2"):
            x = self.fc2(x)
        with nvtx.range(f"fc3"):
            x = self.fc3(x)
        return x

def train(rank, world_size, model, train_loader, loss_fn, optimizer, epoch):
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        with torch.autograd.profiler.emit_nvtx():
            profiler.start()
            nvtx.range_push(f"Step {epoch}-{batch_idx}")
            with nvtx.range(f"Forward {epoch}-{batch_idx}"):
                outputs = model(inputs)
            with nvtx.range(f"Loss func {epoch}-{batch_idx}"):
                loss = loss_fn(outputs, labels)
            with nvtx.range(f"Backward {epoch}-{batch_idx}"):
                loss.backward()
            with nvtx.range(f"Optimizer {epoch}-{batch_idx}"):
                optimizer.step()
            nvtx.range_pop()
            profiler.stop()
        if rank == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    print(f"Hello from rank {rank} of {world_size} on {gethostname()}")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if rank == 0:
        print(f"Group initialized? {dist.is_initialized()}", flush=True)
    
    device = torch.device('cuda')
    model = ToyModel().to(device)
    model = DDP(model, device_ids=[device])

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    inputs = torch.randn(20000, 8192).to(device)
    labels = torch.randn(20000, 4).to(device)
    dataset = ToyDataset(inputs, labels)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(dataset, batch_size=1000, sampler=sampler)
    
    for epoch in range(10):
        train(rank, world_size, model, train_loader, loss_fn, optimizer, epoch)




    dist.destroy_process_group()

if __name__ == "__main__":
    main()
