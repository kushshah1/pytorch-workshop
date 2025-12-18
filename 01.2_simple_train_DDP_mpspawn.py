# 01.2_simple_train_DDP_mpspawn.py
# Same simple training script, mp.spawn version
# Changes from torchrun --> mp.spawn noted with comments
# Run with:
#   python 01.2_simple_train_DDP_mpspawn.py

import os
import torch
from socket import gethostname
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp   ### CHANGED: import mp for mp.spawn

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc2(self.fc1(x))

class ToyDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def main(rank, world_size):   ### CHANGED: mp.spawn passes rank, world_size via args
    # rank = int(os.environ["LOCAL_RANK"])   ### CHANGED: no torchrun env vars; use spawn rank
    # world_size = int(os.environ["WORLD_SIZE"])  ### CHANGED: world_size comes from argument

    # Minimal env setup for single-node spawn run
    os.environ['MASTER_ADDR'] = 'localhost' ### CHANGED: set master addr for init
    os.environ['MASTER_PORT'] = '12355' ### CHANGED: set master port for init

    print(f"Hello from rank {rank} of {world_size} on {gethostname()}")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device('cuda')

    if rank == 0:
        print(f"Group initialized? {dist.is_initialized()}", flush=True)
    
    model = ToyModel().to(device)
    ddp_model = DDP(model, device_ids=[device])

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    if rank == 0:
        print("Entering training loop")
        
    inputs = torch.randn(250, 10).to(device)
    labels = torch.randn(250, 5).to(device)

    dataset = ToyDataset(inputs, labels)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=40, sampler=sampler, shuffle=False)
    
    for epoch in range(4):
        sampler.set_epoch(epoch)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Batch size: {inputs.shape}, Loss: {loss.item()}")
    if rank == 0:
        print("Training is done!")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()   ### CHANGED: determine world_size from GPU count
    mp.spawn(                              ### CHANGED: launch processes via mp.spawn
        main,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )