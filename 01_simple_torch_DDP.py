# Run this script with `torchrun --nnodes 1 --nproc-per-node 2 01_simple_torch_DDP.py` 

import os
import torch
from socket import gethostname
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc2(self.fc1(x))

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
    ddp_model = DDP(model, device_ids=[device])

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(5):
        inputs = torch.randn(20, 10).to(device)
        labels = torch.randn(20, 5).to(device)

        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
