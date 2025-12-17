# 02.1_ESM_train_DDP_torchrun_multinode.py
# In this script, we perform embedding of ESM models using DDP. 
# Run with:
#   torchrun --nnodes 2 --nproc_per_node 2 02.1_ESM_train_DDP_torchrun_multinode.py

import os
import torch
from socket import gethostname
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torch.optim as optim
import esm
import numpy as np

class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def gen_data(num):
    def rand_seq(length):
        return ''.join(np.random.choice([x for x in 'ACDEFGHIKLMNPQRSTVWY'], length, replace=True))
    data = []
    for i in range(num):
        data.append((i, rand_seq(100)))
    return data

class DownstreamFromESM(nn.Module):
    def __init__(self, esm_model):
        super(DownstreamFromESM, self).__init__()
        self.esm_model = esm_model

        # Freeze entire esm_model
        self.esm_model.eval()
        for param in self.esm_model.parameters():
            param.requires_grad = False

        # One layer, connect from embeddings to 10 classes
        self.downstream = nn.Linear(self.esm_model.embed_dim, 10) 
        
    def forward(self, x):
        outputs = self.esm_model(x, repr_layers=[self.esm_model.num_layers])
        representations = outputs["representations"][self.esm_model.num_layers]
        embeddings = representations[:, 1:-2].mean(1)
        return self.downstream(embeddings)

def train(local_rank, rank, model, train_loader, criterion, optimizer, epoch):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(local_rank), targets.to(local_rank)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        if rank == 0 and batch_idx % 10 == 0:
            print(f"[Epoch {epoch} | Batch {batch_idx}] Loss = {loss.item():.4f}")

def main(): 
    # torchrun provides these automatically
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])              # global rank
    world_size = int(os.environ["WORLD_SIZE"])  # total processes
    
    print(f"Hello from global rank {rank}, local rank {local_rank} of {world_size} on {gethostname()}")
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    
    # Load ESM-2 8M model
    if rank == 0:
        print("Downloading 8M model ...")
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    
    model = DownstreamFromESM(esm_model).to(local_rank)
    
    # Wrap the model with DDP
    ddp_model = DDP(model, device_ids=[local_rank])
  
    # Prepare dummy inputs (protein sequence embeddings)
    batch_converter = alphabet.get_batch_converter()
    data = gen_data(500)
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    inputs = [batch_converter([(label, seq)])[2].squeeze(0) for label, seq in data]
    targets = torch.randint(0, 10, (len(data),))  # Dummy target classes
    
    # Create a Dataset and DataLoader with DistributedSampler
    dataset = ProteinDataset(list(zip(inputs, targets)))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=200, sampler=sampler)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(3):
        sampler.set_epoch(epoch)
        train(local_rank, rank, ddp_model, train_loader, criterion, optimizer, epoch)
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()