# 02_ESM_train_DDP_torchrun.py
# In this script, we perform embedding of ESM models using DDP. 
# Run with:
#   torchrun --nnodes 1 --nproc_per_node 2 02_ESM_train_DDP_torchrun.py

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

def gen_data(num): # Generates length-n list: [(index, 100-residue sequence), ...]
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

    # Forward pass outputs class logits
    def forward(self, x):
        outputs = self.esm_model(x, repr_layers=[self.esm_model.num_layers])
        representations = outputs["representations"][self.esm_model.num_layers]
        embeddings = representations[:, 1:-2].mean(1)
        class_logits = self.downstream(embeddings)
        return class_logits

def train(rank, world_size, model, train_loader, criterion, optimizer, epoch):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(rank), targets.to(rank)
        print(f"Rank {rank} input size: {inputs.shape}")
        optimizer.zero_grad()
        logits = model(inputs) # fwd pass
        loss = criterion(logits, targets) # calc loss
        loss.backward() # calc gradients
        optimizer.step() # update params
        if rank == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

def main(): 
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    print(f"Hello from rank {rank} of {world_size} on {gethostname()}")
    
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Load ESM-2 8M model
    print("Downloading 8M model ...")
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = DownstreamFromESM(esm_model).to(rank)
    batch_converter = alphabet.get_batch_converter() # utility to go from seq to token ID
    
    # Wrap the model with DDP
    ddp_model = DDP(model, device_ids=[rank])
  
    # Prepare (inputs, targets): Needs to be in form of per-sequence (token IDs, label)
    data = gen_data(500)
    inputs = [batch_converter([(label, seq)])[2].squeeze(0) for label, seq in data]
    targets = torch.randint(0, 10, (len(data),))  # Dummy target classes
    
    # Create a Dataset and DataLoader with DistributedSampler
    dataset = ProteinDataset(list(zip(inputs, targets)))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=200, sampler=sampler)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    # Training
    for epoch in range(3):
        sampler.set_epoch(epoch)
        train(rank, world_size, ddp_model, train_loader, criterion, optimizer, epoch)
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

