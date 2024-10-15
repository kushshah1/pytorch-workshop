# In this script, we perform embedding of ESM models using DDP. 
# Run with `torchrun --nnodes 1 --nproc-per-node 2 05_quiz_profiling_ESM_downstream.py`
# There's no correct way of doing benchmarking, just useful ways.
# For possible solution see the solution script

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
import torch.cuda.nvtx as nvtx
import torch.cuda.profiler as profiler
import time

class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dummy_data_preprocessing()
        return self.data[idx]

def dummy_data_preprocessing():
    # Simulates intensive data processing
    arr = torch.randn(20, 5) 
    for _ in range(5000):
        arr = arr ** (1+1e-8)

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
        self.esm_model.eval()

        #### TODO ####: After setting up nvtx, 
        #### enable this part to see the effect of not backpassing through ESM
        #### In reality, whether to freeze ESM is entirely your research call
        #### ESM repo does recommend using this as a frozen embedder

        ## Freeze ESM model. 
        #for param in self.esm_model.parameters():
        #    param.requires_grad = False

        #### END TODO ####

        # One layer, connect from embeddings to 10 classes
        self.downstream = nn.Linear(self.esm_model.embed_dim, 10) 

    def forward(self, x):
        #### TODO ####: Put in some nvtx annotations (for hints see below)
        #### One for each line would be nice

        outputs = self.esm_model(x, repr_layers=[self.esm_model.num_layers])
        representations = outputs["representations"][self.esm_model.num_layers]
        embeddings = representations[:, 1:-2].mean(1)
        logits = self.downstream(embeddings)

        #### END TODO ####

        return logits

def train(rank, world_size, model, train_loader, criterion, optimizer, epoch):
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs, targets = inputs.to(rank), targets.to(rank)

        optimizer.zero_grad()

        with nvtx.range(f"Forward"):
            logits = model(inputs)

        with nvtx.range(f"Calc Loss"):
            loss = criterion(logits, targets)

        with nvtx.range(f"Backward"):
            loss.backward()

        with nvtx.range(f"Update Params"):
            optimizer.step()


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

    # Wrap the model with DDP
    ddp_model = DDP(model, device_ids=[rank])

    # Prepare dummy inputs (protein sequence embeddings)
    batch_converter = alphabet.get_batch_converter()
    data = gen_data(500)
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    inputs = [batch_converter([(label, seq)])[2].squeeze(0) for label, seq in data]
    targets = torch.randint(0, 10, (len(data),))  # Dummy target classes

    # Create a Dataset and DataLoader with DistributedSampler
    dataset = ProteinDataset(list(zip(inputs, targets)))

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=200, sampler=sampler, num_worker=0)

    #### TODO ####: After setting up nvtx,
    #### Try this different setting for num_workers in DataLoader
    #### Observe the differnce in per-epoch time

    #train_loader = DataLoader(dataset, batch_size=20, sampler=sampler, num_worker=10, pin_memory=True)

    ### END TODO ####

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(5):
        if rank == 0:
            print(f'Epoch {epoch}')
        sampler.set_epoch(epoch)
        with nvtx.range(f"Epoch"):
            train(rank, world_size, model, train_loader, criterion, optimizer, epoch)

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
