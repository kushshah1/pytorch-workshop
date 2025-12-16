# [SOLUTION] In this script, we perform embedding of ESM models using DDP. 
# Run with `torchrun --nnodes 1 --nproc_per_node 2 03_quiz_DDP_ESM_inference_solution.py`


import os
import torch
from socket import gethostname
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import esm
import numpy as np
# TODO: Import some of the most important classes

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

def get_embedding(rank, world_size, model, sample_loader):
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(sample_loader):
        inputs, targets = inputs.to(rank), targets.to(rank)
        
        outputs = model(inputs, repr_layers=[model.module.num_layers])
        
        # Extract the logits from the output dict
        logits = outputs["logits"]
        representations = outputs["representations"][model.module.num_layers]
        embeddings = representations[:, 1:-2].mean(1)
        # print(outputs)
        print(f"Rank {rank}: {logits.shape}, {representations.shape}, {embeddings.shape}")

def main(): 
    rank = int(os.environ["LOCAL_RANK"]) 
    world_size = int(os.environ["WORLD_SIZE"])
    
    print(f"Hello from rank {rank} of {world_size} on {gethostname()}")
    
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Load ESM-2 8M model
    print("Downloading 8M model ...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    
    model = model.to(rank)
    
    # Wrap the model with DDP
    # TODO: This clearly isn't right. FIXME!
    ddp_model = model
  
    # Prepare dummy inputs (protein sequence embeddings)
    batch_converter = alphabet.get_batch_converter()
    data = gen_data(100)
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    inputs = [batch_converter([(label, seq)])[2].squeeze(0) for label, seq in data]
    targets = torch.randint(0, 10, (len(data),))  # Dummy target classes
    
    # Create a Dataset and DataLoader with DistributedSampler
    dataset = ProteinDataset(list(zip(inputs, targets)))
    
    # TODO: Place a sampler here for DDP, remember to import the class as well
    sampler = 'How do I make a sampler for DDP?'
    print(sampler) 
    sample_loader = DataLoader(dataset, batch_size=40, sampler=sampler)
    
    # Inference
    get_embedding(rank, world_size, model, sample_loader)
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    
