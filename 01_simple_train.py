# Execute with python 00_all_in_one_PyTorch_script.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(5, 3)
        self.fc2 = nn.Linear(3, 1)

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

def main():
    device = torch.device('cuda') # Specify a device (a GPU) to put model to
    model = ToyModel() # Create a model
    model.to(device) # Move model to the device
    optimizer = optim.SGD(model.parameters(), lr=0.01) # Create an optimizer
    loss_fn = nn.MSELoss() # Create a loss function

    print("MODEL'S STATE_DICT:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    print("\n\nOPTIMIZER'S STATE_DICT:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    print("\n\nPARAMETERS BEFORE TRAINING")
    [print(x) for x in model.named_parameters()]
    print("\n")

    inputs = torch.randn(20, 5).to(device) # Create 20 data points, each is a length-5 vector
    labels = torch.randn(20, 1).to(device) # Create 20 target values, each is a scalar

    dataset = ToyDataset(inputs, labels)
    train_loader = DataLoader(dataset, batch_size = 4, shuffle = True)

    # Training loop
    for epoch in range(4):
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Optional: move to device if dataset is not already on device
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            optimizer.zero_grad() # Zero the gradient of the optimizer
            outputs = model(inputs) # Forward pass through the model
            loss = loss_fn(outputs, labels) # Calculate loss value
            loss.backward() # Backward pass - torch takes care of that gradient update for you
            optimizer.step() # Have the optimizer update parameters
            print(f"Epoch {epoch}, Batch {batch_idx}, Batch size: {inputs.shape}, Loss: {loss.item()}")

    print("\n\nPARAMETERS AFTER TRAINING")
    [print(x) for x in model.named_parameters()]
    
    torch.save(model.state_dict(), 'test.pt') # Save the torch checkpoint

if __name__ == "__main__":
    main()
