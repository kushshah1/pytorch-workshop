# Execute with python 00_all_in_one_PyTorch.py

import os
import torch
import torch.nn as nn
import torch.optim as optim

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(5, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc2(self.fc1(x))

def main():
    # Specify a device (a GPU) to put model to
    device = torch.device('cuda')
    
    # Create a model
    model = ToyModel()

    # Move model to the device
    model.to(device)

    # Create an optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Create a loss function
    loss_fn = nn.MSELoss()

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    # Print optimizer's state_dict
    print("\n\nOptimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    # Print out parameters
    print("\n\nPARAMETERS BEFORE TRAINING")
    [print(x) for x in model.named_parameters()]

    # Training loop
    for epoch in range(5):
        
        # Create 20 data points, each is a length-5 vector
        inputs = torch.randn(20, 5).to(device)
        # Create 20 target values, each is a scalar
        labels = torch.randn(20, 1).to(device)

        # Zero the gradient of the optimizer
        optimizer.zero_grad()
        
        # Forward pass through the model
        outputs = model(inputs)

        # Calculate loss value
        loss = loss_fn(outputs, labels)

        # Backward pass - torch takes care of that graient update for you
        loss.backward()

        # Have the optimizer update parameters
        optimizer.step()

    # Print out parameters
    print("\n\n\nPARAMETERS AFTER TRAINING")
    [print(x) for x in model.named_parameters()]
    
    # Save the torch checkpoint
    torch.save(model.state_dict(), 'test.pt')

if __name__ == "__main__":
    main()
