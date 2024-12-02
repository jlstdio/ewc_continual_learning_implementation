import json
import torch
from torch import nn


# Load Fisher information from a JSON file
def load_fisher_from_json(file_path, device):
    with open(file_path, 'r') as f:
        fisher_json = json.load(f)
    fisher = {name: torch.tensor(param, device=device) for name, param in fisher_json.items()}
    return fisher


# Save Fisher information to a JSON file
def save_fisher_to_json(fisher, file_path):
    fisher_json = {name: param.tolist() for name, param in fisher.items()}
    with open(file_path, 'w') as f:
        json.dump(fisher_json, f)


def visualize_fisher(fisher, file_path):
    pass


# Compute Fisher information for a client
def compute_fisher(model, dataloader, costFunc, device):
    fisher = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
    model.to(device)
    model.eval()

    if costFunc == 'CEloss':
        criterion = nn.CrossEntropyLoss()
    elif costFunc == 'BCEloss':
        criterion = nn.BCELoss()
    elif costFunc == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()


    for inputs, targets in dataloader:
        inputs = inputs.to(device)

        if costFunc == 'CEloss':
            targets = targets.long().to(device)  # CE
        elif costFunc == 'BCEloss':
            targets = targets.to(device)  # BCE
        elif costFunc == 'BCEWithLogitsLoss':
            targets = targets.long().to(device)  # CE

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.pow(2)

    # Average Fisher information across batches
    fisher = {name: value / len(dataloader) for name, value in fisher.items()}
    return fisher