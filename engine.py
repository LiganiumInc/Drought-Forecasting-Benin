import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
import pandas as pd
import numpy as np
import os
import tqdm

def train_step(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Train the model for one epoch.
    
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to train the model on (CPU or GPU).
    
    Returns:
        float: Training loss.
    """
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def test_step(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Evaluate the model on the test data.
    
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the test data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate the model on (CPU or GPU).
    
    Returns:
        float: Test loss.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    return running_loss / len(dataloader)

def engine(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, num_epochs: int) -> Dict[str, List[float]]:
    """Train and evaluate the model.
    
    Args:
        model (nn.Module): The model to train and evaluate.
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the test data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to train and evaluate the model on (CPU or GPU).
        num_epochs (int): Number of epochs to train the model.
    
    Returns:
        Dict[str, List[float]]: Dictionary containing training and test losses for each epoch.
    """
    history = {'train_loss': [], 'test_loss': []}
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        train_loss = train_step(model, train_loader, criterion, optimizer, device)
        test_loss = test_step(model, test_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    return history
