import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from models import BackpropNetwork
from utils import load_mnist_data, plot_training_curves


def train_backprop_baseline(epochs=20, batch_size=128, hidden_dims=[2000, 2000, 2000, 2000],
                           learning_rate=0.01, permutation_invariant=True, use_cuda=torch.cuda.is_available(),
                           verbose=True):
    """
    Train a baseline neural network on MNIST using backpropagation as described in Section 3.1.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        hidden_dims: List of hidden layer dimensions
        learning_rate: Learning rate for optimizer
        permutation_invariant: Whether to use permutation-invariant MNIST
        use_cuda: Whether to use CUDA
        verbose: Whether to print progress
        
    Returns:
        model: Trained model
        train_losses: List of training losses per epoch
        val_accuracies: List of validation accuracies per epoch
        test_accuracy: Final test accuracy
    """
    # Set device
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader = load_mnist_data(batch_size, use_cuda, permutation_invariant)
    
    # Calculate input dimension based on permutation_invariant setting
    input_dim = 784 if permutation_invariant else (28 * 28)
    
    # Create model
    model = BackpropNetwork(input_dim, hidden_dims).to(device)
    
    # Define optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    train_losses = []
    val_accuracies = []
    
    if verbose:
        print("Starting training...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        # Use tqdm for progress bar if verbose
        train_iter = tqdm(train_loader) if verbose else train_loader
        
        for data, target in train_iter:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            
            if verbose:
                train_iter.set_description(f"Epoch {epoch+1}/{epochs} [Train]")
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        val_accuracy = 100. * correct / len(val_loader.dataset)
        val_accuracies.append(val_accuracy)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    # Test phase
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", disable=not verbose):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_accuracy = 100. * correct / len(test_loader.dataset)
    test_error = 100. - test_accuracy
    
    if verbose:
        print(f"Test Accuracy: {test_accuracy:.2f}%, Test Error: {test_error:.2f}%")
    
    return model, train_losses, val_accuracies, test_accuracy


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Backpropagation Baseline for MNIST')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--permutation-invariant', action='store_true', default=True, 
                        help='Use permutation-invariant MNIST')
    parser.add_argument('--plot', action='store_true', default=True, help='Plot training curves')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Run training
    model, train_losses, val_accuracies, test_accuracy = train_backprop_baseline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        permutation_invariant=args.permutation_invariant,
        use_cuda=use_cuda
    )
    
    # Plot training curves if requested
    if args.plot:
        plot_training_curves(train_losses, val_accuracies, title="Backpropagation Baseline Training")
