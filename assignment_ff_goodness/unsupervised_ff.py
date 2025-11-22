import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from models import UnsupervisedFFNetwork
from utils import load_mnist_data, generate_negative_samples_masked, plot_training_curves


def train_unsupervised_ff(epochs=100, batch_size=128, hidden_dims=[2000, 2000, 2000, 2000],
                         thresholds=None, learning_rates=None, use_peer_normalization=False,
                         permutation_invariant=True, use_cuda=torch.cuda.is_available(),
                         verbose=True):
    """
    Train an unsupervised Forward-Forward network on MNIST as described in Section 3.2.
    Uses real data as positive samples and masked/hybrid images as negative samples.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        hidden_dims: List of hidden layer dimensions
        thresholds: Thresholds for goodness function per layer
        learning_rates: Learning rates per layer
        use_peer_normalization: Whether to use peer normalization
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
    
    # Set default thresholds and learning rates if not provided
    if thresholds is None:
        thresholds = [2.0] * len(hidden_dims)
    if learning_rates is None:
        learning_rates = [0.01] * len(hidden_dims)
    
    # Create model
    model = UnsupervisedFFNetwork(
        input_dim, hidden_dims, thresholds, learning_rates, use_peer_normalization
    ).to(device)
    
    # Initialize classifier for evaluation
    classifier = nn.Linear(sum(hidden_dims[-3:]), 10).to(device)  # Using last three layers as in paper
    classifier_optimizer = optim.SGD(classifier.parameters(), lr=0.02)
    classifier_criterion = nn.CrossEntropyLoss()
    
    # Train FF network layer by layer
    train_losses = []
    val_accuracies = []
    
    if verbose:
        print("Starting Forward-Forward unsupervised training...")
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_pos_goodness = []
        epoch_neg_goodness = []
        
        # Training phase - train each layer separately in each epoch
        for layer_idx in range(len(hidden_dims)):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Training layer {layer_idx+1}/{len(hidden_dims)}")
            
            layer_losses = []
            layer_pos_goodness = []
            layer_neg_goodness = []
            
            train_iter = tqdm(train_loader) if verbose else train_loader
            
            for data, target in train_iter:
                data, target = data.to(device), target.to(device)
                
                # Generate negative samples using masked/hybrid images
                neg_data, _ = generate_negative_samples_masked(data, target)
                
                # Train the current layer
                loss, pos_goodness, neg_goodness = model.train_layer(layer_idx, data, neg_data)
                
                layer_losses.append(loss)
                layer_pos_goodness.append(pos_goodness)
                layer_neg_goodness.append(neg_goodness)
                
                if verbose:
                    train_iter.set_description(
                        f"Layer {layer_idx+1} - Loss: {loss:.4f}, "
                        f"Pos: {pos_goodness:.4f}, Neg: {neg_goodness:.4f}"
                    )
            
            # Record layer statistics
            epoch_losses.append(np.mean(layer_losses))
            epoch_pos_goodness.append(np.mean(layer_pos_goodness))
            epoch_neg_goodness.append(np.mean(layer_neg_goodness))
        
        # Record overall epoch statistics
        train_losses.append(np.mean(epoch_losses))
        
        # Train the classifier on extracted features (every 10 epochs)
        if epoch % 10 == 0 or epoch == epochs - 1:
            # Extract features from the trained FF network
            model.eval()
            
            # Train classifier
            for _ in range(5):  # Train classifier for a few epochs
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    
                    # Extract features using FF network
                    features = model.get_features(data, layers_to_use=[-3, -2, -1])  # Last three layers
                    
                    # Train classifier
                    classifier_optimizer.zero_grad()
                    output = classifier(features)
                    loss = classifier_criterion(output, target)
                    loss.backward()
                    classifier_optimizer.step()
            
            # Evaluate on validation set
            correct = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    
                    # Extract features using FF network
                    features = model.get_features(data, layers_to_use=[-3, -2, -1])
                    
                    # Classify
                    output = classifier(features)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            val_accuracy = 100. * correct / len(val_loader.dataset)
            val_accuracies.append(val_accuracy)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Val Accuracy: {val_accuracy:.2f}%")
    
    # Final test evaluation
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", disable=not verbose):
            data, target = data.to(device), target.to(device)
            
            # Extract features using FF network
            features = model.get_features(data, layers_to_use=[-3, -2, -1])
            
            # Classify
            output = classifier(features)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_accuracy = 100. * correct / len(test_loader.dataset)
    test_error = 100. - test_accuracy
    
    if verbose:
        print(f"Test Accuracy: {test_accuracy:.2f}%, Test Error: {test_error:.2f}%")
    
    return model, train_losses, val_accuracies, test_accuracy


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Unsupervised Forward-Forward Algorithm for MNIST')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Base learning rate')
    parser.add_argument('--threshold', type=float, default=2.0, help='Base threshold for goodness function')
    parser.add_argument('--peer-normalization', action='store_true', default=False, 
                        help='Use peer normalization')
    parser.add_argument('--permutation-invariant', action='store_true', default=True, 
                        help='Use permutation-invariant MNIST')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--plot', action='store_true', default=True, help='Plot training curves')
    parser.add_argument('--visualize', action='store_true', default=True, 
                        help='Visualize first layer weights after training')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Configure learning rates and thresholds
    hidden_dims = [2000, 2000, 2000, 2000]
    learning_rates = [args.lr] * len(hidden_dims)
    thresholds = [args.threshold] * len(hidden_dims)
    
    # Run training
    model, train_losses, val_accuracies, test_accuracy = train_unsupervised_ff(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dims=hidden_dims,
        thresholds=thresholds,
        learning_rates=learning_rates,
        use_peer_normalization=args.peer_normalization,
        permutation_invariant=args.permutation_invariant,
        use_cuda=use_cuda
    )
    
    # Plot training curves if requested
    if args.plot:
        plot_training_curves(train_losses, val_accuracies, title="Unsupervised Forward-Forward Training")
    
    # Visualize first layer weights if requested
    if args.visualize:
        from utils import visualize_first_layer_weights
        visualize_first_layer_weights(model.ff_layers[0])
