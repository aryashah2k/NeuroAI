import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from models import RecurrentFFNetwork
from utils import load_mnist_data, generate_supervised_negative_samples, plot_training_curves


def train_recurrent_ff(epochs=60, batch_size=128, hidden_dims=[2000, 2000, 2000],
                     thresholds=None, learning_rates=None, damping=0.3, iterations=8,
                     permutation_invariant=True, use_cuda=torch.cuda.is_available(),
                     verbose=True):
    """
    Train a recurrent Forward-Forward network on MNIST as described in Section 3.4.
    Uses multi-layer recurrent network with top-down connections.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        hidden_dims: List of hidden layer dimensions
        thresholds: Thresholds for goodness function per layer
        learning_rates: Learning rates per layer
        damping: Damping factor for recurrent updates (0.3 in the paper)
        iterations: Number of recurrent iterations (8 in the paper)
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
    output_dim = 10  # 10 classes for MNIST
    
    # Set default thresholds and learning rates if not provided
    if thresholds is None:
        # Use aggressive layer-specific thresholds to create stronger gradients
        # First layer has lower threshold to detect fine patterns
        # Output layer has highest threshold to force clearer distinctions
        base_threshold = 1.5
        thresholds = [base_threshold] + \
                    [base_threshold + 0.5 * i for i in range(len(hidden_dims))]
    
    if learning_rates is None:
        # Use larger learning rates for early layers
        base_lr = 0.05  # Higher base learning rate
        learning_rates = [base_lr] + \
                        [base_lr * (0.8 ** i) for i in range(len(hidden_dims))]
    
    # Create model
    model = RecurrentFFNetwork(
        input_dim, hidden_dims, output_dim, thresholds, learning_rates, damping, iterations
    ).to(device)
    
    # Train
    train_losses = []
    val_accuracies = []
    
    if verbose:
        print("Starting Recurrent Forward-Forward training...")
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_pos_goodness = []
        epoch_neg_goodness = []
        
        train_iter = tqdm(train_loader) if verbose else train_loader
        
        for batch_idx, (data, target) in enumerate(train_iter, 1):
            data, target = data.to(device), target.to(device)
            
            # For recurrent network with top-down feedback, we need to prepare the data properly
            
            # Convert images to proper input format if not already (flatten if needed)
            if len(data.shape) > 2:  # If images are not flattened
                data_processed = data.reshape(data.size(0), -1)  # Flatten
            else:
                data_processed = data
            
            # Convert targets to one-hot encoding for output layer
            target_one_hot = F.one_hot(target, num_classes=output_dim).float()
            
            # Generate negative labels proportionally to current model predictions
            with torch.no_grad():
                probs = model.output_probabilities(data_processed)
                # Zero out the true class and renormalize
                mask = F.one_hot(target, num_classes=output_dim).float()
                probs = probs * (1.0 - mask) + 1e-12
                probs = probs / probs.sum(dim=1, keepdim=True)
                neg_target_idx = torch.multinomial(probs, 1).squeeze()
                neg_target_one_hot = F.one_hot(neg_target_idx, num_classes=output_dim).float()

            # Use original images as negatives (paper Sec. 3.4)
            neg_data_processed = data_processed

            # Train the network with positive and negative samples
            metrics = model.train_iteration(data_processed, target_one_hot, neg_data_processed, neg_target_one_hot)
            
            # Diagnostics every 200 mini-batches
            if batch_idx % 200 == 0:
                with torch.no_grad():
                    probs_diag = model.output_probabilities(data_processed)
                    true_prob = probs_diag.gather(1, target.unsqueeze(1)).mean().item()
                    entropy = (- (probs_diag * torch.log(probs_diag + 1e-12)).sum(dim=1).mean()).item()
                    print(f"  [Diag] Batch {batch_idx}: entropy={entropy:.3f}, true_cls_prob={true_prob:.3f}, loss={metrics['total_loss']:.4f}")
            
            epoch_losses.append(metrics["total_loss"])
            epoch_pos_goodness.append(metrics["pos_goodness"])
            epoch_neg_goodness.append(metrics["neg_goodness"])
            
            if verbose:
                train_iter.set_description(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {metrics['total_loss']:.4f}, "
                    f"Pos: {metrics['pos_goodness']:.4f}, "
                    f"Neg: {metrics['neg_goodness']:.4f}"
                )
        
        # Record training statistics
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        # ---- Layer-wise goodness diagnostics ----
        with torch.no_grad():
            try:
                data_val, target_val = next(iter(val_loader))
            except StopIteration:
                data_val, target_val = next(iter(train_loader))  # fallback
            data_val, target_val = data_val.to(device), target_val.to(device)
            if len(data_val.shape) > 2:
                data_val_flat = data_val.reshape(data_val.size(0), -1)
            else:
                data_val_flat = data_val
            y_pos = F.one_hot(target_val, num_classes=output_dim).float()
            states_pos, _ = model.forward(data_val_flat, y_pos)
            # Negative labels sampled as before
            probs_tmp = model.output_probabilities(data_val_flat)
            mask_tmp = F.one_hot(target_val, num_classes=output_dim).float()
            probs_tmp = probs_tmp * (1.0 - mask_tmp) + 1e-12
            probs_tmp = probs_tmp / probs_tmp.sum(dim=1, keepdim=True)
            neg_idx_tmp = torch.multinomial(probs_tmp, 1).squeeze()
            y_neg = F.one_hot(neg_idx_tmp, num_classes=output_dim).float()
            states_neg, _ = model.forward(data_val_flat, y_neg)
            print("  [Layer Gap] pos_good - neg_good:")
            for li in range(len(model.layers)):
                g_pos = model.layers[li].compute_goodness(states_pos[li+1]).mean().item()
                g_neg = model.layers[li].compute_goodness(states_neg[li+1]).mean().item()
                print(f"    Layer {li}: {g_pos - g_neg:.4f} (pos {g_pos:.2f}, neg {g_neg:.2f})")
        
        # ---- Validation ----
        if epoch % 5 == 0 or epoch == epochs - 1:
            val_accuracy = evaluate_recurrent_ff(model, val_loader, device, output_dim)
            val_accuracies.append(val_accuracy)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    # Final test evaluation
    test_accuracy = evaluate_recurrent_ff(model, test_loader, device, output_dim)
    test_error = 100. - test_accuracy
    
    if verbose:
        print(f"Test Accuracy: {test_accuracy:.2f}%, Test Error: {test_error:.2f}%")
    
    return model, train_losses, val_accuracies, test_accuracy


def evaluate_recurrent_ff(model, data_loader, device, num_classes=10):
    """
    Evaluate recurrent FF model by trying each possible label and measuring goodness.
    
    Args:
        model: Trained recurrent FF model
        data_loader: DataLoader for evaluation
        device: Device to use
        num_classes: Number of classes
        
    Returns:
        accuracy: Classification accuracy in percentage
    """
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Predict using the model
            pred = model.predict(data, num_classes)
            correct += pred.eq(target).sum().item()
    
    return 100. * correct / len(data_loader.dataset)




if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Recurrent Forward-Forward Algorithm for MNIST')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Base learning rate')
    parser.add_argument('--threshold', type=float, default=2.0, help='Base threshold for goodness function')
    parser.add_argument('--damping', type=float, default=0.3, help='Damping factor for state updates')
    parser.add_argument('--iterations', type=int, default=8, help='Number of recurrent iterations')
    parser.add_argument('--permutation-invariant', action='store_true', default=True, 
                        help='Use permutation-invariant MNIST')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--plot', action='store_true', default=True, help='Plot training curves')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Configure model parameters
    hidden_dims = [2000, 2000, 2000]  # 3 hidden layers as in the paper
    learning_rates = [args.lr] * (len(hidden_dims) + 1)  # +1 for output layer
    thresholds = [args.threshold] * (len(hidden_dims) + 1)
    
    # Run training
    model, train_losses, val_accuracies, test_accuracy = train_recurrent_ff(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dims=hidden_dims,
        thresholds=thresholds,
        learning_rates=learning_rates,
        damping=args.damping,
        iterations=args.iterations,
        permutation_invariant=args.permutation_invariant,
        use_cuda=use_cuda
    )
    
    # Plot training curves if requested
    if args.plot:
        plot_training_curves(train_losses, val_accuracies, title="Recurrent Forward-Forward Training")
