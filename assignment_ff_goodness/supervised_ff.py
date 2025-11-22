import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from models import SupervisedFFNetwork
from utils import (
    load_mnist_data, generate_supervised_negative_samples,
    prepare_images_with_labels, generate_neutral_label, plot_training_curves
)


def train_supervised_ff(epochs=60, batch_size=128, hidden_dims=[2000, 2000, 2000, 2000],
                       thresholds=None, learning_rates=None, use_peer_normalization=False,
                       permutation_invariant=True, use_cuda=torch.cuda.is_available(),
                       verbose=True, use_augmented_data=False):
    """
    Train a supervised Forward-Forward network on MNIST as described in Section 3.3.
    Embeds labels in the input, with positive data having correct labels and negative 
    data having incorrect labels.
    
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
        use_augmented_data: Whether to use data augmentation with jittering
        
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
    
    # Calculate input dimensions based on settings
    num_classes = 10
    input_dim = 784 + num_classes if permutation_invariant else (28 * 28 + num_classes)
    
    # Set default thresholds and learning rates if not provided
    if thresholds is None:
        # Use more aggressive thresholds with layer-specific values
        base_threshold = 1.8  # Lower base threshold for better gradients
        # Higher thresholds for deeper layers to enforce clearer distinctions
        thresholds = [base_threshold + 0.4 * i for i in range(len(hidden_dims))]
        
    if learning_rates is None:
        # Use higher learning rates that decrease with depth
        base_lr = 0.05  # Higher base learning rate
        learning_rates = [base_lr * (0.8 ** i) for i in range(len(hidden_dims))]
    
    # Create model
    model = SupervisedFFNetwork(
        input_dim, hidden_dims, thresholds, learning_rates, use_peer_normalization
    ).to(device)
    
    # Initialize classifier for evaluation
    classifier = nn.Linear(sum(hidden_dims[1:]), 10).to(device)  # Using all but first layer
    classifier_optimizer = optim.SGD(classifier.parameters(), lr=0.02)
    classifier_criterion = nn.CrossEntropyLoss()
    
    # Train
    train_losses = []
    val_accuracies = []
    
    if verbose:
        print("Starting Forward-Forward supervised training...")
    
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
                
                # Data augmentation if enabled (simple jittering as in Section 3.3)
                if use_augmented_data and not permutation_invariant:
                    data = augment_with_jittering(data)
                
                # Generate negative data using incorrect labels with enhanced distortions
                neg_data_raw, neg_target = generate_supervised_negative_samples(data, target, enhance=True)
                
                # Prepare positive data with embedded correct labels
                pos_data = prepare_images_with_labels(data, target, num_classes, embed_location='append')
                
                # Prepare negative data with embedded incorrect labels using the distorted images
                neg_data = prepare_images_with_labels(neg_data_raw, neg_target, num_classes, embed_location='append')
                
                # Train the current layer
                if layer_idx == 0:
                    # For first layer, use raw inputs
                    loss, pos_goodness, neg_goodness = model.ff_layers[0].update_weights(pos_data, neg_data)
                else:
                    # For subsequent layers, propagate through previous layers first
                    # Process positive data
                    pos_x = pos_data
                    for i in range(layer_idx):
                        pos_x = model.ff_layers[i](pos_x)
                        pos_x = model.ff_layers[i].layer_normalization(pos_x)
                    
                    # Process negative data
                    neg_x = neg_data
                    for i in range(layer_idx):
                        neg_x = model.ff_layers[i](neg_x)
                        neg_x = model.ff_layers[i].layer_normalization(neg_x)
                    
                    # Train current layer
                    loss, pos_goodness, neg_goodness = model.ff_layers[layer_idx].update_weights(pos_x, neg_x)
                
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
        
        # Evaluate on validation set (every 5 epochs)
        if epoch % 5 == 0 or epoch == epochs - 1:
            # Train classifier on extracted features
            train_classifier_for_evaluation(model, classifier, classifier_optimizer, 
                                            classifier_criterion, train_loader, device, num_classes)
            
            # Evaluate using classifier
            val_accuracy = evaluate_with_classifier(model, classifier, val_loader, device, num_classes)
            val_accuracies.append(val_accuracy)
            
            # Also evaluate using direct goodness-based prediction
            goodness_accuracy = evaluate_with_goodness(model, val_loader, device, num_classes)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Val Accuracy (Classifier): {val_accuracy:.2f}%, "
                      f"Val Accuracy (Goodness): {goodness_accuracy:.2f}%")
    
    # Final test evaluation
    model.eval()
    
    # Evaluate using classifier
    test_accuracy_clf = evaluate_with_classifier(model, classifier, test_loader, device, num_classes)
    
    # Evaluate using direct goodness-based prediction
    test_accuracy_good = evaluate_with_goodness(model, test_loader, device, num_classes)
    
    if verbose:
        print(f"Test Accuracy (Classifier): {test_accuracy_clf:.2f}%, "
              f"Test Error: {100-test_accuracy_clf:.2f}%")
        print(f"Test Accuracy (Goodness): {test_accuracy_good:.2f}%, "
              f"Test Error: {100-test_accuracy_good:.2f}%")
    
    return model, train_losses, val_accuracies, max(test_accuracy_clf, test_accuracy_good)


def augment_with_jittering(data, max_shift=2):
    """
    Apply jittering augmentation to images as described in Section 3.3.
    Shifts images by up to max_shift pixels in each direction.
    
    Args:
        data: Batch of images (B, C, H, W)
        max_shift: Maximum pixel shift in each direction
        
    Returns:
        augmented_data: Jittered images
    """
    batch_size, channels, height, width = data.shape
    augmented_data = torch.zeros_like(data)
    
    for i in range(batch_size):
        # Random shift
        h_shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        w_shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        
        # Apply shift with zero padding
        if h_shift >= 0 and w_shift >= 0:
            augmented_data[i, :, h_shift:, w_shift:] = data[i, :, :height-h_shift, :width-w_shift]
        elif h_shift >= 0 and w_shift < 0:
            augmented_data[i, :, h_shift:, :width+w_shift] = data[i, :, :height-h_shift, -w_shift:]
        elif h_shift < 0 and w_shift >= 0:
            augmented_data[i, :, :height+h_shift, w_shift:] = data[i, :, -h_shift:, :width-w_shift]
        else:  # h_shift < 0 and w_shift < 0
            augmented_data[i, :, :height+h_shift, :width+w_shift] = data[i, :, -h_shift:, -w_shift:]
    
    return augmented_data


def train_classifier_for_evaluation(model, classifier, optimizer, criterion, 
                                   data_loader, device, num_classes, epochs=5):
    """
    Train a classifier on features extracted from the FF network.
    
    Args:
        model: Trained FF network
        classifier: Linear classifier to train
        optimizer: Optimizer for the classifier
        criterion: Loss function
        data_loader: DataLoader for training data
        device: Device to use
        num_classes: Number of classes
        epochs: Number of training epochs for the classifier
    """
    model.eval()
    classifier.train()
    
    for _ in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Prepare data with neutral label for feature extraction
            neutral_label = generate_neutral_label(data.size(0), num_classes, device)
            data_with_label = prepare_images_with_labels(data, neutral_label.argmax(dim=1), num_classes, embed_location='append')
            
            # Extract features
            with torch.no_grad():
                features = model.get_features(data_with_label, layers_to_use=[1, 2, 3])  # Skip first layer
            
            # Train classifier
            optimizer.zero_grad()
            output = classifier(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def evaluate_with_classifier(model, classifier, data_loader, device, num_classes):
    """
    Evaluate the model using a classifier on extracted features.
    
    Args:
        model: Trained FF network
        classifier: Trained linear classifier
        data_loader: DataLoader for evaluation data
        device: Device to use
        num_classes: Number of classes
        
    Returns:
        accuracy: Classification accuracy in percentage
    """
    model.eval()
    classifier.eval()
    correct = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Prepare data with neutral label for feature extraction
            neutral_label = generate_neutral_label(data.size(0), num_classes, device)
            data_with_label = prepare_images_with_labels(data, neutral_label.argmax(dim=1), num_classes, embed_location='append')
            
            # Extract features
            features = model.get_features(data_with_label, layers_to_use=[1, 2, 3])  # Skip first layer
            
            # Classify
            output = classifier(features)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    return 100. * correct / len(data_loader.dataset)


def evaluate_with_goodness(model, data_loader, device, num_classes):
    """
    Evaluate the model by measuring goodness for each possible label.
    
    Args:
        model: Trained FF network
        data_loader: DataLoader for evaluation data
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
            
            # Predict using label scoring method from Section 3.3
            pred = model.predict_with_label_scoring(data, num_classes)
            correct += pred.eq(target).sum().item()
    
    return 100. * correct / len(data_loader.dataset)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Supervised Forward-Forward Algorithm for MNIST')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Base learning rate')
    parser.add_argument('--threshold', type=float, default=2.0, help='Base threshold for goodness function')
    parser.add_argument('--peer-normalization', action='store_true', default=False, 
                        help='Use peer normalization')
    parser.add_argument('--permutation-invariant', action='store_true', default=True, 
                        help='Use permutation-invariant MNIST')
    parser.add_argument('--augmented-data', action='store_true', default=False,
                       help='Use data augmentation with jittering')
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
    model, train_losses, val_accuracies, test_accuracy = train_supervised_ff(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dims=hidden_dims,
        thresholds=thresholds,
        learning_rates=learning_rates,
        use_peer_normalization=args.peer_normalization,
        permutation_invariant=args.permutation_invariant,
        use_cuda=use_cuda,
        use_augmented_data=args.augmented_data
    )
    
    # Plot training curves if requested
    if args.plot:
        plot_training_curves(train_losses, val_accuracies, title="Supervised Forward-Forward Training")
    
    # Visualize first layer weights if requested
    if args.visualize:
        from utils import visualize_first_layer_weights
        visualize_first_layer_weights(model.ff_layers[0])
