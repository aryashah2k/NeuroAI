import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from utils import load_mnist_data, generate_negative_samples_masked, generate_supervised_negative_samples, prepare_images_with_labels, plot_training_curves


class LayerWithAlternativeGoodness(nn.Module):
    """
    A layer for Forward-Forward algorithm with alternative goodness calculations.
    """
    def __init__(self, input_dim, output_dim, goodness_type='squared_sum', 
                threshold=2.0, learning_rate=0.01, temperature=0.1):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            goodness_type: Type of goodness function to use
            threshold: Threshold for goodness function
            learning_rate: Learning rate for local weight updates
            temperature: Temperature parameter for certain goodness functions
        """
        super(LayerWithAlternativeGoodness, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.goodness_type = goodness_type
        self.temperature = temperature
        
        # Initialize weights
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        
        # For contrastive learning
        self.register_buffer('feature_bank', torch.randn(1000, output_dim))
        self.register_buffer('bank_ptr', torch.zeros(1, dtype=torch.long))
        
    def forward(self, x):
        """Forward pass: compute pre-activations and ReLU activations."""
        z = self.linear(x)
        h = F.relu(z)
        return h
    
    def compute_goodness(self, x):
        """Compute goodness using the selected goodness function."""
        h = self.forward(x)
        
        if self.goodness_type == 'squared_sum':
            # Original FF goodness: sum of squared activations
            goodness = torch.sum(h ** 2, dim=1)
            
        elif self.goodness_type == 'cosine_similarity':
            # Cosine similarity with fixed target
            # Idea: Each layer learns to orient representations towards a target direction
            target = torch.ones_like(h)
            h_norm = F.normalize(h, dim=1)
            target_norm = F.normalize(target, dim=1)
            goodness = F.cosine_similarity(h_norm, target_norm, dim=1)
            
        elif self.goodness_type == 'entropy':
            # Negative entropy as goodness measure
            # Low entropy (concentrated activations) = high goodness
            # Use a smaller temperature for sharper distributions
            probs = F.softmax(h / (self.temperature * 0.1), dim=1)
            # Calculate negative entropy (higher for more concentrated distributions)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            # Add a sparsity bonus to reward activations with fewer active units
            active_units = torch.sum(h > 0.1, dim=1).float() / h.size(1)
            sparsity_bonus = 1.0 - active_units
            # Combined goodness: negative entropy plus sparsity bonus
            goodness = -entropy + sparsity_bonus * 2.0
            
        elif self.goodness_type == 'contrastive':
            # Create a much more distinctive contrastive goodness function
            # For positive samples: prefer high activation concentration
            # For negative samples: prefer dispersed, low activations
            
            # 1. Basic activation statistics
            h_norm = F.normalize(h, dim=1)
            h_max = torch.max(h, dim=1)[0]  # Maximum activation
            h_mean = torch.mean(h, dim=1)   # Mean activation
            h_var = torch.var(h, dim=1)     # Variance across units
            
            # 2. Activation concentration measures
            # Count active units (activation > 0.5)
            active_units = torch.sum(h > 0.5, dim=1).float() / h.size(1)
            
            # 3. L1/L2 ratio (sparsity measure)
            l1_norm = torch.norm(h, p=1, dim=1)
            l2_norm = torch.norm(h, p=2, dim=1)
            l1_l2_ratio = l1_norm / (l2_norm + 1e-6)
            
            # 4. Top-k concentration
            k = min(5, h.size(1))
            topk_values, _ = torch.topk(h, k, dim=1)
            topk_concentration = torch.sum(topk_values, dim=1) / (torch.sum(h, dim=1) + 1e-6)
            
            # 5. Reference vector correlation
            # Create a reference pattern that will differentiate pos/neg
            # For positive: alternating high/low pattern
            ref_size = h.size(1)
            pos_ref = torch.zeros(ref_size, device=h.device)
            pos_ref[::2] = 1.0  # Set every other element to 1
            pos_ref = F.normalize(pos_ref, dim=0)
            
            # Get correlation with reference
            pos_correlation = torch.mm(h_norm, pos_ref.unsqueeze(1)).squeeze()
            
            # Create final goodness score - intentionally make values further apart
            # Square the components to exaggerate differences
            goodness = (
                h_max * 0.3 + 
                h_mean * 0.2 + 
                topk_concentration * 0.3 + 
                torch.abs(pos_correlation) * 0.2 +
                (1.0 - active_units) * 0.2
            )
            
            # Apply non-linearity to separate values more dramatically
            # This will push values further apart
            goodness = torch.sigmoid(3.0 * goodness) * 3.0
            
            # Feature bank updates for monitoring only (not used in goodness)
            with torch.no_grad():
                batch_size = h.size(0)
                store_size = min(batch_size, self.feature_bank.size(0))
                ptr = int(self.bank_ptr)
                
                if ptr + store_size > self.feature_bank.size(0):
                    store_size = self.feature_bank.size(0) - ptr
                
                if store_size > 0:
                    new_bank = self.feature_bank.detach().clone()
                    new_bank[ptr:ptr+store_size] = h_norm[:store_size].detach()
                    self.feature_bank = new_bank
                    self.bank_ptr[0] = (ptr + store_size) % self.feature_bank.size(0)
            
        elif self.goodness_type == 'energy':
            # Enhanced energy-based goodness function
            # More sophisticated energy calculation to differentiate pos/neg samples
            
            # Get basic statistics
            mean_act = h.mean(dim=1)
            max_act = h.max(dim=1)[0]
            active_ratio = torch.sum(h > 0.1, dim=1).float() / h.size(1)
            
            # Create reference vectors for class prototypes
            # Use alternating pattern as one prototype
            size = h.size(1)
            alt_pattern = torch.zeros(size, device=h.device)
            alt_pattern[::2] = 1.0  # Set every other element to 1
            
            # Normalize representations and reference
            h_norm = F.normalize(h, dim=1)
            alt_pattern = F.normalize(alt_pattern, dim=0)
            
            # Compute similarity to reference pattern
            sim = torch.mm(h_norm, alt_pattern.unsqueeze(1)).squeeze()
            
            # Compute free energy terms
            # Lower energy = higher goodness in energy-based models
            activation_energy = -torch.logsumexp(h, dim=1)  # Traditional energy term
            pattern_energy = -torch.abs(sim) * 5.0          # Pattern matching energy
            sparsity_energy = -active_ratio * 3.0           # Sparsity energy (prefer fewer active units)
            
            # Combine energy components with different weights
            # Negate to convert to goodness (higher = better)
            energy = activation_energy + pattern_energy + sparsity_energy
            goodness = -energy  # Negative energy as goodness
            
            # Apply non-linear transformation to increase separation
            goodness = torch.tanh(goodness * 0.1) * 5.0
            
        elif self.goodness_type == 'mutual_info':
            # Approximation of mutual information as goodness
            # High mutual information = high goodness
            joint = torch.mm(h.T, h) / h.size(0)
            joint = joint / torch.sum(joint)
            
            marginal_i = torch.sum(joint, dim=1, keepdim=True)
            marginal_j = torch.sum(joint, dim=0, keepdim=True)
            
            mutual_info = torch.sum(joint * torch.log((joint + 1e-10) / 
                                                    ((marginal_i * marginal_j) + 1e-10)))
            
            # Since MI is computed for the whole batch, return same value for all samples
            goodness = torch.ones(x.size(0), device=x.device) * mutual_info
            
        else:
            raise ValueError(f"Unknown goodness type: {self.goodness_type}")
        
        return goodness, h
    
    def layer_normalization(self, x):
        """Simple layer normalization that divides by vector length."""
        norm = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8
        return x / norm
    
    def update_weights(self, x_pos, x_neg, normalize_inputs=True):
        """Update the weights based on positive and negative samples.
        
        Args:
            x_pos: Positive samples
            x_neg: Negative samples
            normalize_inputs: Whether to normalize inputs
        
        Returns:
            loss, pos_goodness, neg_goodness: Training metrics
        """
        batch_size = x_pos.size(0)
        
        # Extract the label part (first 10 elements) and the image part (rest)
        # This is crucial for supervised mode where labels are embedded in the inputs
        label_size = 10  # MNIST has 10 classes
        
        # Enhanced processing for supervised learning with embedded labels
        # Add noise to the image part of negative samples to make them more distinct
        if x_pos.size(1) > label_size:  # Only if we have embedded labels
            with torch.no_grad():
                # Get the image parts
                img_part_neg = x_neg[:, label_size:].clone()
                
                # Add noise to image part of negative examples (5-15% noise)
                noise_level = 0.1
                noise = torch.randn_like(img_part_neg) * noise_level
                img_part_neg += noise
                
                # Replace the image part in negative samples
                x_neg_enhanced = x_neg.clone()
                x_neg_enhanced[:, label_size:] = img_part_neg
                
                # Use the enhanced negative samples
                x_neg = x_neg_enhanced
        
        if normalize_inputs:
            # Apply input normalization
            x_pos = x_pos / (x_pos.norm(dim=1, keepdim=True) + 1e-4)
            x_neg = x_neg / (x_neg.norm(dim=1, keepdim=True) + 1e-4)
        
        # Process positive samples
        pos_goodness, pos_activations = self.compute_goodness(x_pos)

        # Process negative samples
        neg_goodness, neg_activations = self.compute_goodness(x_neg)
        
        # Compute loss based on goodness difference with an increased margin
        # Positive samples should have higher goodness than negative samples
        margin = 1.0  # Increased margin for better separation
        loss = F.relu(neg_goodness - pos_goodness + margin).mean()
        
        # Compute gradients
        # Manually compute gradients for weight updates
        grad = torch.autograd.grad(
            outputs=loss,
            inputs=self.linear.weight,
            retain_graph=False
        )[0]
        
        # Update weights manually
        with torch.no_grad():
            # Focus more on the label part of the input for supervised learning
            if x_pos.size(1) > label_size:  # Only if we have embedded labels
                # Increase gradient magnitude for label weights (first 10)
                label_gradient_multiplier = 5.0
                grad[:, :label_size] *= label_gradient_multiplier
            
            # Update weights with the modified gradient
            self.linear.weight -= self.learning_rate * grad
        
        return loss.item(), pos_goodness.mean().item(), neg_goodness.mean().item()


class FFNetworkWithAlternativeGoodness(nn.Module):
    """
    Forward-Forward network using alternative goodness functions.
    """
    def __init__(self, input_dim=784, hidden_dims=[2000, 2000, 2000, 2000],
                goodness_type='squared_sum', thresholds=None, learning_rates=None,
                temperature=0.1):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            goodness_type: Type of goodness function to use
            thresholds: Thresholds for goodness function per layer
            learning_rates: Learning rates per layer
            temperature: Temperature parameter for certain goodness functions
        """
        super(FFNetworkWithAlternativeGoodness, self).__init__()
        
        # Set default thresholds and learning rates if not provided
        if thresholds is None:
            if goodness_type == 'squared_sum':
                thresholds = [2.0] * len(hidden_dims)
            elif goodness_type in ['cosine_similarity', 'contrastive']:
                thresholds = [0.2] * len(hidden_dims)
            elif goodness_type in ['entropy', 'energy', 'mutual_info']:
                thresholds = [0.5] * len(hidden_dims)
            
        if learning_rates is None:
            learning_rates = [0.01] * len(hidden_dims)
            
        # Create layers
        self.ff_layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(hidden_dims)):
            self.ff_layers.append(
                LayerWithAlternativeGoodness(
                    dims[i], dims[i+1], goodness_type, 
                    thresholds[i], learning_rates[i], temperature
                )
            )
            
        # Linear classifier for the last layer
        self.classifier = nn.Linear(hidden_dims[-1], 10)
        
        # Store parameters
        self.goodness_type = goodness_type
        
    def forward(self, x):
        """
        Forward pass through all layers.
        Returns the activations from each layer.
        """
        activations = []
        h = x
        
        for layer in self.ff_layers:
            h = layer.forward(h)
            # Apply layer normalization before passing to the next layer
            h = layer.layer_normalization(h)
            activations.append(h)
            
        return activations
    
    def train_layer(self, layer_idx, x_pos, x_neg):
        """
        Train a single layer using positive and negative samples.
        
        Args:
            layer_idx: Index of the layer to train
            x_pos: Positive samples
            x_neg: Negative samples
            
        Returns:
            loss, pos_goodness, neg_goodness: Loss and goodness values
        """
        # Get the layer to train
        layer = self.ff_layers[layer_idx]
        
        # If not the first layer, propagate inputs through previous layers
        if layer_idx > 0:
            for i in range(layer_idx):
                x_pos = self.ff_layers[i](x_pos)
                x_neg = self.ff_layers[i](x_neg)
                # Apply layer normalization
                x_pos = self.ff_layers[i].layer_normalization(x_pos)
                x_neg = self.ff_layers[i].layer_normalization(x_neg)
        
        # Update weights of the current layer
        return layer.update_weights(x_pos, x_neg)
    
    def get_features(self, x, layers_to_use=None):
        """
        Extract features from specified layers for downstream classification.
        
        Args:
            x: Input tensor
            layers_to_use: List of layer indices to use for features
            
        Returns:
            features: Concatenated features from specified layers
        """
        if layers_to_use is None:
            # Default to last layer
            layers_to_use = [len(self.ff_layers) - 1]
            
        activations = self.forward(x)
        
        # Extract and concatenate features from specified layers
        features = []
        for idx in layers_to_use:
            if idx < len(activations):
                features.append(activations[idx])
        
        # Concatenate along feature dimension
        if len(features) == 1:
            return features[0]
        else:
            return torch.cat(features, dim=1)
    
    def classify(self, x, layers_to_use=None):
        """
        Classify input using the linear classifier on extracted features.
        
        Args:
            x: Input tensor
            layers_to_use: List of layer indices to use for features
            
        Returns:
            logits: Classification logits
        """
        features = self.get_features(x, layers_to_use)
        return self.classifier(features)


def train_ff_with_alternative_goodness(goodness_type='squared_sum', supervised=True,
                                      epochs=60, batch_size=128, 
                                      hidden_dims=[2000, 2000, 2000, 2000],
                                      thresholds=None, learning_rates=None, temperature=0.1,
                                      permutation_invariant=True, use_cuda=torch.cuda.is_available(),
                                      verbose=True):
    """
    Train a Forward-Forward network with alternative goodness functions.
    
    Args:
        goodness_type: Type of goodness function to use
        supervised: Whether to use supervised learning with label embedding
        epochs: Number of training epochs
        batch_size: Training batch size
        hidden_dims: List of hidden layer dimensions
        thresholds: Thresholds for goodness function per layer
        learning_rates: Learning rates per layer
        temperature: Temperature parameter for certain goodness functions
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
        print(f"Goodness type: {goodness_type}")
        print(f"Mode: {'Supervised' if supervised else 'Unsupervised'}")
    
    # Load data
    train_loader, val_loader, test_loader = load_mnist_data(batch_size, use_cuda, permutation_invariant)
    
    # Calculate input dimensions
    num_classes = 10
    base_dim = 784 if permutation_invariant else (28 * 28)
    # In supervised mode, we're replacing first 10 pixels, not adding to dimension
    input_dim = base_dim
    
    # Create model
    model = FFNetworkWithAlternativeGoodness(
        input_dim, hidden_dims, goodness_type, thresholds, learning_rates, temperature
    ).to(device)
    
    # Initialize classifier for evaluation
    feature_dim = sum(hidden_dims[-3:]) if not supervised else sum(hidden_dims[1:])
    classifier = nn.Linear(feature_dim, 10).to(device)
    classifier_optimizer = optim.SGD(classifier.parameters(), lr=0.02)
    classifier_criterion = nn.CrossEntropyLoss()
    
    # Train
    train_losses = []
    val_accuracies = []
    
    if verbose:
        print(f"Starting Forward-Forward training with {goodness_type} goodness...")
    
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
                
                if supervised:
                    # Supervised mode with label embedding
                    # Generate negative data using incorrect labels
                    _, neg_target = generate_supervised_negative_samples(data, target)
                    
                    # Prepare positive data with embedded correct labels
                    pos_data = prepare_images_with_labels(data, target, num_classes)
                    
                    # Prepare negative data with embedded incorrect labels
                    neg_data = prepare_images_with_labels(data, neg_target, num_classes)
                else:
                    # Unsupervised mode with masked/hybrid images
                    pos_data = data
                    neg_data, _ = generate_negative_samples_masked(data, target)
                
                # Train the current layer
                loss, pos_goodness, neg_goodness = model.train_layer(layer_idx, pos_data, neg_data)
                
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
            layers_to_use = [-3, -2, -1] if not supervised else [1, 2, 3]  # Skip first layer in supervised mode
            train_classifier_for_evaluation(model, classifier, classifier_optimizer, 
                                           classifier_criterion, train_loader, device, 
                                           num_classes, supervised, layers_to_use)
            
            # Evaluate
            val_accuracy = evaluate_with_classifier(model, classifier, val_loader, device, 
                                                  num_classes, supervised, layers_to_use)
            val_accuracies.append(val_accuracy)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Val Accuracy: {val_accuracy:.2f}%")
    
    # Final test evaluation
    model.eval()
    layers_to_use = [-3, -2, -1] if not supervised else [1, 2, 3]
    
    test_accuracy = evaluate_with_classifier(model, classifier, test_loader, device, 
                                           num_classes, supervised, layers_to_use)
    test_error = 100. - test_accuracy
    
    if verbose:
        print(f"Test Accuracy: {test_accuracy:.2f}%, Test Error: {test_error:.2f}%")
    
    return model, train_losses, val_accuracies, test_accuracy


def train_classifier_for_evaluation(model, classifier, optimizer, criterion, 
                                  data_loader, device, num_classes, supervised=True,
                                  layers_to_use=None, epochs=5):
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
        supervised: Whether the model was trained in supervised mode
        layers_to_use: List of layer indices to use for features
        epochs: Number of training epochs for the classifier
    """
    model.eval()
    classifier.train()
    
    for _ in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Prepare data based on mode
            if supervised:
                # For supervised mode, use neutral label
                neutral_label = torch.ones(data.size(0), num_classes, device=device) * 0.1
                data_processed = prepare_images_with_labels(data, neutral_label.argmax(dim=1), num_classes)
            else:
                # For unsupervised mode, use raw data
                data_processed = data
            
            # Extract features
            with torch.no_grad():
                features = model.get_features(data_processed, layers_to_use)
            
            # Train classifier
            optimizer.zero_grad()
            output = classifier(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def evaluate_with_classifier(model, classifier, data_loader, device, num_classes, 
                           supervised=True, layers_to_use=None):
    """
    Evaluate the model using a classifier on extracted features.
    
    Args:
        model: Trained FF network
        classifier: Trained linear classifier
        data_loader: DataLoader for evaluation data
        device: Device to use
        num_classes: Number of classes
        supervised: Whether the model was trained in supervised mode
        layers_to_use: List of layer indices to use for features
        
    Returns:
        accuracy: Classification accuracy in percentage
    """
    model.eval()
    classifier.eval()
    correct = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Prepare data based on mode
            if supervised:
                # For supervised mode, use neutral label
                neutral_label = torch.ones(data.size(0), num_classes, device=device) * 0.1
                data_processed = prepare_images_with_labels(data, neutral_label.argmax(dim=1), num_classes)
            else:
                # For unsupervised mode, use raw data
                data_processed = data
            
            # Extract features
            features = model.get_features(data_processed, layers_to_use)
            
            # Classify
            output = classifier(features)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    return 100. * correct / len(data_loader.dataset)


def compare_goodness_functions(goodness_types, supervised=True, epochs=30, batch_size=128, 
                              hidden_dims=[2000, 2000, 2000, 2000], use_cuda=torch.cuda.is_available(),
                              permutation_invariant=True, verbose=True):
    """
    Compare different goodness functions for the Forward-Forward algorithm.
    
    Args:
        goodness_types: List of goodness functions to compare
        supervised: Whether to use supervised learning
        epochs: Number of training epochs per model
        batch_size: Training batch size
        hidden_dims: List of hidden layer dimensions
        use_cuda: Whether to use CUDA
        permutation_invariant: Whether to use permutation-invariant MNIST
        verbose: Whether to print progress
        
    Returns:
        results: Dictionary of results for each goodness function
    """
    results = {}
    
    for goodness_type in goodness_types:
        if verbose:
            print(f"\n{'-'*80}\nTraining with goodness function: {goodness_type}\n{'-'*80}")
        
        # Set appropriate thresholds and temperature based on goodness type
        if goodness_type == 'squared_sum':
            thresholds = [2.0] * len(hidden_dims)
            temperature = 0.1
        elif goodness_type in ['cosine_similarity', 'contrastive']:
            thresholds = [0.2] * len(hidden_dims)
            temperature = 0.1
        elif goodness_type in ['entropy', 'energy']:
            thresholds = [0.5] * len(hidden_dims)
            temperature = 0.05
        elif goodness_type == 'mutual_info':
            thresholds = [0.1] * len(hidden_dims)
            temperature = 0.01
        
        # Train model with this goodness function
        model, train_losses, val_accuracies, test_accuracy = train_ff_with_alternative_goodness(
            goodness_type=goodness_type,
            supervised=supervised,
            epochs=epochs,
            batch_size=batch_size,
            hidden_dims=hidden_dims,
            thresholds=thresholds,
            learning_rates=None,  # Use default
            temperature=temperature,
            permutation_invariant=permutation_invariant,
            use_cuda=use_cuda,
            verbose=verbose
        )
        
        # Store results
        results[goodness_type] = {
            'model': model,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'test_accuracy': test_accuracy,
            'test_error': 100.0 - test_accuracy
        }
    
    # Print comparison table
    if verbose:
        print("\n" + "="*80)
        print("Goodness Function Comparison")
        print("="*80)
        print(f"{'Goodness Type':<20} {'Test Accuracy':<15} {'Test Error':<15}")
        print("-"*80)
        
        for goodness_type in goodness_types:
            test_acc = results[goodness_type]['test_accuracy']
            test_err = results[goodness_type]['test_error']
            print(f"{goodness_type:<20} {test_acc:<15.2f} {test_err:<15.2f}")
        
        print("="*80)
    
    return results


def visualize_comparison(results, title="Comparison of Goodness Functions"):
    """
    Visualize comparison of different goodness functions.
    
    Args:
        results: Dictionary of results per goodness function
        title: Plot title
    """
    goodness_types = list(results.keys())
    test_accuracies = [results[g]['test_accuracy'] for g in goodness_types]
    test_errors = [results[g]['test_error'] for g in goodness_types]
    
    # Plot test error comparison
    plt.figure(figsize=(10, 6))
    plt.bar(goodness_types, test_errors)
    plt.xlabel('Goodness Function')
    plt.ylabel('Test Error (%)')
    plt.title(f'{title} - Test Error Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('goodness_comparison.png')
    plt.show()
    
    # Plot training curves for each goodness function
    num_types = len(goodness_types)
    fig, axs = plt.subplots(num_types, 2, figsize=(15, 4*num_types))
    
    for i, goodness_type in enumerate(goodness_types):
        # Plot training loss
        axs[i, 0].plot(results[goodness_type]['train_losses'])
        axs[i, 0].set_xlabel('Epoch')
        axs[i, 0].set_ylabel('Training Loss')
        axs[i, 0].set_title(f'{goodness_type} - Training Loss')
        axs[i, 0].grid(True)
        
        # Plot validation accuracy
        epochs_per_val = len(results[goodness_type]['train_losses']) // len(results[goodness_type]['val_accuracies'])
        val_epochs = range(0, len(results[goodness_type]['val_accuracies']) * epochs_per_val, epochs_per_val)
        
        axs[i, 1].plot(val_epochs, results[goodness_type]['val_accuracies'])
        axs[i, 1].set_xlabel('Epoch')
        axs[i, 1].set_ylabel('Validation Accuracy (%)')
        axs[i, 1].set_title(f'{goodness_type} - Validation Accuracy')
        axs[i, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('goodness_training_curves.png')
    plt.show()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Alternative Goodness Measures for Forward-Forward Algorithm')
    parser.add_argument('--goodness-type', type=str, default='all',
                        choices=['all', 'squared_sum', 'cosine_similarity', 'entropy', 
                                'contrastive', 'energy', 'mutual_info'],
                        help='Goodness function to use (default: all)')
    parser.add_argument('--supervised', action='store_true', default=True,
                        help='Use supervised learning with label embedding')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs per model')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--permutation-invariant', action='store_true', default=True, 
                        help='Use permutation-invariant MNIST')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--plot', action='store_true', default=True, help='Plot comparison results')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Determine which goodness functions to test
    if args.goodness_type == 'all':
        goodness_types = ['squared_sum', 'cosine_similarity', 'entropy', 'contrastive', 'energy', 'mutual_info']
    else:
        goodness_types = [args.goodness_type]
    
    # Run comparison
    results = compare_goodness_functions(
        goodness_types=goodness_types,
        supervised=args.supervised,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dims=[2000, 2000, 2000, 2000],
        use_cuda=use_cuda,
        permutation_invariant=args.permutation_invariant,
        verbose=True
    )
    
    # Visualize results if requested
    if args.plot:
        mode_str = "Supervised" if args.supervised else "Unsupervised"
        visualize_comparison(results, f"{mode_str} Forward-Forward Algorithm")
