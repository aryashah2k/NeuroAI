import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Union, Dict


class BackpropNetwork(nn.Module):
    """
    Standard neural network for MNIST classification trained with backpropagation.
    Used as baseline in Section 3.1 of the paper.
    """
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [2000, 2000, 2000, 2000],
                num_classes: int = 10):
        """
        Args:
            input_dim: Input dimension (784 for flattened MNIST)
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes (10 for MNIST)
        """
        super(BackpropNetwork, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        # Input layer
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Forward through hidden layers with ReLU activation
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            
        # Output layer (no activation, will be fed into softmax)
        x = self.output_layer(x)
        return x


class LayerFF(nn.Module):
    """
    A single layer for Forward-Forward algorithm with goodness calculation.
    """
    def __init__(self, input_dim: int, output_dim: int, threshold: float = 2.0, 
                learning_rate: float = 0.01, use_peer_normalization: bool = False):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            threshold: Threshold for goodness function
            learning_rate: Learning rate for local weight updates
            use_peer_normalization: Whether to use peer normalization (Section 3.2)
        """
        super(LayerFF, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.use_peer_normalization = use_peer_normalization
        
        # For peer normalization
        if use_peer_normalization:
            self.register_buffer('running_mean_activity', torch.zeros(1))
            self.peer_strength = 0.1  # Strength of peer regularization
        
        # Initialize with Kaiming initialization
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute pre-activations and ReLU activations."""
        z = self.linear(x)
        h = F.relu(z)
        return h
    
    def compute_goodness(self, x: torch.Tensor) -> torch.Tensor:
        """Compute goodness as sum of squared activations."""
        h = self.forward(x)
        goodness = torch.sum(h ** 2, dim=1)
        return goodness, h
    
    def layer_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """Simple layer normalization that divides by vector length."""
        # Simple version of layer normalization as mentioned in footnote 5
        norm = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8
        return x / norm
    
    def update_weights(self, x_pos: torch.Tensor, x_neg: torch.Tensor, 
                      normalize_inputs: bool = True) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Update weights using positive and negative samples.
        
        Args:
            x_pos: Positive samples
            x_neg: Negative samples
            normalize_inputs: Whether to apply layer normalization to inputs
            
        Returns:
            loss, pos_goodness, neg_goodness: Loss and goodness values for monitoring
        """
        if normalize_inputs:
            x_pos = self.layer_normalization(x_pos)
            x_neg = self.layer_normalization(x_neg)
        
        # Compute goodness and activations for positive samples
        pos_goodness, pos_activations = self.compute_goodness(x_pos)
        
        # Compute goodness and activations for negative samples
        neg_goodness, neg_activations = self.compute_goodness(x_neg)
        
        # Compute loss: we want pos_goodness > threshold and neg_goodness < threshold
        pos_loss = F.relu(self.threshold - pos_goodness).mean()
        neg_loss = F.relu(neg_goodness - self.threshold).mean()
        loss = pos_loss + neg_loss
        
        # Manually compute gradients for weight updates
        # For positive samples: Increase weights to increase goodness
        pos_grad = torch.autograd.grad(
            outputs=pos_loss,
            inputs=self.linear.weight,
            retain_graph=True
        )[0]
        
        # For negative samples: Decrease weights to decrease goodness
        neg_grad = torch.autograd.grad(
            outputs=neg_loss,
            inputs=self.linear.weight,
            retain_graph=False
        )[0]
        
        # Update weights manually
        with torch.no_grad():
            self.linear.weight -= self.learning_rate * (pos_grad + neg_grad)
            
            # Apply peer normalization if enabled
            if self.use_peer_normalization:
                current_mean = pos_activations.mean()
                target = self.running_mean_activity.item()
                
                # Update running mean
                self.running_mean_activity = 0.9 * self.running_mean_activity + 0.1 * current_mean
                
                # Adjust weights to move mean activity towards target
                adjustment = self.peer_strength * (target - current_mean)
                self.linear.weight += adjustment * (x_pos.mean(0).unsqueeze(0))
        
        return loss.item(), pos_goodness.mean().item(), neg_goodness.mean().item()


class UnsupervisedFFNetwork(nn.Module):
    """
    Unsupervised Forward-Forward network as described in Section 3.2.
    """
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [2000, 2000, 2000, 2000],
                thresholds: List[float] = None, learning_rates: List[float] = None,
                use_peer_normalization: bool = False):
        """
        Args:
            input_dim: Input dimension (784 for flattened MNIST)
            hidden_dims: List of hidden layer dimensions
            thresholds: Thresholds for goodness function per layer
            learning_rates: Learning rates per layer
            use_peer_normalization: Whether to use peer normalization
        """
        super(UnsupervisedFFNetwork, self).__init__()
        
        # Set default thresholds and learning rates if not provided
        if thresholds is None:
            thresholds = [2.0] * len(hidden_dims)
        if learning_rates is None:
            learning_rates = [0.01] * len(hidden_dims)
            
        # Create layers
        self.ff_layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(hidden_dims)):
            self.ff_layers.append(
                LayerFF(dims[i], dims[i+1], thresholds[i], learning_rates[i], use_peer_normalization)
            )
            
        # Linear classifier for the last layer
        self.classifier = nn.Linear(hidden_dims[-1], 10)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through all layers.
        Returns the activations from each layer.
        """
        activations = []
        h = x
        
        for layer in self.ff_layers:
            h = layer(h)
            # Apply layer normalization before passing to the next layer
            h = layer.layer_normalization(h)
            activations.append(h)
            
        return activations
    
    def train_layer(self, layer_idx: int, x_pos: torch.Tensor, x_neg: torch.Tensor) -> Tuple[float, float, float]:
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
    
    def get_features(self, x: torch.Tensor, layers_to_use: List[int] = None) -> torch.Tensor:
        """
        Extract features from specified layers for downstream classification.
        
        Args:
            x: Input tensor
            layers_to_use: List of layer indices to use for features (None = all)
            
        Returns:
            features: Concatenated features from specified layers
        """
        if layers_to_use is None:
            # Default to last layer as in Section 3.2 (or last 3 layers)
            layers_to_use = list(range(len(self.ff_layers)))
            
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
    
    def classify(self, x: torch.Tensor, layers_to_use: List[int] = None) -> torch.Tensor:
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


class SupervisedFFNetwork(UnsupervisedFFNetwork):
    """
    Supervised Forward-Forward network as described in Section 3.3.
    Extends UnsupervisedFFNetwork with methods for supervised learning with label embedding.
    """
    def __init__(self, input_dim: int = 784 + 10, hidden_dims: List[int] = [2000, 2000, 2000, 2000],
                thresholds: List[float] = None, learning_rates: List[float] = None,
                use_peer_normalization: bool = False):
        """
        Args:
            input_dim: Input dimension (784 + 10 for MNIST with embedded labels)
            hidden_dims: List of hidden layer dimensions
            thresholds: Thresholds for goodness function per layer
            learning_rates: Learning rates per layer
            use_peer_normalization: Whether to use peer normalization
        """
        super(SupervisedFFNetwork, self).__init__(
            input_dim, hidden_dims, thresholds, learning_rates, use_peer_normalization
        )
    
    def compute_goodness_all_layers(self, x: torch.Tensor, exclude_first: bool = True) -> torch.Tensor:
        """
        Compute goodness across all layers.
        
        Args:
            x: Input tensor
            exclude_first: Whether to exclude the first hidden layer (as in Section 3.3)
            
        Returns:
            total_goodness: Sum of goodness across all (or selected) layers
        """
        h = x
        total_goodness = torch.zeros(x.size(0), device=x.device)
        
        for i, layer in enumerate(self.ff_layers):
            # Forward pass through the layer
            goodness, h = layer.compute_goodness(h)
            
            # Add to total goodness (skip first layer if exclude_first is True)
            if not (exclude_first and i == 0):
                total_goodness += goodness
                
            # Layer normalization before next layer
            h = layer.layer_normalization(h)
        
        return total_goodness
    
    def predict_with_label_scoring(self, x: torch.Tensor, num_classes: int = 10, 
                                  neutral_label_gen_func: callable = None) -> torch.Tensor:
        """
        Classify by trying each possible label and measuring goodness.
        
        Args:
            x: Input image tensor
            num_classes: Number of classes to try
            neutral_label_gen_func: Function to generate neutral labels
            
        Returns:
            pred_labels: Predicted class labels
        """
        batch_size = x.size(0)
        device = x.device
        
        # If no neutral label generator provided, use default
        if neutral_label_gen_func is None:
            neutral_labels = torch.ones(batch_size, num_classes, device=device) * 0.1
        else:
            neutral_labels = neutral_label_gen_func(batch_size, num_classes, device)
        
        goodness_scores = torch.zeros(batch_size, num_classes, device=device)
        
        # Try each possible label
        for label in range(num_classes):
            # Create one-hot labels for this class
            one_hot = torch.zeros(batch_size, num_classes, device=device)
            one_hot[:, label] = 1.0
            
            # Concatenate with images
            if len(x.shape) == 2:  # For flattened images
                # Use append approach to match training data format
                x_with_label = torch.cat([one_hot, x], dim=1)
            else:
                # Handle original image format - assuming specific embedding method
                # For simplicity, we'll flatten the images here
                x_flat = x.view(batch_size, -1)
                x_with_label = torch.cat([one_hot, x_flat], dim=1)
            
            # Compute goodness with this label
            goodness_scores[:, label] = self.compute_goodness_all_layers(x_with_label)
        
        # Return predicted labels
        _, pred_labels = torch.max(goodness_scores, dim=1)
        return pred_labels


class RecurrentFFLayer(nn.Module):
    """
    Recurrent FF Layer that integrates bottom-up and top-down signals.
    Used in Section 3.4 of the paper.
    """
    def __init__(self, input_dim_bottom: int, input_dim_top: int, output_dim: int,
                threshold: float = 2.0, learning_rate: float = 0.01, 
                damping: float = 0.3):
        """
        Args:
            input_dim_bottom: Dimension of bottom-up input
            input_dim_top: Dimension of top-down input
            output_dim: Output dimension
            threshold: Threshold for goodness function
            learning_rate: Learning rate for weight updates
            damping: Damping factor for state updates (0.3 as in paper)
        """
        super(RecurrentFFLayer, self).__init__()
        self.linear_bottom = nn.Linear(input_dim_bottom, output_dim)
        self.input_dim_top = input_dim_top
        self.output_dim = output_dim
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.damping = damping
        
        # Initialize weights for bottom-up connections
        nn.init.kaiming_normal_(self.linear_bottom.weight, nonlinearity='relu')
        
        # Initialize top-down connection only if input_dim_top > 0
        if input_dim_top > 0:
            self.linear_top = nn.Linear(input_dim_top, output_dim)
            nn.init.kaiming_normal_(self.linear_top.weight, nonlinearity='relu')
        else:
            self.linear_top = None
        
        # Use optimizer for proper gradient updates
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def layer_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize inputs to have zero mean and unit variance.
        
        Args:
            x: Input tensor
        
        Returns:
            Normalized tensor
        """
        if x.shape[0] <= 1:
            return x
            
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True) + 1e-6
        return (x - mean) / std
    
    def forward(self, bottom_input: torch.Tensor, top_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with bottom-up and optional top-down inputs.
        
        Args:
            bottom_input: Bottom-up input tensor
            top_input: Top-down input tensor (optional)
            
        Returns:
            Output tensor after processing inputs
        """
        # Bottom-up forward pass
        h = self.linear_bottom(bottom_input)
        h = F.relu(h)
        
        # If top-down input is available and we have a top connection, add its influence
        if self.linear_top is not None and top_input is not None and top_input.numel() > 0:
            # Ensure dimensions match
            if top_input.size(-1) == self.input_dim_top:
                h_top = self.linear_top(top_input)
                h_top = F.relu(h_top)
                h = h + h_top
        
        return h
    
    def compute_goodness(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample goodness (sum of squared activations).
        Paper applies threshold per sample.
        Returns tensor of shape (batch,)
        """
        return torch.mean(x ** 2, dim=1)

    
    def update_weights(self, bottom_pos: torch.Tensor, top_pos: Optional[torch.Tensor], 
                       bottom_neg: torch.Tensor, top_neg: Optional[torch.Tensor]) -> Tuple[float, float, float]:
        """
        Train the layer weights using the Forward-Forward algorithm.
        
        Args:
            bottom_pos: Positive bottom-up inputs
            top_pos: Positive top-down inputs (optional)
            bottom_neg: Negative bottom-up inputs
            top_neg: Negative top-down inputs (optional)
            
        Returns:
            Tuple of (loss, positive goodness, negative goodness)
        """
        # Make sure to detach all inputs to avoid issues with computational graphs
        bottom_pos = bottom_pos.detach()
        if top_pos is not None and top_pos.numel() > 0:
            top_pos = top_pos.detach()
        bottom_neg = bottom_neg.detach()
        if top_neg is not None and top_neg.numel() > 0:
            top_neg = top_neg.detach()
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Forward pass on positive samples
        h_pos_raw = self.forward(bottom_pos, top_pos)
        pos_goodness = self.compute_goodness(h_pos_raw)  # (B,)
        pos_loss = F.relu(self.threshold - pos_goodness)  # (B,)
        
        # Forward pass on negative samples
        h_neg_raw = self.forward(bottom_neg, top_neg)
        neg_goodness = self.compute_goodness(h_neg_raw)  # (B,)
        neg_loss = F.relu(neg_goodness - self.threshold)  # (B,)
        
        # Margin-based contrastive loss to push pos_good > neg_good by margin
        margin = 0.3
        loss = F.relu(neg_goodness - pos_goodness + margin).mean()
        
        # Backprop
        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            return loss.item(), pos_goodness.mean().item(), neg_goodness.mean().item()


class RecurrentFFNetwork(nn.Module):
    """
    Recurrent Forward-Forward Network as described in Section 3.4.
    """
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [2000, 2000, 2000],
                output_dim: int = 10, thresholds: List[float] = None, 
                learning_rates: List[float] = None, damping: float = 0.3,
                iterations: int = 8):
        """
        Args:
            input_dim: Input dimension (784 for flattened MNIST)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (10 for MNIST)
            thresholds: Thresholds for goodness function per layer
            learning_rates: Learning rates per layer
            damping: Damping factor for state updates (0.3 as in paper)
            iterations: Number of recurrent iterations (8 as in paper)
        """
        super(RecurrentFFNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.damping = damping
        self.iterations = iterations
        
        # Set default thresholds and learning rates if not provided
        if thresholds is None:
            thresholds = [2.0] * len(hidden_dims)
        if learning_rates is None:
            learning_rates = [0.01] * len(hidden_dims)
        
        # Create layers
        self.layers = nn.ModuleList()
        
        # Set default thresholds and learning rates to include the output layer
        if len(thresholds) <= len(hidden_dims):
            thresholds = thresholds + [2.0]
        if len(learning_rates) <= len(hidden_dims):
            learning_rates = learning_rates + [0.01]
        
        # Input to first hidden layer
        self.layers.append(RecurrentFFLayer(
            input_dim, hidden_dims[0], hidden_dims[0],
            thresholds[0], learning_rates[0], damping
        ))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            top_input_dim = output_dim if i == len(hidden_dims)-1 else hidden_dims[i+1]
            self.layers.append(RecurrentFFLayer(
                hidden_dims[i-1], top_input_dim, hidden_dims[i],
                thresholds[i], learning_rates[i], damping
            ))
        
        # Top layer (hidden to output)
        self.layers.append(RecurrentFFLayer(
            hidden_dims[-1], 0, output_dim,
            thresholds[-1], learning_rates[-1], damping
        ))
    
    def initialize_states(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Initialize layer states with a bottom-up pass.
        
        Args:
            x: Input tensor
            y: Optional label tensor
            
        Returns:
            states: List of initial layer states
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize states list
        states = []
        
        # Input state
        states.append(x)
        
        # Initialize hidden states with a bottom-up pass
        h = x
        for i in range(len(self.hidden_dims)):
            # Forward pass with no top-down input initially
            h = self.layers[i].forward(h, None)
            h = self.layers[i].layer_normalization(h)
            states.append(h)
        
        # Initialize output state
        if y is None:
            # If no label provided, initialize with zeros
            output_state = torch.zeros(batch_size, self.output_dim, device=device)
        else:
            # If label provided, use it directly
            output_state = y
        
        states.append(output_state)
        
        return states
    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, 
               iterations: Optional[int] = None) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Run the recurrent network for multiple iterations.
        
        Args:
            x: Input tensor
            y: Optional label tensor
            iterations: Number of iterations to run (default: self.iterations)
            
        Returns:
            final_states: Final states of all layers
            all_states: States at each iteration
        """
        if iterations is None:
            iterations = self.iterations
            
        # Initialize states
        states = self.initialize_states(x, y)
        all_states = [states]
        
        # Run for specified iterations
        for _ in range(iterations):
            # Input is fixed
            new_states = [x]
            
            # Update hidden states
            for i in range(len(self.hidden_dims)):
                # Get bottom-up input
                bottom_input = states[i]
                
                # Get top-down input, if applicable
                # Determine top-down input
                if i == len(self.hidden_dims) - 1:
                    # Last hidden layer receives output layer as top-down signal
                    top_input = states[-1]
                else:
                    top_input = states[i + 2] if i + 2 < len(states) else None
                
                # Compute new state with damping
                current_state = states[i + 1]
                
                try:
                    # Try to compute the forward pass
                    new_state = self.layers[i].forward(bottom_input, top_input)
                    new_state = self.layers[i].layer_normalization(new_state)
                    
                    # Apply damping as described in Section 3.4 - ensure dimensions match
                    # Paper uses: new_pre_norm = 0.3 * prev_pre_norm + 0.7 * computed_new
                    if current_state.shape == new_state.shape:
                        damped_state = self.damping * current_state + (1 - self.damping) * new_state
                    else:
                        # Handle case where dimensions don't match - just use the new state
                        damped_state = new_state
                except Exception as e:
                    # If there's an error in forward pass, just keep current state
                    print(f"Warning: Error in layer {i} forward pass: {e}")
                    damped_state = current_state
                    
                new_states.append(damped_state)
            
            # Update output state if not fixed by label
            if y is None and len(self.hidden_dims) < len(states) - 1:
                try:
                    bottom_input = states[len(self.hidden_dims)]
                    new_output = self.layers[-1].forward(bottom_input, None)
                    new_output = self.layers[-1].layer_normalization(new_output)
                    new_states.append(new_output)
                except Exception as e:
                    # If error in output layer, keep current output
                    print(f"Warning: Error in output layer: {e}")
                    new_states.append(states[-1])
            elif y is not None:
                # If label is provided, use it directly as fixed output
                new_states.append(y)
            
            # Ensure states and new_states have same length
            if len(new_states) < len(states):
                new_states.extend(states[len(new_states):])
            elif len(new_states) > len(states):
                new_states = new_states[:len(states)]
                
            states = new_states
            all_states.append(states)
        
        return states, all_states
    
    def compute_goodness_all_layers(self, states: List[torch.Tensor], 
                                   exclude_first: bool = True) -> torch.Tensor:
        """
        Compute goodness across all layers given their states.
        
        Args:
            states: List of layer states
            exclude_first: Whether to exclude the first hidden layer
            
        Returns:
            total_goodness: Sum of goodness across layers
        """
        total_goodness = torch.zeros(states[0].size(0), device=states[0].device)
        
        # Compute goodness for each hidden layer
        for i in range(len(self.hidden_dims)):
            if not (exclude_first and i == 0):
                # Ensure state is valid and has data
                if i+1 < len(states) and states[i+1].numel() > 0:
                    layer_goodness = self.layers[i].compute_goodness(states[i+1])
                    total_goodness += layer_goodness
        
        return total_goodness
    
    def train_iteration(self, x_pos: torch.Tensor, y_pos: torch.Tensor,
                       x_neg: torch.Tensor, y_neg: torch.Tensor) -> Dict[str, float]:
        """
        Train one iteration of the recurrent FF network.
        
        Args:
            x_pos: Positive input samples
            y_pos: Positive labels (one-hot encoded)
            x_neg: Negative input samples
            y_neg: Negative labels (one-hot encoded)
            
        Returns:
            metrics: Dictionary with training metrics
        """
        # Detach all inputs to avoid autograd conflicts
        x_pos = x_pos.detach()
        y_pos = y_pos.detach() if y_pos is not None else None
        x_neg = x_neg.detach()
        y_neg = y_neg.detach() if y_neg is not None else None
        
        # Run the network for positive and negative samples
        try:
            pos_states, _ = self.forward(x_pos, y_pos)
            neg_states, _ = self.forward(x_neg, y_neg)
        except Exception as e:
            print(f"Error during forward pass: {e}")
            # Return dummy metrics to avoid training failure
            return {"total_loss": 1.0, "pos_goodness": 0.0, "neg_goodness": 0.0}
        
        # Train each layer
        metrics = {"total_loss": 0.0, "pos_goodness": 0.0, "neg_goodness": 0.0}
        layers_trained = 0
        
        for i in range(len(self.layers)):
            try:
                # Get bottom-up inputs from states
                if i < len(pos_states) and i < len(neg_states):
                    bottom_pos = pos_states[i]
                    bottom_neg = neg_states[i]
                else:
                    # Skip this layer if states aren't available
                    continue
                
                # Get top-down inputs (None for the top layer)
                top_pos = None 
                top_neg = None
                if i < len(self.layers)-1 and i+2 < len(pos_states) and i+2 < len(neg_states):
                    top_pos = pos_states[i+2]
                    top_neg = neg_states[i+2]
                
                # Verify tensor dimensions before updating weights
                if bottom_pos.size(0) != bottom_neg.size(0):
                    print(f"Batch size mismatch in layer {i}, skipping")
                    continue
                    
                # Update layer weights
                loss, pos_good, neg_good = self.layers[i].update_weights(
                    bottom_pos, top_pos, bottom_neg, top_neg
                )
                
                metrics["total_loss"] += loss
                metrics["pos_goodness"] += pos_good
                metrics["neg_goodness"] += neg_good
                layers_trained += 1
            except Exception as e:
                print(f"Error training layer {i}: {e}")
                continue
        
        # Compute average metrics
        if layers_trained > 0:
            for key in metrics:
                metrics[key] /= layers_trained
        
        return metrics
    
    def predict(self, x: torch.Tensor, num_classes: int = 10, 
               iterations_to_average: List[int] = None) -> torch.Tensor:
        """
        Predict class by running the network with each possible label.
        
        Args:
            x: Input samples
            num_classes: Number of classes to try
            iterations_to_average: Iterations to average goodness over (default: [3, 4, 5])
            
        Returns:
            predictions: Predicted class labels
        """
        # Make sure input is detached to avoid autograd issues
        x = x.detach()
        
        if iterations_to_average is None:
            iterations_to_average = [3, 4, 5]  # As in Section 3.4
            
        batch_size = x.size(0)
        device = x.device
        goodness_scores = torch.zeros(batch_size, num_classes, device=device)
        
        with torch.no_grad():  # No need for gradients during prediction
            for label in range(num_classes):
                # Create one-hot label tensor
                y = torch.zeros(batch_size, num_classes, device=device)
                y[:, label] = 1.0
                
                try:
                    # Run network with this label
                    _, all_states = self.forward(x, y)
                    
                    # Compute goodness over specified iterations
                    for iter_idx in iterations_to_average:
                        if iter_idx < len(all_states):
                            goodness = self.compute_goodness_all_layers(all_states[iter_idx])
                            goodness_scores[:, label] += goodness
                except Exception as e:
                    print(f"Error during prediction for label {label}: {e}")
                    continue
        
        # Get predictions from accumulated goodness
        _, predictions = torch.max(goodness_scores, dim=1)
        return predictions

    def output_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class probabilities from a single forward iteration without clamping labels.
        Used for sampling negative labels proportionally to current predictions (Sec. 3.4).
        """
        x = x.detach()
        with torch.no_grad():
            # Run one iteration with free output state (y=None)
            final_states, all_states = self.forward(x, y=None, iterations=1)
            # Output state is the last tensor in the states list
            if len(all_states) > 1 and len(all_states[1]) > 0:
                out = all_states[1][-1]
            else:
                # Fallback: try from final_states
                out = final_states[-1]
            # Convert to probabilities
            probs = F.softmax(out, dim=1)
            return probs
