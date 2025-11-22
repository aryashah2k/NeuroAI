"""
Biologically Plausible Learning Rules for MicRons-based Perceptrons
Implements Hebbian learning, STDP, and other biological learning mechanisms
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class HebbianLearning:
    """
    Hebbian learning rule: "Neurons that fire together, wire together"
    """
    
    def __init__(self, learning_rate: float = 0.001,  # Reduced learning rate
                 decay_rate: float = 0.01,  # Increased decay
                 homeostatic_scaling: bool = True):
        """
        Initialize Hebbian learning rule
        
        Args:
            learning_rate: Learning rate for weight updates
            decay_rate: Weight decay rate for stability
            homeostatic_scaling: Apply homeostatic scaling to prevent runaway growth
        """
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.homeostatic_scaling = homeostatic_scaling
        self.activity_history = {}
        
    def update_weights(self, module: nn.Module, pre_activity: torch.Tensor, 
                      post_activity: torch.Tensor, module_name: str = ""):
        """
        Apply Hebbian weight update
        
        Args:
            module: Neural network module to update
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            module_name: Name for tracking activity history
        """
        if not isinstance(module, nn.Linear):
            return
        
        with torch.no_grad():
            # Hebbian update: ΔW = η * pre * post^T
            batch_size = pre_activity.size(0)
            
            # Check for valid tensors
            if torch.isnan(pre_activity).any() or torch.isnan(post_activity).any():
                return
            
            # Average over batch
            pre_mean = pre_activity.mean(dim=0, keepdim=True)  # [1, input_dim]
            post_mean = post_activity.mean(dim=0, keepdim=True)  # [1, output_dim]
            
            # Ensure dimensions match module.weight [output_dim, input_dim]
            if pre_mean.size(1) != module.weight.size(1) or post_mean.size(1) != module.weight.size(0):
                return
            
            # Outer product for weight update: [output_dim, input_dim]
            weight_update = self.learning_rate * torch.mm(post_mean.t(), pre_mean)
            
            # Normalize weight update by input/output dimensions for stability
            weight_update = weight_update / (pre_mean.size(1) * post_mean.size(1))
            
            # Clip weight updates to prevent explosion
            weight_update = torch.clamp(weight_update, -0.01, 0.01)  # Much smaller updates
            
            # Check for NaN in weight update
            if torch.isnan(weight_update).any():
                return
            
            # Apply weight update
            module.weight += weight_update
            
            # Weight decay for stability (apply before clipping)
            module.weight *= (1 - self.decay_rate)
            
            # Clip weights to reasonable range
            module.weight.data = torch.clamp(module.weight.data, -2.0, 2.0)  # Tighter bounds
            
            # L2 normalization to prevent weight explosion
            weight_norm = torch.norm(module.weight, dim=1, keepdim=True)
            max_norm = 1.0
            module.weight.data = torch.where(
                weight_norm > max_norm,
                module.weight * (max_norm / weight_norm),
                module.weight
            )
            
            # Homeostatic scaling
            if self.homeostatic_scaling:
                self._apply_homeostatic_scaling(module, post_activity, module_name)
    
    def _apply_homeostatic_scaling(self, module: nn.Module, activity: torch.Tensor, 
                                  module_name: str):
        """
        Apply homeostatic scaling to maintain stable activity levels
        """
        # Track activity history
        if torch.isnan(activity).any():
            return
            
        current_activity = activity.mean().item()
        
        # Skip if activity is invalid
        if np.isnan(current_activity) or np.isinf(current_activity):
            return
        
        if module_name not in self.activity_history:
            self.activity_history[module_name] = []
        
        self.activity_history[module_name].append(current_activity)
        
        # Keep only recent history
        if len(self.activity_history[module_name]) > 100:
            self.activity_history[module_name] = self.activity_history[module_name][-100:]
        
        # Apply scaling if activity is too high or low
        if len(self.activity_history[module_name]) > 10:
            avg_activity = np.mean(self.activity_history[module_name])
            target_activity = 0.1  # Target average activity
            
            if avg_activity > target_activity * 2:
                # Scale down weights if activity too high
                module.weight *= 0.98
            elif avg_activity < target_activity * 0.5 and avg_activity > 0:
                # Scale up weights if activity too low (but not zero)
                module.weight *= 1.02
            
            # Ensure weights stay in reasonable range
            module.weight.data = torch.clamp(module.weight.data, -10.0, 10.0)

class STDPLearning:
    """
    Spike-Timing Dependent Plasticity (STDP) learning rule
    """
    
    def __init__(self, learning_rate: float = 0.001,  # Reduced learning rate
                 tau_plus: float = 20.0,
                 tau_minus: float = 20.0,
                 A_plus: float = 0.1,  # Reduced amplitudes
                 A_minus: float = 0.1):
        """
        Initialize STDP learning rule
        
        Args:
            learning_rate: Base learning rate
            tau_plus: Time constant for potentiation
            tau_minus: Time constant for depression
            A_plus: Amplitude for potentiation
            A_minus: Amplitude for depression
        """
        self.learning_rate = learning_rate
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.spike_history = {}
    
    def update_weights(self, module: nn.Module, pre_activity: torch.Tensor,
                      post_activity: torch.Tensor, module_name: str = ""):
        """
        Apply STDP weight update based on spike timing
        
        Args:
            module: Neural network module to update
            pre_activity: Presynaptic activity (treated as spike rates)
            post_activity: Postsynaptic activity (treated as spike rates)
            module_name: Name for tracking spike history
        """
        if not isinstance(module, nn.Linear):
            return
        
        with torch.no_grad():
            # Check for valid tensors
            if torch.isnan(pre_activity).any() or torch.isnan(post_activity).any():
                return
            
            # Ensure dimensions match
            if pre_activity.size(1) != module.weight.size(1) or post_activity.size(1) != module.weight.size(0):
                return
            
            # Convert activities to binary spikes (simplified)
            pre_spikes = (pre_activity > torch.rand_like(pre_activity)).float()
            post_spikes = (post_activity > torch.rand_like(post_activity)).float()
            
            # STDP update approximation
            batch_size = pre_spikes.size(0)
            
            # Potentiation: correlation between pre and post activity
            potentiation = torch.mm(post_spikes.t(), pre_spikes) / batch_size
            
            # Depression: anti-correlation (simplified as negative correlation)
            depression = torch.mm((1 - post_spikes).t(), pre_spikes) / batch_size
            
            # Apply STDP rule with proper scaling
            weight_update = self.learning_rate * (
                self.A_plus * potentiation - self.A_minus * depression
            )
            
            # Normalize by batch size and dimensions
            weight_update = weight_update / (batch_size * pre_spikes.size(1))
            
            # Clip weight updates to prevent explosion
            weight_update = torch.clamp(weight_update, -0.001, 0.001)  # Much smaller updates
            
            # Check for NaN in weight update
            if torch.isnan(weight_update).any():
                return
            
            module.weight += weight_update
            
            # Apply weight decay
            module.weight *= 0.999  # Small decay
            
            # Clip weights to prevent instability
            module.weight.data = torch.clamp(module.weight.data, -1.0, 1.0)
            
            # L2 normalization
            weight_norm = torch.norm(module.weight, dim=1, keepdim=True)
            max_norm = 0.5
            module.weight.data = torch.where(
                weight_norm > max_norm,
                module.weight * (max_norm / weight_norm),
                module.weight
            )

class BCMLearning:
    """
    BCM (Bienenstock-Cooper-Munro) learning rule with sliding threshold
    """
    
    def __init__(self, learning_rate: float = 0.01,
                 threshold_decay: float = 0.99):
        """
        Initialize BCM learning rule
        
        Args:
            learning_rate: Learning rate
            threshold_decay: Decay rate for sliding threshold
        """
        self.learning_rate = learning_rate
        self.threshold_decay = threshold_decay
        self.thresholds = {}
    
    def update_weights(self, module: nn.Module, pre_activity: torch.Tensor,
                      post_activity: torch.Tensor, module_name: str = ""):
        """
        Apply BCM weight update with sliding threshold
        """
        if not isinstance(module, nn.Linear):
            return
        
        with torch.no_grad():
            # Initialize threshold if not exists
            if module_name not in self.thresholds:
                self.thresholds[module_name] = torch.ones_like(post_activity.mean(dim=0)) * 0.1
            
            threshold = self.thresholds[module_name]
            
            # BCM learning function: φ(y) = y(y - θ)
            post_mean = post_activity.mean(dim=0)
            phi = post_mean * (post_mean - threshold)
            
            # Weight update
            pre_mean = pre_activity.mean(dim=0, keepdim=True)
            weight_update = self.learning_rate * torch.mm(phi.unsqueeze(1), pre_mean)
            
            module.weight += weight_update
            
            # Update sliding threshold
            self.thresholds[module_name] = (
                self.threshold_decay * threshold + 
                (1 - self.threshold_decay) * post_mean ** 2
            )

class BiologicalTrainer:
    """
    Trainer for biological perceptrons using biologically plausible learning rules
    """
    
    def __init__(self, model: nn.Module, learning_rule: str = "hebbian",
                 learning_rate: float = 0.01, **kwargs):
        """
        Initialize biological trainer
        
        Args:
            model: Biological perceptron model
            learning_rule: Type of learning rule ("hebbian", "stdp", "bcm")
            learning_rate: Learning rate
            **kwargs: Additional arguments for learning rules
        """
        self.model = model
        self.learning_rule_name = learning_rule
        self.learning_rate = learning_rate
        
        # Initialize learning rule
        if learning_rule == "hebbian":
            self.learning_rule = HebbianLearning(learning_rate, **kwargs)
        elif learning_rule == "stdp":
            self.learning_rule = STDPLearning(learning_rate, **kwargs)
        elif learning_rule == "bcm":
            self.learning_rule = BCMLearning(learning_rate, **kwargs)
        else:
            raise ValueError(f"Unknown learning rule: {learning_rule}")
        
        # Track module activities for learning
        self.module_activities = {}
        self._register_hooks()
        
        logger.info(f"Initialized biological trainer with {learning_rule} learning")
    
    def _register_hooks(self):
        """
        Register forward hooks to capture module activities
        """
        def hook_fn(name):
            def hook(module, input, output):
                self.module_activities[name] = {
                    'input': input[0] if input else None,
                    'output': output
                }
            return hook
        
        # Register hooks for linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(hook_fn(name))
    
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step with biological learning
        
        Args:
            inputs: Input batch
            targets: Target batch
            
        Returns:
            Dictionary with training metrics
        """
        # Forward pass
        self.model.train()
        outputs = self.model(inputs)
        
        # Calculate loss for monitoring (not used for weight updates)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        
        # Apply biological learning to each module
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.module_activities:
                activity_data = self.module_activities[name]
                
                if activity_data['input'] is not None and activity_data['output'] is not None:
                    self.learning_rule.update_weights(
                        module, 
                        activity_data['input'], 
                        activity_data['output'],
                        name
                    )
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == targets).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'learning_rule': self.learning_rule_name
        }
    
    def evaluate(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model without learning updates
        
        Args:
            inputs: Input batch
            targets: Target batch
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(inputs)
            
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, targets)
            
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == targets).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }

class MetaplasticityLearning:
    """
    Metaplasticity: plasticity of synaptic plasticity
    Learning rate adapts based on recent synaptic activity
    """
    
    def __init__(self, base_learning_rate: float = 0.01,
                 adaptation_rate: float = 0.001,
                 activity_window: int = 100):
        """
        Initialize metaplasticity learning
        
        Args:
            base_learning_rate: Base learning rate
            adaptation_rate: Rate of learning rate adaptation
            activity_window: Window for activity averaging
        """
        self.base_learning_rate = base_learning_rate
        self.adaptation_rate = adaptation_rate
        self.activity_window = activity_window
        self.activity_history = {}
        self.learning_rates = {}
    
    def update_weights(self, module: nn.Module, pre_activity: torch.Tensor,
                      post_activity: torch.Tensor, module_name: str = ""):
        """
        Apply metaplastic weight update with adaptive learning rate
        """
        if not isinstance(module, nn.Linear):
            return
        
        # Track activity
        if module_name not in self.activity_history:
            self.activity_history[module_name] = []
            self.learning_rates[module_name] = self.base_learning_rate
        
        current_activity = post_activity.mean().item()
        self.activity_history[module_name].append(current_activity)
        
        # Keep only recent history
        if len(self.activity_history[module_name]) > self.activity_window:
            self.activity_history[module_name] = self.activity_history[module_name][-self.activity_window:]
        
        # Adapt learning rate based on activity variance
        if len(self.activity_history[module_name]) > 10:
            activity_var = np.var(self.activity_history[module_name])
            
            # Higher variance -> lower learning rate (more stable)
            # Lower variance -> higher learning rate (more plastic)
            adaptation = 1.0 / (1.0 + activity_var)
            
            self.learning_rates[module_name] = (
                (1 - self.adaptation_rate) * self.learning_rates[module_name] +
                self.adaptation_rate * self.base_learning_rate * adaptation
            )
        
        # Apply Hebbian update with adaptive learning rate
        with torch.no_grad():
            current_lr = self.learning_rates[module_name]
            
            pre_mean = pre_activity.mean(dim=0, keepdim=True)
            post_mean = post_activity.mean(dim=0, keepdim=True)
            
            weight_update = current_lr * torch.mm(post_mean.t(), pre_mean)
            module.weight += weight_update
            
            # Weight decay
            module.weight *= 0.999

def create_biological_trainer(model: nn.Module, learning_rule: str = "hebbian",
                            **kwargs) -> BiologicalTrainer:
    """
    Factory function to create biological trainer
    
    Args:
        model: Biological perceptron model
        learning_rule: Type of learning rule
        **kwargs: Additional arguments
        
    Returns:
        BiologicalTrainer instance
    """
    return BiologicalTrainer(model, learning_rule, **kwargs)

def main():
    """
    Test biological learning rules
    """
    print("Testing Biological Learning Rules")
    print("=" * 40)
    
    # Test individual learning rules
    print("Available learning rules:")
    print("- Hebbian Learning")
    print("- STDP (Spike-Timing Dependent Plasticity)")
    print("- BCM (Bienenstock-Cooper-Munro)")
    print("- Metaplasticity")
    
    print("\nBiological learning rules ready for training")

if __name__ == "__main__":
    main()
