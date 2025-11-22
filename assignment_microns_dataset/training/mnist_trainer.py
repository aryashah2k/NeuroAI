"""
MNIST Trainer for MicRons-based Biological Perceptrons
Trains biological perceptron models on MNIST using biologically plausible learning rules
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import logging
import os
from tqdm import tqdm
import json

# Import our biological components
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.biological_perceptron import BiologicalPerceptron, MultiNeuronPerceptron
from models.learning_rules import BiologicalTrainer

logger = logging.getLogger(__name__)

class MNISTBiologicalTrainer:
    """
    MNIST trainer for biological perceptrons using real MicRons morphology
    """
    
    def __init__(self, morphology_data_list: List[Dict[str, Any]],
                 learning_rule: str = "hebbian",
                 batch_size: int = 32,
                 biological_constraints: bool = True,
                 device: str = "cpu"):
        """
        Initialize MNIST trainer for biological perceptrons
        
        Args:
            morphology_data_list: List of processed MicRons morphology data
            learning_rule: Biological learning rule to use
            batch_size: Training batch size
            biological_constraints: Apply biological constraints
            device: Device to run on
        """
        self.morphology_data_list = morphology_data_list
        self.learning_rule = learning_rule
        self.batch_size = batch_size
        self.biological_constraints = biological_constraints
        self.device = device
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(device)
        
        # Initialize biological trainer
        self.trainer = BiologicalTrainer(
            self.model, 
            learning_rule=learning_rule,
            learning_rate=0.01
        )
        
        # Load MNIST data
        self.train_loader, self.test_loader = self._load_mnist_data()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'epochs': []
        }
        
        logger.info(f"Initialized MNIST trainer with {len(morphology_data_list)} neurons")
        logger.info(f"Learning rule: {learning_rule}")
        logger.info(f"Biological constraints: {biological_constraints}")
    
    def _create_model(self) -> nn.Module:
        """
        Create biological perceptron model from MicRons data
        """
        if len(self.morphology_data_list) == 1:
            model = BiologicalPerceptron(
                self.morphology_data_list[0],
                input_dim=784,  # MNIST flattened
                output_dim=10,  # MNIST classes
                biological_constraints=self.biological_constraints
            )
        else:
            model = MultiNeuronPerceptron(
                self.morphology_data_list,
                input_dim=784,
                output_dim=10,
                biological_constraints=self.biological_constraints
            )
        
        return model
    
    def _load_mnist_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load and preprocess MNIST data
        """
        # MNIST preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784 dimensions
        ])
        
        # Load datasets
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        logger.info(f"Loaded MNIST: {len(train_dataset)} train, {len(test_dataset)} test samples")
        
        return train_loader, test_loader
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch using biological learning
        
        Returns:
            Dictionary with epoch training metrics
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Biological training step
            metrics = self.trainer.train_step(data, targets)
            
            epoch_loss += metrics['loss']
            epoch_accuracy += metrics['accuracy']
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{metrics['loss']:.4f}",
                'Acc': f"{metrics['accuracy']:.4f}"
            })
        
        # Calculate epoch averages
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, targets in tqdm(self.test_loader, desc="Evaluating"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                metrics = self.trainer.evaluate(data, targets)
                
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    def train(self, num_epochs: int = 10, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train biological perceptron on MNIST
        
        Args:
            num_epochs: Number of training epochs
            save_path: Path to save model and results
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Model: {type(self.model).__name__}")
        logger.info(f"Learning rule: {self.learning_rule}")
        
        best_test_accuracy = 0.0
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Evaluate
            test_metrics = self.evaluate()
            
            # Update history
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            self.training_history['test_loss'].append(test_metrics['loss'])
            self.training_history['test_accuracy'].append(test_metrics['accuracy'])
            
            # Log results
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
            
            # Save best model
            if test_metrics['accuracy'] > best_test_accuracy:
                best_test_accuracy = test_metrics['accuracy']
                if save_path:
                    self.save_model(save_path)
        
        # Final results
        final_results = {
            'training_history': self.training_history,
            'best_test_accuracy': best_test_accuracy,
            'final_train_accuracy': self.training_history['train_accuracy'][-1],
            'final_test_accuracy': self.training_history['test_accuracy'][-1],
            'model_info': {
                'num_neurons': len(self.morphology_data_list),
                'learning_rule': self.learning_rule,
                'biological_constraints': self.biological_constraints,
                'neuron_ids': [data['root_id'] for data in self.morphology_data_list]
            }
        }
        
        logger.info(f"\nTraining completed!")
        logger.info(f"Best test accuracy: {best_test_accuracy:.4f}")
        logger.info(f"Final test accuracy: {final_results['final_test_accuracy']:.4f}")
        
        return final_results
    
    def save_model(self, save_path: str):
        """
        Save trained model and metadata
        
        Args:
            save_path: Path to save model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'morphology_data': self.morphology_data_list,
            'training_history': self.training_history,
            'model_config': {
                'learning_rule': self.learning_rule,
                'biological_constraints': self.biological_constraints,
                'batch_size': self.batch_size
            }
        }
        
        torch.save(save_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """
        Load trained model
        
        Args:
            load_path: Path to load model from
        """
        save_data = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(save_data['model_state_dict'])
        self.training_history = save_data.get('training_history', self.training_history)
        
        logger.info(f"Model loaded from {load_path}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves
        
        Args:
            save_path: Path to save plot
        """
        if not self.training_history['epochs']:
            logger.warning("No training history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = self.training_history['epochs']
        
        # Loss curves
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.training_history['test_loss'], 'r-', label='Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training Curves - {self.learning_rule.upper()} Learning')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, self.training_history['train_accuracy'], 'b-', label='Train Accuracy')
        ax2.plot(epochs, self.training_history['test_accuracy'], 'r-', label='Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Accuracy Curves - {self.learning_rule.upper()} Learning')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def get_dendritic_analysis(self, sample_input: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze dendritic activations for a sample input
        
        Args:
            sample_input: Sample input tensor
            
        Returns:
            Dictionary with dendritic analysis
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get model output
            output = self.model(sample_input)
            
            # Get dendritic activations if available
            analysis = {
                'output': output.cpu().numpy(),
                'predicted_class': torch.argmax(output, dim=1).cpu().numpy()
            }
            
            # Add dendritic activations for single neuron models
            if hasattr(self.model, 'get_dendritic_activations'):
                dendritic_activations = self.model.get_dendritic_activations(sample_input)
                analysis['dendritic_activations'] = [
                    act.cpu().numpy() for act in dendritic_activations
                ]
            
            # Add morphology info
            if hasattr(self.model, 'get_morphology_info'):
                analysis['morphology_info'] = self.model.get_morphology_info()
            elif hasattr(self.model, 'get_all_morphology_info'):
                analysis['morphology_info'] = self.model.get_all_morphology_info()
        
        return analysis

def compare_learning_rules(morphology_data_list: List[Dict[str, Any]],
                          learning_rules: List[str] = ["hebbian", "stdp", "bcm"],
                          num_epochs: int = 5) -> Dict[str, Any]:
    """
    Compare different biological learning rules on MNIST
    
    Args:
        morphology_data_list: MicRons morphology data
        learning_rules: List of learning rules to compare
        num_epochs: Number of epochs for each rule
        
    Returns:
        Comparison results
    """
    results = {}
    
    for rule in learning_rules:
        logger.info(f"\nTraining with {rule} learning rule...")
        
        trainer = MNISTBiologicalTrainer(
            morphology_data_list,
            learning_rule=rule,
            batch_size=32,
            biological_constraints=True
        )
        
        # Train model
        rule_results = trainer.train(num_epochs=num_epochs)
        results[rule] = rule_results
        
        logger.info(f"{rule} - Final test accuracy: {rule_results['final_test_accuracy']:.4f}")
    
    return results

def main():
    """
    Test MNIST trainer
    """
    print("Testing MNIST Biological Trainer")
    print("=" * 40)
    
    print("MNIST trainer ready for real MicRons morphology data")
    print("Use after downloading and processing neurons")
    print("\nAvailable learning rules:")
    print("- hebbian: Hebbian learning")
    print("- stdp: Spike-timing dependent plasticity")
    print("- bcm: Bienenstock-Cooper-Munro")

if __name__ == "__main__":
    main()
