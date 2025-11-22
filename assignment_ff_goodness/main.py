import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from time import time

from backprop_baseline import train_backprop_baseline
from unsupervised_ff import train_unsupervised_ff
from supervised_ff import train_supervised_ff
from recurrent_ff import train_recurrent_ff
from utils import plot_training_curves


def run_experiments(models_to_run, epochs, batch_size, learning_rate, 
                  threshold, permutation_invariant, use_cuda):
    """
    Run selected experiments and compare their performance.
    
    Args:
        models_to_run: List of model names to run
        epochs: Dictionary of epochs per model
        batch_size: Batch size
        learning_rate: Learning rate
        threshold: Threshold for FF models
        permutation_invariant: Whether to use permutation-invariant MNIST
        use_cuda: Whether to use CUDA
        
    Returns:
        results: Dictionary of results per model
    """
    results = {}
    
    for model_name in models_to_run:
        print(f"\n{'-'*80}\nTraining {model_name}\n{'-'*80}")
        
        start_time = time()
        
        if model_name == 'backprop':
            # Run backpropagation baseline (Section 3.1)
            model, train_losses, val_accuracies, test_accuracy = train_backprop_baseline(
                epochs=epochs.get('backprop', 20),
                batch_size=batch_size,
                learning_rate=learning_rate,
                permutation_invariant=permutation_invariant,
                use_cuda=use_cuda
            )
            
        elif model_name == 'unsupervised_ff':
            # Run unsupervised FF (Section 3.2)
            model, train_losses, val_accuracies, test_accuracy = train_unsupervised_ff(
                epochs=epochs.get('unsupervised_ff', 100),
                batch_size=batch_size,
                learning_rates=[learning_rate] * 4,
                thresholds=[threshold] * 4,
                permutation_invariant=permutation_invariant,
                use_cuda=use_cuda
            )
            
        elif model_name == 'supervised_ff':
            # Run supervised FF with label embedding (Section 3.3)
            model, train_losses, val_accuracies, test_accuracy = train_supervised_ff(
                epochs=epochs.get('supervised_ff', 60),
                batch_size=batch_size,
                learning_rates=[learning_rate] * 4,
                thresholds=[threshold] * 4,
                permutation_invariant=permutation_invariant,
                use_cuda=use_cuda
            )
            
        elif model_name == 'recurrent_ff':
            # Run recurrent FF with top-down modeling (Section 3.4)
            model, train_losses, val_accuracies, test_accuracy = train_recurrent_ff(
                epochs=epochs.get('recurrent_ff', 60),
                batch_size=batch_size,
                learning_rates=[learning_rate] * 4,
                thresholds=[threshold] * 4,
                damping=0.3,
                iterations=8,
                permutation_invariant=permutation_invariant,
                use_cuda=use_cuda
            )
        
        end_time = time()
        training_time = end_time - start_time
        
        # Store results
        results[model_name] = {
            'model': model,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'test_accuracy': test_accuracy,
            'test_error': 100.0 - test_accuracy,
            'training_time': training_time
        }
        
        print(f"Finished training {model_name}")
        print(f"Test Accuracy: {test_accuracy:.2f}%, Test Error: {100.0-test_accuracy:.2f}%")
        print(f"Training Time: {training_time:.2f} seconds")
    
    return results


def compare_results(results):
    """
    Compare and visualize results from different models.
    
    Args:
        results: Dictionary of results per model
    """
    # Prepare data for comparison
    model_names = list(results.keys())
    test_accuracies = [results[name]['test_accuracy'] for name in model_names]
    test_errors = [results[name]['test_error'] for name in model_names]
    training_times = [results[name]['training_time'] for name in model_names]
    
    # Print comparison table
    print("\n" + "="*80)
    print("Model Comparison")
    print("="*80)
    print(f"{'Model':<15} {'Test Accuracy':<15} {'Test Error':<15} {'Training Time (s)':<20}")
    print("-"*80)
    
    for i, name in enumerate(model_names):
        print(f"{name:<15} {test_accuracies[i]:<15.2f} {test_errors[i]:<15.2f} {training_times[i]:<20.2f}")
    
    print("="*80)
    
    # Plot comparison
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Test error comparison
    axs[0].bar(model_names, test_errors)
    axs[0].set_ylabel('Test Error (%)')
    axs[0].set_title('Test Error Comparison')
    axs[0].grid(True, alpha=0.3)
    
    # Training time comparison
    axs[1].bar(model_names, training_times)
    axs[1].set_ylabel('Training Time (seconds)')
    axs[1].set_title('Training Time Comparison')
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    # Plot training curves for each model
    for name in model_names:
        plt.figure(figsize=(12, 5))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(results[name]['train_losses'])
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title(f'{name} - Training Loss')
        plt.grid(True)
        
        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(0, len(results[name]['val_accuracies']) * (len(results[name]['train_losses']) // len(results[name]['val_accuracies'])), 
                      len(results[name]['train_losses']) // len(results[name]['val_accuracies'])), 
                results[name]['val_accuracies'])
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy (%)')
        plt.title(f'{name} - Validation Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{name}_training.png')
        plt.show()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Forward-Forward Algorithm Experiments')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'backprop', 'unsupervised_ff', 'supervised_ff', 'recurrent_ff'],
                        help='Model to train (default: all)')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--threshold', type=float, default=2.0, help='Threshold for FF goodness function')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--permutation-invariant', action='store_true', default=True, 
                        help='Use permutation-invariant MNIST')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Run with fewer epochs for quick testing')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Set models to run
    if args.model == 'all':
        models_to_run = ['backprop', 'unsupervised_ff', 'supervised_ff', 'recurrent_ff']
    else:
        models_to_run = [args.model]
    
    # Set epochs based on fast flag
    if args.fast:
        epochs = {
            'backprop': 5,
            'unsupervised_ff': 10,
            'supervised_ff': 10,
            'recurrent_ff': 10
        }
    else:
        epochs = {
            'backprop': 25,
            'unsupervised_ff': 50,
            'supervised_ff': 25,
            'recurrent_ff': 25
        }
    
    # Run experiments
    results = run_experiments(
        models_to_run=models_to_run,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        threshold=args.threshold,
        permutation_invariant=args.permutation_invariant,
        use_cuda=use_cuda
    )
    
    # Compare results
    if len(models_to_run) > 1:
        compare_results(results)
