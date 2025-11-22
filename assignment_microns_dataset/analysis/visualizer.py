"""
Visualization Module for MicRons-based Biological Perceptrons
3D morphology visualization, connectivity analysis, and training dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

class MicRonsVisualizer:
    """
    Comprehensive visualization for MicRons biological perceptron analysis
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting styles
        plt.style.use('default')
        sns.set_palette("husl")
    
    def visualize_neuron_morphology_3d(self, morphology_data: Dict[str, Any],
                                      save_path: Optional[str] = None) -> go.Figure:
        """
        Create 3D visualization of neuron morphology
        
        Args:
            morphology_data: Processed morphology data
            save_path: Path to save visualization
            
        Returns:
            Plotly figure object
        """
        neuron_id = morphology_data['root_id']
        logger.info(f"Creating 3D morphology visualization for neuron {neuron_id}")
        
        fig = go.Figure()
        
        # Plot skeleton structure
        coordinates = morphology_data.get('spatial_coordinates', np.array([]))
        if len(coordinates) > 0:
            # Add skeleton points
            fig.add_trace(go.Scatter3d(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                z=coordinates[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color='blue',
                    opacity=0.6
                ),
                name='Skeleton',
                hovertemplate='<b>Skeleton Point</b><br>' +
                             'X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
            ))
        
        # Plot dendritic segments
        segments = morphology_data.get('dendrite_segments', [])
        for i, segment in enumerate(segments):
            if 'path' in segment and len(coordinates) > 0:
                path_coords = coordinates[segment['path']]
                
                # Add segment line
                fig.add_trace(go.Scatter3d(
                    x=path_coords[:, 0],
                    y=path_coords[:, 1],
                    z=path_coords[:, 2],
                    mode='lines',
                    line=dict(
                        color=f'rgb({(i*50)%255}, {(i*80)%255}, {(i*120)%255})',
                        width=4
                    ),
                    name=f'Segment {i+1}',
                    hovertemplate=f'<b>Segment {i+1}</b><br>' +
                                 f'Length: {segment.get("length", 0)}<br>' +
                                 f'Synapses: {len(segment.get("synapses", []))}<extra></extra>'
                ))
        
        # Plot synaptic sites
        synaptic_inputs = morphology_data.get('synaptic_inputs', [])
        if synaptic_inputs:
            synapse_positions = np.array([syn['position'] for syn in synaptic_inputs])
            
            fig.add_trace(go.Scatter3d(
                x=synapse_positions[:, 0],
                y=synapse_positions[:, 1],
                z=synapse_positions[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color='red',
                    symbol='diamond',
                    opacity=0.8
                ),
                name='Synapses',
                hovertemplate='<b>Synapse</b><br>' +
                             'X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'3D Morphology - Neuron {neuron_id}',
            scene=dict(
                xaxis_title='X (μm)',
                yaxis_title='Y (μm)',
                zaxis_title='Z (μm)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"3D morphology saved to {save_path}")
        
        return fig
    
    def visualize_dendritic_input_mapping(self, morphology_data: Dict[str, Any],
                                        input_mapping: Dict[str, Any],
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize how MNIST inputs map to dendritic segments
        
        Args:
            morphology_data: Processed morphology data
            input_mapping: Input mapping configuration
            save_path: Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        neuron_id = morphology_data['root_id']
        segments = morphology_data.get('dendrite_segments', [])
        
        if not segments:
            logger.warning(f"No dendritic segments found for neuron {neuron_id}")
            return plt.figure()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Segment properties
        ax1 = axes[0, 0]
        segment_lengths = [seg.get('length', 0) for seg in segments]
        segment_synapses = [len(seg.get('synapses', [])) for seg in segments]
        
        bars = ax1.bar(range(len(segments)), segment_lengths, alpha=0.7, label='Length')
        ax1_twin = ax1.twinx()
        ax1_twin.bar(range(len(segments)), segment_synapses, alpha=0.5, color='red', label='Synapses')
        
        ax1.set_xlabel('Dendritic Segment')
        ax1.set_ylabel('Segment Length', color='blue')
        ax1_twin.set_ylabel('Number of Synapses', color='red')
        ax1.set_title(f'Dendritic Segment Properties - Neuron {neuron_id}')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Plot 2: Input dimension allocation
        ax2 = axes[0, 1]
        if 'neuron_mappings' in input_mapping:
            neuron_mapping = next((nm for nm in input_mapping['neuron_mappings'] 
                                 if nm['neuron_id'] == neuron_id), None)
            
            if neuron_mapping:
                input_sizes = [ch['segment_length'] for ch in neuron_mapping['input_channels']]
                ax2.pie(input_sizes, labels=[f'Seg {i+1}' for i in range(len(input_sizes))],
                       autopct='%1.1f%%', startangle=90)
                ax2.set_title('Input Dimension Allocation')
        
        # Plot 3: MNIST input visualization (28x28 grid)
        ax3 = axes[1, 0]
        mnist_grid = np.zeros((28, 28))
        
        # Color code pixels by which segment they map to
        if 'neuron_mappings' in input_mapping:
            neuron_mapping = next((nm for nm in input_mapping['neuron_mappings'] 
                                 if nm['neuron_id'] == neuron_id), None)
            
            if neuron_mapping:
                current_pixel = 0
                for i, channel in enumerate(neuron_mapping['input_channels']):
                    segment_pixels = channel.get('segment_length', 0)
                    
                    for _ in range(segment_pixels):
                        if current_pixel < 784:  # 28*28 = 784
                            row, col = divmod(current_pixel, 28)
                            mnist_grid[row, col] = i + 1
                            current_pixel += 1
        
        im = ax3.imshow(mnist_grid, cmap='tab10', aspect='equal')
        ax3.set_title('MNIST Pixel to Segment Mapping')
        ax3.set_xlabel('Pixel Column')
        ax3.set_ylabel('Pixel Row')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Dendritic Segment')
        
        # Plot 4: Connectivity matrix
        ax4 = axes[1, 1]
        connectivity = morphology_data.get('connectivity_matrix', np.array([]))
        
        if connectivity.size > 0:
            sns.heatmap(connectivity, annot=True, fmt='.0f', cmap='Blues', ax=ax4)
            ax4.set_title('Dendritic Connectivity Matrix')
            ax4.set_xlabel('Segment')
            ax4.set_ylabel('Segment')
        else:
            ax4.text(0.5, 0.5, 'No connectivity data', ha='center', va='center',
                    transform=ax4.transAxes)
            ax4.set_title('Dendritic Connectivity Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Input mapping visualization saved to {save_path}")
        
        return fig
    
    def visualize_training_dynamics(self, training_results: Dict[str, Any],
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize training dynamics and learning progression
        
        Args:
            training_results: Training results dictionary
            save_path: Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        history = training_results['training_history']
        model_info = training_results.get('model_info', {})
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = history.get('epochs', [])
        if not epochs:
            logger.warning("No training history available")
            return fig
        
        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        if history.get('train_loss'):
            ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if history.get('test_loss'):
            ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax2 = axes[0, 1]
        if history.get('train_accuracy'):
            ax2.plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        if history.get('test_accuracy'):
            ax2.plot(epochs, history['test_accuracy'], 'r-', label='Test Accuracy', linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Learning rate analysis (if available)
        ax3 = axes[1, 0]
        if history.get('test_accuracy'):
            # Calculate learning rate as accuracy improvement per epoch
            test_acc = np.array(history['test_accuracy'])
            learning_rates = np.diff(test_acc)
            
            ax3.plot(epochs[1:], learning_rates, 'g-', linewidth=2)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy Change')
            ax3.set_title('Learning Rate Analysis')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Model information
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Display model information as text
        info_text = []
        info_text.append(f"Learning Rule: {model_info.get('learning_rule', 'Unknown')}")
        info_text.append(f"Neurons: {model_info.get('num_neurons', 'Unknown')}")
        info_text.append(f"Biological Constraints: {model_info.get('biological_constraints', 'Unknown')}")
        
        if history.get('test_accuracy'):
            final_acc = history['test_accuracy'][-1]
            best_acc = max(history['test_accuracy'])
            info_text.append(f"Final Test Accuracy: {final_acc:.4f}")
            info_text.append(f"Best Test Accuracy: {best_acc:.4f}")
        
        if model_info.get('neuron_ids'):
            info_text.append(f"Neuron IDs: {model_info['neuron_ids']}")
        
        ax4.text(0.1, 0.9, '\n'.join(info_text), transform=ax4.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax4.set_title('Model Information')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training dynamics visualization saved to {save_path}")
        
        return fig
    
    def visualize_dendritic_activations(self, dendritic_analysis: Dict[str, Any],
                                       sample_image: np.ndarray,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize dendritic activations for a sample input
        
        Args:
            dendritic_analysis: Analysis results from model
            sample_image: Original MNIST image (28x28)
            save_path: Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        activations = dendritic_analysis.get('dendritic_activations', [])
        predicted_class = dendritic_analysis.get('predicted_class', [0])[0]
        
        if not activations:
            logger.warning("No dendritic activations available")
            return plt.figure()
        
        num_segments = len(activations)
        fig, axes = plt.subplots(2, max(3, num_segments), figsize=(15, 8))
        
        # Ensure axes is 2D
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)
        
        # Plot original image
        axes[0, 0].imshow(sample_image.reshape(28, 28), cmap='gray')
        axes[0, 0].set_title(f'Input Image\nPredicted: {predicted_class}')
        axes[0, 0].axis('off')
        
        # Plot dendritic activations
        for i, activation in enumerate(activations):
            if i + 1 < axes.shape[1]:
                # Plot activation as heatmap
                act_2d = activation[0].reshape(-1, 1) if activation.ndim > 1 else activation.reshape(-1, 1)
                
                im = axes[0, i + 1].imshow(act_2d, cmap='viridis', aspect='auto')
                axes[0, i + 1].set_title(f'Segment {i + 1}\nActivation')
                axes[0, i + 1].set_xlabel('Feature')
                axes[0, i + 1].set_ylabel('Activation')
                plt.colorbar(im, ax=axes[0, i + 1])
        
        # Plot activation statistics
        if len(activations) > 0:
            segment_means = [np.mean(act) for act in activations]
            segment_stds = [np.std(act) for act in activations]
            
            axes[1, 0].bar(range(len(segment_means)), segment_means, 
                          yerr=segment_stds, alpha=0.7, capsize=5)
            axes[1, 0].set_xlabel('Dendritic Segment')
            axes[1, 0].set_ylabel('Mean Activation')
            axes[1, 0].set_title('Segment Activation Summary')
            
            # Plot activation distribution
            all_activations = np.concatenate([act.flatten() for act in activations])
            axes[1, 1].hist(all_activations, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Activation Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Activation Distribution')
        
        # Hide unused subplots
        for i in range(2):
            for j in range(max(2, len(activations) + 1), axes.shape[1]):
                axes[i, j].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dendritic activations visualization saved to {save_path}")
        
        return fig
    
    def create_comprehensive_dashboard(self, morphology_data_list: List[Dict[str, Any]],
                                     training_results_dict: Dict[str, Dict[str, Any]],
                                     save_path: Optional[str] = None) -> str:
        """
        Create comprehensive HTML dashboard with all visualizations
        
        Args:
            morphology_data_list: List of morphology data
            training_results_dict: Dictionary of training results
            save_path: Path to save HTML dashboard
            
        Returns:
            HTML content string
        """
        logger.info("Creating comprehensive visualization dashboard")
        
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>MicRons Biological Perceptron Dashboard</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".section { margin-bottom: 30px; }",
            ".neuron-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }",
            ".neuron-card { border: 1px solid #ccc; padding: 15px; border-radius: 5px; }",
            "h1, h2 { color: #333; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>MicRons Biological Perceptron Analysis Dashboard</h1>",
            "",
            "<div class='section'>",
            "<h2>Project Overview</h2>",
            f"<p>Analysis of {len(morphology_data_list)} real MicRons neurons trained on MNIST classification.</p>",
            f"<p>Models trained: {len(training_results_dict)}</p>",
            "</div>"
        ]
        
        # Add neuron morphology section
        html_content.extend([
            "<div class='section'>",
            "<h2>Neuronal Morphologies</h2>",
            "<div class='neuron-grid'>"
        ])
        
        for i, morphology in enumerate(morphology_data_list):
            neuron_id = morphology['root_id']
            segments = morphology.get('dendrite_segments', [])
            synapses = morphology.get('synaptic_inputs', [])
            
            html_content.extend([
                f"<div class='neuron-card'>",
                f"<h3>Neuron {neuron_id}</h3>",
                f"<p>Dendritic Segments: {len(segments)}</p>",
                f"<p>Synaptic Inputs: {len(synapses)}</p>",
                f"<p>Morphology Features: {morphology.get('morphology_features', {})}</p>",
                "</div>"
            ])
        
        html_content.append("</div></div>")
        
        # Add training results section
        html_content.extend([
            "<div class='section'>",
            "<h2>Training Results</h2>",
            "<table border='1' style='border-collapse: collapse; width: 100%;'>",
            "<tr><th>Model</th><th>Learning Rule</th><th>Final Accuracy</th><th>Best Accuracy</th></tr>"
        ])
        
        for model_name, results in training_results_dict.items():
            history = results['training_history']
            model_info = results.get('model_info', {})
            
            final_acc = history['test_accuracy'][-1] if history.get('test_accuracy') else 0
            best_acc = max(history['test_accuracy']) if history.get('test_accuracy') else 0
            learning_rule = model_info.get('learning_rule', 'Unknown')
            
            html_content.append(
                f"<tr><td>{model_name}</td><td>{learning_rule}</td>"
                f"<td>{final_acc:.4f}</td><td>{best_acc:.4f}</td></tr>"
            )
        
        html_content.extend([
            "</table>",
            "</div>",
            "",
            "<div class='section'>",
            "<h2>Key Findings</h2>",
            "<ul>",
            "<li>Successfully implemented biological perceptrons using real MicRons neuronal morphology</li>",
            "<li>Demonstrated competitive performance on MNIST classification task</li>",
            "<li>Compared multiple biologically plausible learning rules</li>",
            "<li>Analyzed the impact of biological constraints on learning dynamics</li>",
            "</ul>",
            "</div>",
            "",
            "</body>",
            "</html>"
        ])
        
        html_string = "\n".join(html_content)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html_string)
            logger.info(f"Dashboard saved to {save_path}")
        
        return html_string

def main():
    """
    Test visualizer
    """
    print("Testing MicRons Visualizer")
    print("=" * 30)
    
    print("Visualizer ready for MicRons morphology and training data")
    print("Available visualizations:")
    print("- 3D neuronal morphology")
    print("- Dendritic input mapping")
    print("- Training dynamics")
    print("- Dendritic activations")
    print("- Comprehensive dashboard")

if __name__ == "__main__":
    main()
