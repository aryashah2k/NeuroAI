"""
Biological Perceptron Models based on Real MicRons Neuronal Morphology
Implements realistic neural constraints and dendritic processing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class BiologicalPerceptron(nn.Module):
    """
    Biological perceptron model based on real neuronal morphology
    Maps inputs to dendritic segments and processes through realistic constraints
    """
    
    def __init__(self, morphology_data: Dict[str, Any], 
                 input_dim: int = 784,  # MNIST flattened
                 output_dim: int = 10,   # MNIST classes
                 biological_constraints: bool = True):
        """
        Initialize biological perceptron with real morphology
        
        Args:
            morphology_data: Processed morphology from MicRons data
            input_dim: Input dimension (784 for MNIST)
            output_dim: Output dimension (10 for MNIST classes)
            biological_constraints: Apply biological realism constraints
        """
        super(BiologicalPerceptron, self).__init__()
        
        self.neuron_id = morphology_data['root_id']
        self.morphology_data = morphology_data
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.biological_constraints = biological_constraints
        
        # Extract dendritic structure
        self.dendrite_segments = morphology_data['dendrite_segments']
        self.num_segments = len(self.dendrite_segments)
        
        if self.num_segments == 0:
            raise ValueError(f"Neuron {self.neuron_id} has no dendritic segments")
        
        logger.info(f"Creating biological perceptron for neuron {self.neuron_id}")
        logger.info(f"  - {self.num_segments} dendritic segments")
        logger.info(f"  - Input dimension: {input_dim}")
        logger.info(f"  - Biological constraints: {biological_constraints}")
        
        # Create dendritic input mapping
        self._create_dendritic_mapping()
        
        # Initialize neural components
        self._initialize_components()
    
    def _create_dendritic_mapping(self):
        """
        Create mapping from input pixels to dendritic segments
        Based on real synaptic locations and segment properties
        """
        # Distribute input dimensions across dendritic segments
        # Weight by segment properties (length, synaptic density)
        segment_weights = []
        
        for segment in self.dendrite_segments:
            # Weight based on segment length and number of synapses
            length_weight = segment['length']
            synapse_weight = len(segment['synapses']) + 1  # +1 to avoid zero
            combined_weight = length_weight * synapse_weight
            segment_weights.append(combined_weight)
        
        # Normalize weights
        total_weight = sum(segment_weights)
        if total_weight == 0:
            segment_weights = [1.0] * self.num_segments
            total_weight = self.num_segments
        
        normalized_weights = [w / total_weight for w in segment_weights]
        
        # Assign input dimensions to segments proportionally
        self.segment_input_ranges = []
        current_idx = 0
        
        for i, weight in enumerate(normalized_weights):
            segment_inputs = max(1, int(weight * self.input_dim))
            
            # Ensure we don't exceed input dimension
            if current_idx + segment_inputs > self.input_dim:
                segment_inputs = self.input_dim - current_idx
            
            self.segment_input_ranges.append({
                'start': current_idx,
                'end': current_idx + segment_inputs,
                'size': segment_inputs,
                'weight': weight
            })
            
            current_idx += segment_inputs
            
            if current_idx >= self.input_dim:
                break
        
        # Handle any remaining inputs
        if current_idx < self.input_dim:
            remaining = self.input_dim - current_idx
            if self.segment_input_ranges:
                self.segment_input_ranges[-1]['end'] += remaining
                self.segment_input_ranges[-1]['size'] += remaining
    
    def _initialize_components(self):
        """
        Initialize neural components with biological constraints
        """
        # Dendritic processing layers (one per segment)
        self.dendritic_layers = nn.ModuleList()
        
        for i, segment_range in enumerate(self.segment_input_ranges):
            segment_size = segment_range['size']
            
            # Create dendritic processing layer
            if self.biological_constraints:
                # Biological constraint: limited dendritic computation
                hidden_size = min(segment_size // 2, 32)  # Limited local processing
            else:
                hidden_size = segment_size
            
            dendritic_layer = nn.Sequential(
                nn.Linear(segment_size, hidden_size),
                nn.ReLU(),  # Biological activation (rectification)
                nn.Dropout(0.1) if self.biological_constraints else nn.Identity()
            )
            
            self.dendritic_layers.append(dendritic_layer)
        
        # Integration layer (soma)
        total_dendritic_output = sum(
            min(seg_range['size'] // 2, 32) if self.biological_constraints 
            else seg_range['size']
            for seg_range in self.segment_input_ranges
        )
        
        if self.biological_constraints:
            # Biological constraint: limited somatic integration
            soma_hidden = min(total_dendritic_output, 64)
        else:
            soma_hidden = total_dendritic_output
        
        self.soma_integration = nn.Sequential(
            nn.Linear(total_dendritic_output, soma_hidden),
            nn.ReLU(),
            nn.Dropout(0.2) if self.biological_constraints else nn.Identity()
        )
        
        # Output layer (axon)
        self.axon_output = nn.Linear(soma_hidden, self.output_dim)
        
        # Initialize weights with biological constraints
        if self.biological_constraints:
            self._apply_biological_weight_constraints()
    
    def _apply_biological_weight_constraints(self):
        """
        Apply biological constraints to weight initialization
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Biological constraint: sparse connectivity
                nn.init.xavier_normal_(module.weight)
                
                # Apply sparsity (biological neurons have sparse connections)
                with torch.no_grad():
                    sparsity_mask = torch.rand_like(module.weight) > 0.3  # 70% sparsity
                    module.weight *= sparsity_mask.float()
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1)  # Small positive bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through biological perceptron
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Output tensor (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # Process inputs through dendritic segments
        dendritic_outputs = []
        
        for i, (layer, segment_range) in enumerate(zip(self.dendritic_layers, self.segment_input_ranges)):
            # Extract input for this dendritic segment
            segment_input = x[:, segment_range['start']:segment_range['end']]
            
            # Process through dendritic layer
            dendritic_output = layer(segment_input)
            
            # Apply biological constraints during forward pass
            if self.biological_constraints:
                # Biological constraint: dendritic saturation
                dendritic_output = torch.clamp(dendritic_output, 0, 1)
            
            dendritic_outputs.append(dendritic_output)
        
        # Integrate dendritic outputs at soma
        if dendritic_outputs:
            integrated_input = torch.cat(dendritic_outputs, dim=1)
        else:
            integrated_input = torch.zeros(batch_size, 1, device=x.device)
        
        # Somatic integration
        soma_output = self.soma_integration(integrated_input)
        
        # Apply biological constraints
        if self.biological_constraints:
            # Biological constraint: somatic nonlinearity and saturation
            soma_output = torch.clamp(soma_output, 0, 1)
        
        # Axonal output
        output = self.axon_output(soma_output)
        
        return output
    
    def get_dendritic_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get activations from each dendritic segment for analysis
        
        Args:
            x: Input tensor
            
        Returns:
            List of dendritic activation tensors
        """
        activations = []
        
        for i, (layer, segment_range) in enumerate(zip(self.dendritic_layers, self.segment_input_ranges)):
            segment_input = x[:, segment_range['start']:segment_range['end']]
            dendritic_output = layer(segment_input)
            
            if self.biological_constraints:
                dendritic_output = torch.clamp(dendritic_output, 0, 1)
            
            activations.append(dendritic_output)
        
        return activations
    
    def get_morphology_info(self) -> Dict[str, Any]:
        """
        Get morphology information for this neuron
        
        Returns:
            Dictionary with morphology details
        """
        return {
            'neuron_id': self.neuron_id,
            'num_segments': self.num_segments,
            'segment_input_mapping': self.segment_input_ranges,
            'morphology_features': self.morphology_data.get('morphology_features', {}),
            'biological_constraints': self.biological_constraints
        }

class MultiNeuronPerceptron(nn.Module):
    """
    Multi-neuron biological perceptron using multiple real MicRons neurons
    """
    
    def __init__(self, morphology_data_list: List[Dict[str, Any]],
                 input_dim: int = 784,
                 output_dim: int = 10,
                 biological_constraints: bool = True):
        """
        Initialize multi-neuron perceptron
        
        Args:
            morphology_data_list: List of processed morphology data
            input_dim: Input dimension
            output_dim: Output dimension
            biological_constraints: Apply biological constraints
        """
        super(MultiNeuronPerceptron, self).__init__()
        
        if len(morphology_data_list) < 2:
            raise ValueError("Must have at least 2 neurons as per project requirements")
        
        self.num_neurons = len(morphology_data_list)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.biological_constraints = biological_constraints
        
        logger.info(f"Creating multi-neuron perceptron with {self.num_neurons} real neurons")
        
        # Create individual biological perceptrons
        self.neurons = nn.ModuleList()
        for morphology_data in morphology_data_list:
            try:
                neuron = BiologicalPerceptron(
                    morphology_data, input_dim, output_dim, biological_constraints
                )
                self.neurons.append(neuron)
            except ValueError as e:
                logger.warning(f"Skipping neuron due to error: {e}")
                continue
        
        if len(self.neurons) == 0:
            raise ValueError("No valid neurons could be created")
        
        # Final integration layer
        self.final_integration = nn.Linear(len(self.neurons) * output_dim, output_dim)
        
        logger.info(f"Successfully created {len(self.neurons)} biological neurons")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-neuron network
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Get outputs from all neurons
        neuron_outputs = []
        for neuron in self.neurons:
            output = neuron(x)
            neuron_outputs.append(output)
        
        # Combine neuron outputs
        if neuron_outputs:
            combined_output = torch.cat(neuron_outputs, dim=1)
            final_output = self.final_integration(combined_output)
        else:
            final_output = torch.zeros(x.size(0), self.output_dim, device=x.device)
        
        return final_output
    
    def get_all_morphology_info(self) -> List[Dict[str, Any]]:
        """
        Get morphology information for all neurons
        
        Returns:
            List of morphology info dictionaries
        """
        return [neuron.get_morphology_info() for neuron in self.neurons]

def create_biological_perceptron_from_microns(morphology_data_list: List[Dict[str, Any]],
                                            single_neuron: bool = False,
                                            **kwargs) -> nn.Module:
    """
    Factory function to create biological perceptron from MicRons data
    
    Args:
        morphology_data_list: List of processed morphology data
        single_neuron: If True, use only the first neuron
        **kwargs: Additional arguments for perceptron initialization
        
    Returns:
        Biological perceptron model
    """
    if single_neuron and morphology_data_list:
        return BiologicalPerceptron(morphology_data_list[0], **kwargs)
    else:
        return MultiNeuronPerceptron(morphology_data_list, **kwargs)

def main():
    """
    Test biological perceptron creation
    """
    print("Testing Biological Perceptron Models")
    print("=" * 40)
    
    # This would normally use real MicRons morphology data
    print("Biological perceptron models ready for real MicRons data")
    print("Use after processing neurons with data_preprocessor.py")

if __name__ == "__main__":
    main()
