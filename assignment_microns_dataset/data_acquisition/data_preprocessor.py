"""
Data Preprocessor for MicRons Neuronal Morphology
Converts raw MicRons data into structured format for biological perceptron modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import networkx as nx

logger = logging.getLogger(__name__)

class MicRonsDataPreprocessor:
    """
    Preprocesses MicRons neuronal data for biological perceptron modeling
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def process_neuron_morphology(self, morphology_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw morphology data into structured format for perceptron modeling
        
        Args:
            morphology_data: Raw morphology data from MicRons
            
        Returns:
            Processed morphology data with dendritic structure
        """
        root_id = morphology_data['root_id']
        logger.info(f"Processing morphology for neuron {root_id}")
        
        # Extract and process skeleton data
        skeleton_data = self._process_skeleton(
            morphology_data.get('skeleton_vertices'),
            morphology_data.get('skeleton_edges')
        )
        
        # Process synaptic sites
        synaptic_data = self._process_synapses(
            morphology_data.get('presynaptic_sites'),
            morphology_data.get('postsynaptic_sites')
        )
        
        # Create dendritic structure
        dendritic_structure = self._create_dendritic_structure(
            skeleton_data, synaptic_data
        )
        
        processed_data = {
            'root_id': root_id,
            'skeleton_graph': skeleton_data['graph'],
            'dendrite_segments': dendritic_structure['segments'],
            'synaptic_inputs': synaptic_data['inputs'],
            'synaptic_outputs': synaptic_data['outputs'],
            'morphology_features': self._extract_morphology_features(skeleton_data),
            'connectivity_matrix': dendritic_structure['connectivity'],
            'spatial_coordinates': skeleton_data['coordinates']
        }
        
        return processed_data
    
    def _process_skeleton(self, vertices: Optional[np.ndarray], 
                         edges: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Process skeleton vertices and edges into graph structure
        """
        if vertices is None or edges is None:
            logger.warning("No skeleton data available")
            return {'graph': nx.Graph(), 'coordinates': np.array([])}
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add vertices as nodes
        for i, vertex in enumerate(vertices):
            G.add_node(i, pos=vertex, coordinates=vertex)
        
        # Add edges
        if edges is not None:
            G.add_edges_from(edges)
        
        return {
            'graph': G,
            'coordinates': vertices,
            'edges': edges if edges is not None else np.array([])
        }
    
    def _process_synapses(self, pre_synapses: Optional[pd.DataFrame], 
                         post_synapses: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Process presynaptic and postsynaptic sites
        """
        synaptic_inputs = []
        synaptic_outputs = []
        
        # Process postsynaptic sites (inputs to this neuron)
        if post_synapses is not None and not post_synapses.empty:
            for _, synapse in post_synapses.iterrows():
                synaptic_inputs.append({
                    'position': np.array([synapse.get('ctr_pt_position_x', 0),
                                        synapse.get('ctr_pt_position_y', 0),
                                        synapse.get('ctr_pt_position_z', 0)]),
                    'pre_neuron_id': synapse.get('pre_pt_root_id'),
                    'synapse_id': synapse.get('id'),
                    'size': synapse.get('size', 1.0)
                })
        
        # Process presynaptic sites (outputs from this neuron)
        if pre_synapses is not None and not pre_synapses.empty:
            for _, synapse in pre_synapses.iterrows():
                synaptic_outputs.append({
                    'position': np.array([synapse.get('ctr_pt_position_x', 0),
                                        synapse.get('ctr_pt_position_y', 0),
                                        synapse.get('ctr_pt_position_z', 0)]),
                    'post_neuron_id': synapse.get('post_pt_root_id'),
                    'synapse_id': synapse.get('id'),
                    'size': synapse.get('size', 1.0)
                })
        
        return {
            'inputs': synaptic_inputs,
            'outputs': synaptic_outputs
        }
    
    def _create_dendritic_structure(self, skeleton_data: Dict[str, Any], 
                                   synaptic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create dendritic structure by mapping synapses to skeleton segments
        """
        G = skeleton_data['graph']
        coordinates = skeleton_data['coordinates']
        synaptic_inputs = synaptic_data['inputs']
        
        if len(coordinates) == 0 or len(synaptic_inputs) == 0:
            return {'segments': [], 'connectivity': np.array([])}
        
        # Find dendritic segments (paths between branch points)
        segments = self._identify_dendritic_segments(G)
        
        # Map synapses to nearest segments
        segment_synapses = self._map_synapses_to_segments(
            segments, coordinates, synaptic_inputs
        )
        
        # Create connectivity matrix
        connectivity_matrix = self._create_connectivity_matrix(segments, G)
        
        return {
            'segments': segment_synapses,
            'connectivity': connectivity_matrix
        }
    
    def _identify_dendritic_segments(self, G: nx.Graph) -> List[Dict[str, Any]]:
        """
        Identify dendritic segments as paths between branch points
        """
        if G.number_of_nodes() == 0:
            return []
        
        # Find branch points (degree > 2) and endpoints (degree == 1)
        branch_points = [n for n in G.nodes() if G.degree(n) > 2]
        endpoints = [n for n in G.nodes() if G.degree(n) == 1]
        special_points = set(branch_points + endpoints)
        
        segments = []
        visited_edges = set()
        
        # Find paths between special points
        for start_node in special_points:
            for neighbor in G.neighbors(start_node):
                edge = tuple(sorted([start_node, neighbor]))
                if edge in visited_edges:
                    continue
                
                # Trace path until we hit another special point
                path = [start_node, neighbor]
                current = neighbor
                
                while current not in special_points or current == neighbor:
                    visited_edges.add(tuple(sorted([path[-2], path[-1]])))
                    
                    # Find next node (not the previous one)
                    next_nodes = [n for n in G.neighbors(current) if n != path[-2]]
                    if not next_nodes:
                        break
                    
                    current = next_nodes[0]
                    path.append(current)
                    
                    if current in special_points:
                        break
                
                if len(path) > 1:
                    segments.append({
                        'path': path,
                        'length': len(path) - 1,
                        'start_node': path[0],
                        'end_node': path[-1],
                        'synapses': []
                    })
        
        return segments
    
    def _map_synapses_to_segments(self, segments: List[Dict[str, Any]], 
                                 coordinates: np.ndarray, 
                                 synaptic_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map synaptic inputs to nearest dendritic segments
        """
        if not segments or not synaptic_inputs:
            return segments
        
        for synapse in synaptic_inputs:
            synapse_pos = synapse['position']
            best_segment_idx = 0
            min_distance = float('inf')
            
            # Find closest segment
            for seg_idx, segment in enumerate(segments):
                # Calculate distance to segment path
                segment_coords = coordinates[segment['path']]
                distances = cdist([synapse_pos], segment_coords)[0]
                min_seg_distance = np.min(distances)
                
                if min_seg_distance < min_distance:
                    min_distance = min_seg_distance
                    best_segment_idx = seg_idx
            
            # Add synapse to best segment
            segments[best_segment_idx]['synapses'].append(synapse)
        
        return segments
    
    def _create_connectivity_matrix(self, segments: List[Dict[str, Any]], 
                                   G: nx.Graph) -> np.ndarray:
        """
        Create connectivity matrix between segments
        """
        n_segments = len(segments)
        if n_segments == 0:
            return np.array([])
        
        connectivity = np.zeros((n_segments, n_segments))
        
        for i, seg1 in enumerate(segments):
            for j, seg2 in enumerate(segments):
                if i != j:
                    # Check if segments are connected
                    if (seg1['end_node'] == seg2['start_node'] or 
                        seg1['start_node'] == seg2['end_node'] or
                        seg1['end_node'] == seg2['end_node'] or
                        seg1['start_node'] == seg2['start_node']):
                        connectivity[i, j] = 1
        
        return connectivity
    
    def _extract_morphology_features(self, skeleton_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract morphological features for characterizing the neuron
        """
        G = skeleton_data['graph']
        coordinates = skeleton_data['coordinates']
        
        if G.number_of_nodes() == 0:
            return {'total_length': 0, 'branch_points': 0, 'endpoints': 0, 'complexity': 0}
        
        # Basic morphological features
        total_length = G.number_of_edges()
        branch_points = len([n for n in G.nodes() if G.degree(n) > 2])
        endpoints = len([n for n in G.nodes() if G.degree(n) == 1])
        
        # Complexity measure
        complexity = branch_points / max(1, total_length) if total_length > 0 else 0
        
        # Spatial extent
        if len(coordinates) > 0:
            spatial_extent = np.ptp(coordinates, axis=0).mean()
        else:
            spatial_extent = 0
        
        return {
            'total_length': float(total_length),
            'branch_points': float(branch_points),
            'endpoints': float(endpoints),
            'complexity': complexity,
            'spatial_extent': spatial_extent
        }
    
    def create_perceptron_input_mapping(self, processed_neurons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create input mapping for biological perceptron models
        
        Args:
            processed_neurons: List of processed neuron morphologies
            
        Returns:
            Input mapping configuration for perceptron models
        """
        input_mappings = []
        
        for neuron in processed_neurons:
            # Create input channels based on dendritic segments
            segments = neuron['dendrite_segments']
            input_channels = []
            
            for i, segment in enumerate(segments):
                channel = {
                    'segment_id': i,
                    'num_synapses': len(segment['synapses']),
                    'segment_length': segment['length'],
                    'spatial_weight': 1.0 / (segment['length'] + 1),  # Closer to soma = higher weight
                    'synaptic_positions': [syn['position'] for syn in segment['synapses']]
                }
                input_channels.append(channel)
            
            input_mappings.append({
                'neuron_id': neuron['root_id'],
                'input_channels': input_channels,
                'total_inputs': sum(len(seg['synapses']) for seg in segments),
                'morphology_features': neuron['morphology_features']
            })
        
        return {
            'neuron_mappings': input_mappings,
            'total_neurons': len(processed_neurons),
            'input_dimension': max(len(mapping['input_channels']) 
                                 for mapping in input_mappings) if input_mappings else 0
        }

def main():
    """
    Test the data preprocessor
    """
    print("Testing MicRons Data Preprocessor")
    print("=" * 40)
    
    # This would normally be called after downloading real MicRons data
    print("Preprocessor ready for real MicRons data processing")
    print("Use after downloading neurons with microns_downloader.py")

if __name__ == "__main__":
    main()
