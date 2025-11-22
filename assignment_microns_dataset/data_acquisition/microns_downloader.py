"""
MicRons Data Downloader - STRICT AUTHENTICATION REQUIRED
This module handles downloading and caching of real MicRons neuronal data.
NO MOCK DATA OR FALLBACKS - Real data only as per academic requirements.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from caveclient import CAVEclient
import cloudvolume
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MicRonsDownloader:
    """
    Strict MicRons data downloader - REAL DATA ONLY
    Requires proper CAVEclient authentication with no fallbacks
    """
    
    def __init__(self, datastack: str = 'minnie65_public'):
        """
        Initialize MicRons downloader with strict authentication
        
        Args:
            datastack: MicRons datastack name (default: minnie65_public)
        """
        self.datastack = datastack
        self.client = None
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize client with strict error handling
        self._initialize_client()
    
    def _initialize_client(self):
        """
        Initialize CAVEclient with strict authentication requirements
        NO FALLBACKS - Must have real authentication
        """
        try:
            logger.info("Initializing CAVEclient for MicRons data access...")
            self.client = CAVEclient(self.datastack)
            
            # Test authentication
            logger.info("Testing MicRons authentication...")
            info = self.client.info.get_datastack_info()
            datastack_name = info.get('datastack_name', self.datastack)
            logger.info(f"Successfully connected to datastack: {datastack_name}")
            
        except Exception as e:
            error_msg = f"""
CRITICAL ERROR: MicRons Authentication Failed!

This project requires REAL MicRons data access. No mock data or fallbacks allowed.

REQUIRED SETUP STEPS:
1. Visit: https://global.daf-apis.com/sticky_auth/api/v1/tos/2/accept
2. Generate token: client.auth.setup_token(make_new=True)
3. Save token: client.auth.save_token(token=YOUR_TOKEN)
4. Test with: client = CAVEclient('minnie65_public')

Error details: {str(e)}

EXPERIMENT HALTED - Real neuronal data access is mandatory for scientific validity.
"""
            logger.error(error_msg)
            print(error_msg)
            sys.exit(1)
    
    def get_neuron_ids(self, limit: int = 10) -> List[int]:
        """
        Get list of valid neuron IDs from MicRons dataset
        
        Args:
            limit: Maximum number of neurons to retrieve
            
        Returns:
            List of valid neuron root IDs
        """
        try:
            logger.info(f"Fetching neuron IDs from {self.datastack}...")
            
            # Get neuron table
            neuron_df = self.client.materialize.query_table('nucleus_detection_v0')
            
            if neuron_df.empty:
                raise ValueError("No neurons found in dataset")
            
            # Get root IDs (unique neurons) and filter out invalid ones
            all_root_ids = neuron_df['pt_root_id'].unique()
            
            # Filter out invalid IDs (like 0) and get valid ones
            valid_root_ids = []
            for root_id in all_root_ids:
                if root_id > 0:  # Valid root IDs should be positive
                    valid_root_ids.append(root_id)
                    if len(valid_root_ids) >= limit * 3:  # Get extra to account for failures
                        break
            
            logger.info(f"Found {len(valid_root_ids)} valid neurons from {len(all_root_ids)} total")
            return valid_root_ids[:limit * 3]  # Return extra for safety
            
        except Exception as e:
            logger.error(f"Failed to get neuron IDs: {str(e)}")
            raise
    
    def download_neuron_morphology(self, root_id: int) -> Dict[str, Any]:
        """
        Download detailed morphology for a specific neuron
        
        Args:
            root_id: Neuron root ID
            
        Returns:
            Dictionary containing neuron morphology data
        """
        cache_file = os.path.join(self.cache_dir, f'neuron_{root_id}_morphology.pkl')
        
        # Check cache first
        if os.path.exists(cache_file):
            logger.info(f"Loading cached morphology for neuron {root_id}")
            return pd.read_pickle(cache_file)
        
        try:
            logger.info(f"Downloading morphology for neuron {root_id}...")
            
            # Get neuron skeleton using the correct API method
            try:
                skeleton_dict = self.client.skeleton.get_skeleton(root_id, output_format='dict')
                skeleton_vertices = skeleton_dict.get('vertices', None)
                skeleton_edges = skeleton_dict.get('edges', None)
                skeleton_compartment = skeleton_dict.get('compartment', None)
                skeleton_radius = skeleton_dict.get('radius', None)
                logger.info(f"Successfully downloaded skeleton for neuron {root_id}")
            except Exception as e:
                logger.warning(f"Could not get skeleton for neuron {root_id}: {e}")
                skeleton_vertices = None
                skeleton_edges = None
                skeleton_compartment = None
                skeleton_radius = None
            
            # Get synaptic connections
            try:
                synapses_pre = self.client.materialize.synapse_query(
                    pre_ids=[root_id],
                    materialization_version=self.client.materialize.version
                )
            except Exception as e:
                logger.warning(f"Could not get presynaptic sites for neuron {root_id}: {e}")
                synapses_pre = None
            
            try:
                synapses_post = self.client.materialize.synapse_query(
                    post_ids=[root_id],
                    materialization_version=self.client.materialize.version
                )
            except Exception as e:
                logger.warning(f"Could not get postsynaptic sites for neuron {root_id}: {e}")
                synapses_post = None
            
            # Create morphology data with skeleton information
            morphology_data = {
                'root_id': root_id,
                'mesh_vertices': None,  # Mesh API not available in current version
                'mesh_faces': None,     # Mesh API not available in current version
                'skeleton_vertices': skeleton_vertices,
                'skeleton_edges': skeleton_edges,
                'skeleton_compartment': skeleton_compartment,  # Axon/dendrite labels
                'skeleton_radius': skeleton_radius,
                'presynaptic_sites': synapses_pre,
                'postsynaptic_sites': synapses_post,
                'datastack': self.datastack
            }
            
            # Validate that we have at least some usable data
            has_skeleton = skeleton_vertices is not None and len(skeleton_vertices) > 0
            has_synapses = (synapses_post is not None and len(synapses_post) > 0) or (synapses_pre is not None and len(synapses_pre) > 0)
            
            if not has_skeleton and not has_synapses:
                raise ValueError(f"Neuron {root_id} has no usable morphology or synapse data")
            
            if not has_skeleton:
                logger.warning(f"Neuron {root_id} has no skeleton data, but has synapses")
            
            if not has_synapses:
                logger.warning(f"Neuron {root_id} has no synapse data, but has skeleton")
            
            # Cache the data
            pd.to_pickle(morphology_data, cache_file)
            logger.info(f"Cached morphology data for neuron {root_id}")
            
            return morphology_data
            
        except Exception as e:
            logger.error(f"Failed to download morphology for neuron {root_id}: {str(e)}")
            raise
    
    def download_multiple_neurons(self, num_neurons: int = 2) -> List[Dict[str, Any]]:
        """
        Download morphology data for multiple neurons
        
        Args:
            num_neurons: Number of neurons to download (minimum 2 as per requirements)
            
        Returns:
            List of morphology dictionaries
        """
        if num_neurons < 2:
            raise ValueError("Must download at least 2 neurons as per project requirements")
        
        logger.info(f"Downloading {num_neurons} neurons from MicRons dataset...")
        
        # Get neuron IDs
        neuron_ids = self.get_neuron_ids(limit=num_neurons * 2)  # Get extra in case some fail
        
        morphologies = []
        successful_downloads = 0
        
        for root_id in tqdm(neuron_ids, desc="Downloading neurons"):
            if successful_downloads >= num_neurons:
                break
                
            try:
                morphology = self.download_neuron_morphology(root_id)
                morphologies.append(morphology)
                successful_downloads += 1
                logger.info(f"Successfully downloaded neuron {root_id} ({successful_downloads}/{num_neurons})")
                
            except Exception as e:
                logger.warning(f"Failed to download neuron {root_id}: {str(e)}")
                continue
        
        if successful_downloads < num_neurons:
            raise RuntimeError(f"Could only download {successful_downloads} neurons, needed {num_neurons}")
        
        logger.info(f"Successfully downloaded {len(morphologies)} neurons")
        return morphologies
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the MicRons dataset
        
        Returns:
            Dataset information dictionary
        """
        try:
            info = self.client.info.get_datastack_info()
            return {
                'datastack_name': info.get('datastack_name', self.datastack),
                'description': info.get('description', 'No description available'),
                'viewer_site': info.get('viewer_site', 'No viewer available'),
                'materialization_version': self.client.materialize.version
            }
        except Exception as e:
            logger.error(f"Failed to get dataset info: {str(e)}")
            raise

def main():
    """
    Test the MicRons downloader with strict authentication
    """
    print("Testing MicRons Data Downloader - STRICT MODE")
    print("=" * 50)
    
    try:
        # Initialize downloader (will fail if no authentication)
        downloader = MicRonsDownloader()
        
        # Get dataset info
        info = downloader.get_dataset_info()
        print(f"Connected to: {info['datastack_name']}")
        print(f"Version: {info['materialization_version']}")
        
        # Download sample neurons
        print("\nDownloading sample neurons...")
        neurons = downloader.download_multiple_neurons(num_neurons=2)
        
        print(f"\nSuccessfully downloaded {len(neurons)} neurons:")
        for i, neuron in enumerate(neurons):
            print(f"  Neuron {i+1}: ID {neuron['root_id']}")
            print(f"    Presynaptic sites: {len(neuron['presynaptic_sites'])}")
            print(f"    Postsynaptic sites: {len(neuron['postsynaptic_sites'])}")
        
        print("\nMicRons data access successful!")
        
    except SystemExit:
        print("Authentication failed - experiment halted as required")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
