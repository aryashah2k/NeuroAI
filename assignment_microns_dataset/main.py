"""
MicRons Neuron-as-Perceptron MNIST Classification Pipeline
Complete end-to-end pipeline for biological perceptron modeling using real MicRons data

CRITICAL: This implementation requires REAL MicRons data authentication.
No mock data or fallbacks - scientific validity depends on real neuronal morphology.
"""

import os
import sys
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional
import argparse
from datetime import datetime

# Add project paths
sys.path.append(os.path.dirname(__file__))
from data_acquisition.microns_downloader import MicRonsDownloader
from data_acquisition.data_preprocessor import MicRonsDataPreprocessor
from models.biological_perceptron import create_biological_perceptron_from_microns
from models.learning_rules import create_biological_trainer
from training.mnist_trainer import MNISTBiologicalTrainer, compare_learning_rules
from analysis.performance_analyzer import PerformanceAnalyzer
from analysis.visualizer import MicRonsVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('microns_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MicRonsExperimentPipeline:
    """
    Complete experimental pipeline for MicRons biological perceptron modeling
    """
    
    def __init__(self, num_neurons: int = 2, 
                 learning_rules: List[str] = ["hebbian", "stdp"],
                 num_epochs: int = 10,
                 biological_constraints: bool = True,
                 output_dir: str = "experiment_results"):
        """
        Initialize experiment pipeline
        
        Args:
            num_neurons: Number of neurons to download (minimum 2)
            learning_rules: List of learning rules to test
            num_epochs: Training epochs per model
            biological_constraints: Apply biological constraints
            output_dir: Output directory for results
        """
        self.num_neurons = max(2, num_neurons)  # Ensure minimum 2 neurons
        self.learning_rules = learning_rules
        self.num_epochs = num_epochs
        self.biological_constraints = biological_constraints
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "analysis"), exist_ok=True)
        
        # Initialize components
        self.downloader = None
        self.preprocessor = MicRonsDataPreprocessor()
        self.analyzer = PerformanceAnalyzer(os.path.join(output_dir, "analysis"))
        self.visualizer = MicRonsVisualizer(os.path.join(output_dir, "visualizations"))
        
        # Data storage
        self.raw_morphologies = []
        self.processed_morphologies = []
        self.training_results = {}
        
        logger.info(f"Initialized MicRons experiment pipeline")
        logger.info(f"Target neurons: {self.num_neurons}")
        logger.info(f"Learning rules: {self.learning_rules}")
        logger.info(f"Training epochs: {self.num_epochs}")
        logger.info(f"Biological constraints: {self.biological_constraints}")
    
    def step1_authenticate_and_download(self) -> bool:
        """
        Step 1: Authenticate with MicRons and download neuronal data
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("STEP 1: MicRons Authentication and Data Download")
        logger.info("=" * 60)
        
        try:
            # Initialize downloader (will fail if no authentication)
            logger.info("Initializing MicRons downloader with strict authentication...")
            self.downloader = MicRonsDownloader()
            
            # Get dataset information
            dataset_info = self.downloader.get_dataset_info()
            logger.info(f"Connected to: {dataset_info['datastack_name']}")
            logger.info(f"Version: {dataset_info['materialization_version']}")
            
            # Download neuronal morphologies
            logger.info(f"Downloading {self.num_neurons} real neurons...")
            self.raw_morphologies = self.downloader.download_multiple_neurons(
                num_neurons=self.num_neurons
            )
            
            logger.info(f"Successfully downloaded {len(self.raw_morphologies)} neurons:")
            for i, morphology in enumerate(self.raw_morphologies):
                neuron_id = morphology['root_id']
                pre_synapses = len(morphology['presynaptic_sites'])
                post_synapses = len(morphology['postsynaptic_sites'])
                logger.info(f"  Neuron {i+1}: ID {neuron_id} ({pre_synapses} pre, {post_synapses} post synapses)")
            
            return True
            
        except SystemExit:
            logger.error("MicRons authentication failed - experiment cannot proceed")
            logger.error("Please complete authentication setup as described in the error message")
            return False
        except Exception as e:
            logger.error(f"Failed to download MicRons data: {str(e)}")
            return False
    
    def step2_preprocess_morphologies(self) -> bool:
        """
        Step 2: Preprocess raw morphology data for perceptron modeling
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Morphology Data Preprocessing")
        logger.info("=" * 60)
        
        if not self.raw_morphologies:
            logger.error("No raw morphologies available for preprocessing")
            return False
        
        try:
            self.processed_morphologies = []
            
            for i, raw_morphology in enumerate(self.raw_morphologies):
                neuron_id = raw_morphology['root_id']
                logger.info(f"Processing neuron {neuron_id} ({i+1}/{len(self.raw_morphologies)})...")
                
                # Process morphology
                processed = self.preprocessor.process_neuron_morphology(raw_morphology)
                self.processed_morphologies.append(processed)
                
                # Log processing results
                segments = processed['dendrite_segments']
                synaptic_inputs = processed['synaptic_inputs']
                logger.info(f"  - {len(segments)} dendritic segments identified")
                logger.info(f"  - {len(synaptic_inputs)} synaptic inputs mapped")
                
                # Create morphology visualization
                viz_path = os.path.join(self.output_dir, "visualizations", f"neuron_{neuron_id}_morphology.html")
                self.visualizer.visualize_neuron_morphology_3d(processed, viz_path)
            
            # Create input mapping for perceptron models
            input_mapping = self.preprocessor.create_perceptron_input_mapping(self.processed_morphologies)
            logger.info(f"Created input mapping with {input_mapping['input_dimension']} input channels")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to preprocess morphologies: {str(e)}")
            return False
    
    def step3_train_biological_perceptrons(self) -> bool:
        """
        Step 3: Train biological perceptron models with different learning rules
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Biological Perceptron Training")
        logger.info("=" * 60)
        
        if not self.processed_morphologies:
            logger.error("No processed morphologies available for training")
            return False
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Training on device: {device}")
            
            # Train models with different learning rules
            for learning_rule in self.learning_rules:
                logger.info(f"\nTraining with {learning_rule.upper()} learning rule...")
                
                # Create trainer
                trainer = MNISTBiologicalTrainer(
                    morphology_data_list=self.processed_morphologies,
                    learning_rule=learning_rule,
                    batch_size=32,
                    biological_constraints=self.biological_constraints,
                    device=str(device)
                )
                
                # Train model
                model_name = f"BiologicalPerceptron_{learning_rule}"
                results = trainer.train(
                    num_epochs=self.num_epochs,
                    save_path=os.path.join(self.output_dir, "models", f"{model_name}.pth")
                )
                
                # Store results
                self.training_results[model_name] = results
                
                # Create training visualization
                viz_path = os.path.join(self.output_dir, "visualizations", f"{model_name}_training.png")
                trainer.plot_training_curves(viz_path)
                
                # Log results
                final_acc = results['final_test_accuracy']
                best_acc = results['best_test_accuracy']
                logger.info(f"{learning_rule.upper()} Results - Final: {final_acc:.4f}, Best: {best_acc:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train biological perceptrons: {str(e)}")
            return False
    
    def step4_analyze_and_compare(self) -> bool:
        """
        Step 4: Analyze results and create comprehensive comparison
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Performance Analysis and Comparison")
        logger.info("=" * 60)
        
        if not self.training_results:
            logger.error("No training results available for analysis")
            return False
        
        try:
            # Generate comprehensive analysis
            logger.info("Generating performance analysis...")
            
            # Create performance comparison visualization
            comparison_path = os.path.join(self.output_dir, "visualizations", "performance_comparison.png")
            self.analyzer.plot_performance_comparison(self.training_results, comparison_path)
            
            # Generate detailed report
            report_path = os.path.join(self.output_dir, "analysis", "experiment_report.md")
            report_text = self.analyzer.generate_report(self.training_results, report_path)
            
            # Save results to JSON
            self.analyzer.save_results(self.training_results, "training_results.json")
            
            # Create comprehensive dashboard
            dashboard_path = os.path.join(self.output_dir, "visualizations", "dashboard.html")
            self.visualizer.create_comprehensive_dashboard(
                self.processed_morphologies,
                self.training_results,
                dashboard_path
            )
            
            # Print summary
            logger.info("\nEXPERIMENT SUMMARY:")
            logger.info("-" * 40)
            
            best_model = max(self.training_results.items(), 
                           key=lambda x: x[1]['final_test_accuracy'])
            best_name, best_results = best_model
            
            logger.info(f"Best Model: {best_name}")
            logger.info(f"Best Accuracy: {best_results['final_test_accuracy']:.4f}")
            logger.info(f"Learning Rule: {best_results['model_info'].get('learning_rule', 'Unknown')}")
            
            logger.info(f"\nResults saved to: {self.output_dir}")
            logger.info(f"Dashboard: {dashboard_path}")
            logger.info(f"Report: {report_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to analyze results: {str(e)}")
            return False
    
    def run_complete_experiment(self) -> bool:
        """
        Run the complete experimental pipeline
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("STARTING MICRONS BIOLOGICAL PERCEPTRON EXPERIMENT")
        logger.info("=" * 80)
        logger.info(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Step 1: Download MicRons data
        if not self.step1_authenticate_and_download():
            logger.error("EXPERIMENT FAILED: Could not download MicRons data")
            return False
        
        # Step 2: Preprocess morphologies
        if not self.step2_preprocess_morphologies():
            logger.error("EXPERIMENT FAILED: Could not preprocess morphologies")
            return False
        
        # Step 3: Train models
        if not self.step3_train_biological_perceptrons():
            logger.error("EXPERIMENT FAILED: Could not train models")
            return False
        
        # Step 4: Analyze results
        if not self.step4_analyze_and_compare():
            logger.error("EXPERIMENT FAILED: Could not analyze results")
            return False
        
        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
        logger.info(f"Experiment finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        return True

def main():
    """
    Main entry point for MicRons biological perceptron experiment
    """
    parser = argparse.ArgumentParser(
        description="MicRons Neuron-as-Perceptron MNIST Classification Experiment"
    )
    
    parser.add_argument("--neurons", type=int, default=2,
                       help="Number of neurons to download (minimum 2)")
    parser.add_argument("--learning-rules", nargs="+", 
                       default=["hebbian", "stdp"],
                       choices=["hebbian", "stdp", "bcm"],
                       help="Learning rules to test")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Training epochs per model")
    parser.add_argument("--no-bio-constraints", action="store_true",
                       help="Disable biological constraints")
    parser.add_argument("--output-dir", type=str, default="experiment_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Print experiment configuration
    print("MicRons Biological Perceptron Experiment")
    print("=" * 50)
    print(f"Neurons to download: {args.neurons}")
    print(f"Learning rules: {args.learning_rules}")
    print(f"Training epochs: {args.epochs}")
    print(f"Biological constraints: {not args.no_bio_constraints}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    print("CRITICAL REQUIREMENTS:")
    print("- Real MicRons data authentication required")
    print("- Visit: https://global.daf-apis.com/sticky_auth/api/v1/tos/2/accept")
    print("- Generate and save authentication token")
    print("- No mock data or fallbacks allowed")
    print()
    
    # Create and run experiment
    pipeline = MicRonsExperimentPipeline(
        num_neurons=args.neurons,
        learning_rules=args.learning_rules,
        num_epochs=args.epochs,
        biological_constraints=not args.no_bio_constraints,
        output_dir=args.output_dir
    )
    
    success = pipeline.run_complete_experiment()
    
    if success:
        print("\nExperiment completed successfully!")
        print(f"Results available in: {args.output_dir}")
        sys.exit(0)
    else:
        print("\nExperiment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
