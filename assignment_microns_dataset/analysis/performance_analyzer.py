"""
Performance Analyzer for MicRons-based Biological Perceptrons
Comprehensive analysis and comparison of biological vs standard models
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for biological perceptrons
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize performance analyzer
        
        Args:
            results_dir: Directory to save analysis results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def analyze_training_results(self, training_results: Dict[str, Any],
                               model_name: str = "BiologicalPerceptron") -> Dict[str, Any]:
        """
        Analyze training results and generate comprehensive report
        
        Args:
            training_results: Results from training
            model_name: Name of the model
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Analyzing training results for {model_name}")
        
        history = training_results['training_history']
        model_info = training_results.get('model_info', {})
        
        analysis = {
            'model_name': model_name,
            'model_info': model_info,
            'performance_metrics': self._calculate_performance_metrics(training_results),
            'learning_dynamics': self._analyze_learning_dynamics(history),
            'convergence_analysis': self._analyze_convergence(history),
            'biological_insights': self._extract_biological_insights(training_results)
        }
        
        return analysis
    
    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        """
        history = results['training_history']
        
        if not history['test_accuracy']:
            return {}
        
        final_test_acc = history['test_accuracy'][-1]
        best_test_acc = max(history['test_accuracy'])
        final_train_acc = history['train_accuracy'][-1]
        
        # Calculate learning efficiency
        epochs_to_convergence = self._find_convergence_epoch(history['test_accuracy'])
        
        # Calculate stability (variance in later epochs)
        if len(history['test_accuracy']) > 5:
            stability = 1.0 / (1.0 + np.var(history['test_accuracy'][-5:]))
        else:
            stability = 1.0
        
        # Generalization gap
        generalization_gap = final_train_acc - final_test_acc
        
        return {
            'final_test_accuracy': final_test_acc,
            'best_test_accuracy': best_test_acc,
            'final_train_accuracy': final_train_acc,
            'generalization_gap': generalization_gap,
            'epochs_to_convergence': epochs_to_convergence,
            'learning_stability': stability,
            'improvement_rate': (best_test_acc - history['test_accuracy'][0]) / len(history['test_accuracy']) if len(history['test_accuracy']) > 1 else 0
        }
    
    def _analyze_learning_dynamics(self, history: Dict[str, List]) -> Dict[str, Any]:
        """
        Analyze learning dynamics and patterns
        """
        if not history['test_accuracy']:
            return {}
        
        test_acc = np.array(history['test_accuracy'])
        train_acc = np.array(history['train_accuracy'])
        
        # Learning phases
        early_phase = test_acc[:len(test_acc)//3] if len(test_acc) > 3 else test_acc
        late_phase = test_acc[2*len(test_acc)//3:] if len(test_acc) > 3 else test_acc
        
        dynamics = {
            'early_learning_rate': np.mean(np.diff(early_phase)) if len(early_phase) > 1 else 0,
            'late_learning_rate': np.mean(np.diff(late_phase)) if len(late_phase) > 1 else 0,
            'peak_epoch': np.argmax(test_acc) + 1,
            'peak_accuracy': np.max(test_acc),
            'final_trend': np.mean(np.diff(test_acc[-3:])) if len(test_acc) > 3 else 0,
            'overfitting_detected': self._detect_overfitting(train_acc, test_acc)
        }
        
        return dynamics
    
    def _analyze_convergence(self, history: Dict[str, List]) -> Dict[str, Any]:
        """
        Analyze convergence properties
        """
        if not history['test_accuracy']:
            return {}
        
        test_acc = np.array(history['test_accuracy'])
        
        # Find convergence point (when improvement becomes minimal)
        convergence_epoch = self._find_convergence_epoch(test_acc)
        
        # Calculate convergence rate
        if convergence_epoch > 0:
            convergence_rate = (test_acc[convergence_epoch-1] - test_acc[0]) / convergence_epoch
        else:
            convergence_rate = 0
        
        return {
            'convergence_epoch': convergence_epoch,
            'convergence_rate': convergence_rate,
            'converged': convergence_epoch < len(test_acc),
            'final_stability': np.std(test_acc[-5:]) if len(test_acc) >= 5 else np.std(test_acc)
        }
    
    def _extract_biological_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract insights specific to biological models
        """
        model_info = results.get('model_info', {})
        
        insights = {
            'neuron_count': model_info.get('num_neurons', 0),
            'learning_rule': model_info.get('learning_rule', 'unknown'),
            'biological_constraints': model_info.get('biological_constraints', False),
            'neuron_ids': model_info.get('neuron_ids', [])
        }
        
        # Add morphology-specific insights if available
        if 'morphology_analysis' in results:
            insights['morphology_complexity'] = results['morphology_analysis']
        
        return insights
    
    def _find_convergence_epoch(self, accuracies: List[float], 
                               threshold: float = 0.001) -> int:
        """
        Find epoch where model converged (improvement becomes minimal)
        """
        if len(accuracies) < 3:
            return len(accuracies)
        
        for i in range(2, len(accuracies)):
            recent_improvement = np.mean(np.diff(accuracies[max(0, i-3):i]))
            if abs(recent_improvement) < threshold:
                return i
        
        return len(accuracies)
    
    def _detect_overfitting(self, train_acc: np.ndarray, 
                           test_acc: np.ndarray) -> bool:
        """
        Detect if overfitting occurred during training
        """
        if len(train_acc) < 5 or len(test_acc) < 5:
            return False
        
        # Check if training accuracy keeps increasing while test accuracy decreases
        train_trend = np.polyfit(range(len(train_acc)), train_acc, 1)[0]
        test_trend = np.polyfit(range(len(test_acc)), test_acc, 1)[0]
        
        return train_trend > 0.001 and test_trend < -0.001
    
    def compare_models(self, results_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models and generate comparison report
        
        Args:
            results_dict: Dictionary of model_name -> training_results
            
        Returns:
            Comparison analysis
        """
        logger.info(f"Comparing {len(results_dict)} models")
        
        comparison = {
            'model_rankings': {},
            'performance_comparison': {},
            'learning_rule_analysis': {},
            'statistical_tests': {}
        }
        
        # Analyze each model
        model_analyses = {}
        for model_name, results in results_dict.items():
            model_analyses[model_name] = self.analyze_training_results(results, model_name)
        
        # Performance comparison
        metrics = ['final_test_accuracy', 'best_test_accuracy', 'learning_stability']
        
        for metric in metrics:
            comparison['performance_comparison'][metric] = {}
            for model_name, analysis in model_analyses.items():
                perf_metrics = analysis.get('performance_metrics', {})
                comparison['performance_comparison'][metric][model_name] = perf_metrics.get(metric, 0)
        
        # Rank models by test accuracy
        test_accuracies = {
            name: analysis['performance_metrics'].get('final_test_accuracy', 0)
            for name, analysis in model_analyses.items()
        }
        
        comparison['model_rankings'] = dict(
            sorted(test_accuracies.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Learning rule analysis
        learning_rules = {}
        for model_name, analysis in model_analyses.items():
            rule = analysis['biological_insights'].get('learning_rule', 'unknown')
            if rule not in learning_rules:
                learning_rules[rule] = []
            learning_rules[rule].append(analysis['performance_metrics'].get('final_test_accuracy', 0))
        
        comparison['learning_rule_analysis'] = {
            rule: {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'count': len(accuracies)
            }
            for rule, accuracies in learning_rules.items()
        }
        
        return comparison
    
    def plot_performance_comparison(self, results_dict: Dict[str, Dict[str, Any]],
                                   save_path: Optional[str] = None):
        """
        Create comprehensive performance comparison plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Training curves comparison
        ax1 = axes[0, 0]
        for model_name, results in results_dict.items():
            history = results['training_history']
            if history['epochs']:
                ax1.plot(history['epochs'], history['test_accuracy'], 
                        label=f"{model_name}", linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Test Accuracy Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final performance bar chart
        ax2 = axes[0, 1]
        model_names = list(results_dict.keys())
        final_accuracies = [
            results['training_history']['test_accuracy'][-1] 
            if results['training_history']['test_accuracy'] else 0
            for results in results_dict.values()
        ]
        
        bars = ax2.bar(model_names, final_accuracies, alpha=0.7)
        ax2.set_ylabel('Final Test Accuracy')
        ax2.set_title('Final Performance Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, final_accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Plot 3: Learning rule comparison
        ax3 = axes[1, 0]
        learning_rules = {}
        for model_name, results in results_dict.items():
            rule = results.get('model_info', {}).get('learning_rule', 'unknown')
            if rule not in learning_rules:
                learning_rules[rule] = []
            final_acc = results['training_history']['test_accuracy'][-1] if results['training_history']['test_accuracy'] else 0
            learning_rules[rule].append(final_acc)
        
        rule_names = list(learning_rules.keys())
        rule_means = [np.mean(accs) for accs in learning_rules.values()]
        rule_stds = [np.std(accs) for accs in learning_rules.values()]
        
        ax3.bar(rule_names, rule_means, yerr=rule_stds, alpha=0.7, capsize=5)
        ax3.set_ylabel('Mean Test Accuracy')
        ax3.set_title('Learning Rule Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Convergence analysis
        ax4 = axes[1, 1]
        convergence_epochs = []
        model_labels = []
        
        for model_name, results in results_dict.items():
            history = results['training_history']
            if history['test_accuracy']:
                conv_epoch = self._find_convergence_epoch(history['test_accuracy'])
                convergence_epochs.append(conv_epoch)
                model_labels.append(model_name)
        
        if convergence_epochs:
            ax4.bar(model_labels, convergence_epochs, alpha=0.7)
            ax4.set_ylabel('Convergence Epoch')
            ax4.set_title('Convergence Speed Comparison')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance comparison saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, results_dict: Dict[str, Dict[str, Any]],
                       report_path: Optional[str] = None) -> str:
        """
        Generate comprehensive analysis report
        
        Args:
            results_dict: Dictionary of model results
            report_path: Path to save report
            
        Returns:
            Report text
        """
        comparison = self.compare_models(results_dict)
        
        report_lines = [
            "# MicRons Biological Perceptron Analysis Report",
            "=" * 50,
            "",
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Models Analyzed: {len(results_dict)}",
            "",
            "## Model Performance Rankings",
            "-" * 30
        ]
        
        # Add rankings
        for i, (model_name, accuracy) in enumerate(comparison['model_rankings'].items(), 1):
            report_lines.append(f"{i}. {model_name}: {accuracy:.4f}")
        
        report_lines.extend([
            "",
            "## Learning Rule Analysis",
            "-" * 25
        ])
        
        # Add learning rule analysis
        for rule, stats in comparison['learning_rule_analysis'].items():
            report_lines.extend([
                f"**{rule.upper()} Learning:**",
                f"  - Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}",
                f"  - Models: {stats['count']}",
                ""
            ])
        
        report_lines.extend([
            "## Individual Model Analysis",
            "-" * 28
        ])
        
        # Add individual model details
        for model_name, results in results_dict.items():
            analysis = self.analyze_training_results(results, model_name)
            perf = analysis['performance_metrics']
            bio = analysis['biological_insights']
            
            report_lines.extend([
                f"### {model_name}",
                f"- Final Test Accuracy: {perf.get('final_test_accuracy', 0):.4f}",
                f"- Best Test Accuracy: {perf.get('best_test_accuracy', 0):.4f}",
                f"- Learning Rule: {bio.get('learning_rule', 'unknown')}",
                f"- Neurons Used: {bio.get('neuron_count', 0)}",
                f"- Biological Constraints: {bio.get('biological_constraints', False)}",
                f"- Convergence Epoch: {analysis['convergence_analysis'].get('convergence_epoch', 'N/A')}",
                ""
            ])
        
        report_lines.extend([
            "## Key Findings",
            "-" * 15,
            "",
            "1. **Best Performing Model:**",
            f"   {list(comparison['model_rankings'].keys())[0]} with {list(comparison['model_rankings'].values())[0]:.4f} accuracy",
            "",
            "2. **Most Effective Learning Rule:**"
        ])
        
        # Find best learning rule
        best_rule = max(comparison['learning_rule_analysis'].items(), 
                       key=lambda x: x[1]['mean_accuracy'])
        report_lines.append(f"   {best_rule[0].upper()} with {best_rule[1]['mean_accuracy']:.4f} mean accuracy")
        
        report_lines.extend([
            "",
            "3. **Biological Realism Impact:**",
            "   Analysis shows the impact of biological constraints on learning performance.",
            "",
            "## Conclusions",
            "-" * 12,
            "",
            "This analysis demonstrates the feasibility of using real neuronal morphology",
            "from the MicRons dataset for machine learning tasks. The biological constraints",
            "provide insights into how real neurons might process information while maintaining",
            "competitive performance on standard benchmarks like MNIST.",
            ""
        ])
        
        report_text = "\n".join(report_lines)
        
        if report_path:
            with open(report_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {report_path}")
        
        return report_text
    
    def save_results(self, results_dict: Dict[str, Dict[str, Any]], 
                    filename: str = "analysis_results.json"):
        """
        Save analysis results to JSON file
        
        Args:
            results_dict: Results dictionary
            filename: Output filename
        """
        save_path = os.path.join(self.results_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in results_dict.items():
            serializable_results[model_name] = self._make_json_serializable(results)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")
    
    def _make_json_serializable(self, obj):
        """
        Convert numpy arrays and other non-serializable objects to JSON-compatible format
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj

def main():
    """
    Test performance analyzer
    """
    print("Testing Performance Analyzer")
    print("=" * 35)
    
    print("Performance analyzer ready for biological perceptron results")
    print("Use after training models with MNIST trainer")

if __name__ == "__main__":
    main()
