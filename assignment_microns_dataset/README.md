# MicRons Neuron-as-Perceptron MNIST Classification Project

A comprehensive implementation of biological perceptron models using **real neuronal morphology** from the MicRons dataset for MNIST classification. This project demonstrates how actual brain neurons can be modeled as computational units with biologically plausible learning rules.

## 🧠 Project Overview

This project implements the academic requirements for modeling real neurons as perceptrons using the MicRons dataset. It strictly uses **real neuronal data only** - no mock data or fallbacks are allowed to maintain scientific validity.

### Key Features

- **Real MicRons Data**: Uses authentic neuronal morphology from the minnie65_public datastack
- **Biological Realism**: Models dendritic processing, synaptic integration, and realistic constraints
- **Multiple Learning Rules**: Implements Hebbian, STDP, and BCM learning algorithms
- **MNIST Classification**: Demonstrates biological computation on a standard ML benchmark
- **Comprehensive Analysis**: Detailed performance analysis and morphology visualization

## 🔧 Installation

### Prerequisites

1. **Python 3.8+** with the following packages:
```bash
pip install -r requirements.txt
```

2. **MicRons Authentication** (CRITICAL):
   - Visit: https://global.daf-apis.com/sticky_auth/api/v1/tos/2/accept
   - Accept terms of service
   - Generate authentication token:
     ```python
     from caveclient import CAVEclient
     client = CAVEclient('minnie65_public')
     client.auth.setup_token(make_new=True)
     # Follow prompts to complete authentication
     client.auth.save_token(token=YOUR_TOKEN)
     ```

### Dependencies

```
caveclient>=5.0.0      # MicRons data access
cloudvolume>=8.0.0     # Volume data handling
meshparty>=1.16.0      # Mesh processing
torch>=2.0.0           # Neural networks
torchvision>=0.15.0    # MNIST dataset
numpy>=1.24.0          # Numerical computing
pandas>=2.0.0          # Data manipulation
matplotlib>=3.7.0      # Plotting
plotly>=5.15.0         # 3D visualization
seaborn>=0.12.0        # Statistical plots
networkx>=3.1.0        # Graph analysis
scikit-learn>=1.3.0    # ML utilities
```

## 🚀 Quick Start

### Basic Usage

```bash
# Run complete experiment with default settings
python main.py

# Customize experiment parameters
python main.py --neurons 3 --learning-rules hebbian stdp bcm --epochs 15

# Disable biological constraints for comparison
python main.py --no-bio-constraints --output-dir unconstrained_results
```

### Step-by-Step Usage

```python
from main import MicRonsExperimentPipeline

# Initialize experiment
pipeline = MicRonsExperimentPipeline(
    num_neurons=2,
    learning_rules=["hebbian", "stdp"],
    num_epochs=10,
    biological_constraints=True
)

# Run complete pipeline
success = pipeline.run_complete_experiment()
```

## 📁 Project Structure

```
assignment2/
├── data_acquisition/
│   ├── microns_downloader.py      # Strict MicRons data download
│   └── data_preprocessor.py       # Morphology preprocessing
├── models/
│   ├── biological_perceptron.py   # Biological perceptron models
│   └── learning_rules.py          # Hebbian, STDP, BCM learning
├── training/
│   └── mnist_trainer.py           # MNIST training pipeline
├── analysis/
│   ├── performance_analyzer.py    # Performance analysis
│   └── visualizer.py              # 3D morphology visualization
├── main.py                        # Complete experiment pipeline
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🧬 Scientific Approach

### Biological Perceptron Model

The project models real neurons as perceptrons by:

1. **Dendritic Processing**: Maps MNIST pixels to dendritic segments based on real morphology
2. **Synaptic Integration**: Processes inputs through realistic synaptic locations
3. **Somatic Computation**: Integrates dendritic outputs with biological constraints
4. **Axonal Output**: Generates classification decisions

### Learning Rules Implemented

- **Hebbian Learning**: "Neurons that fire together, wire together"
- **STDP**: Spike-timing dependent plasticity with temporal dynamics
- **BCM**: Bienenstock-Cooper-Munro rule with sliding threshold
- **Metaplasticity**: Adaptive learning rates based on activity history

### Biological Constraints

- Sparse connectivity patterns
- Dendritic saturation effects
- Limited local computation
- Homeostatic scaling mechanisms
- Realistic weight initialization

## 📊 Experiment Pipeline

### Step 1: Data Acquisition
```python
# Download real MicRons neurons (minimum 2 required)
downloader = MicRonsDownloader()
morphologies = downloader.download_multiple_neurons(num_neurons=2)
```

### Step 2: Morphology Processing
```python
# Extract dendritic structure and synaptic locations
preprocessor = MicRonsDataPreprocessor()
processed = preprocessor.process_neuron_morphology(raw_morphology)
```

### Step 3: Model Training
```python
# Train biological perceptron with different learning rules
trainer = MNISTBiologicalTrainer(morphologies, learning_rule="hebbian")
results = trainer.train(num_epochs=10)
```

### Step 4: Analysis & Visualization
```python
# Comprehensive performance analysis
analyzer = PerformanceAnalyzer()
comparison = analyzer.compare_models(training_results)
```

## 📈 Results and Analysis

The experiment generates comprehensive results including:

### Performance Metrics
- Classification accuracy on MNIST test set
- Learning convergence analysis
- Comparison with standard MLPs
- Biological constraint impact assessment

### Visualizations
- 3D neuronal morphology rendering
- Dendritic input mapping to MNIST pixels
- Training dynamics and learning curves
- Synaptic activation patterns

### Output Files
```
experiment_results/
├── models/                    # Trained model checkpoints
├── visualizations/
│   ├── neuron_*_morphology.html    # 3D morphology views
│   ├── *_training.png              # Training curves
│   ├── performance_comparison.png   # Model comparison
│   └── dashboard.html              # Comprehensive dashboard
└── analysis/
    ├── experiment_report.md        # Detailed analysis report
    └── training_results.json       # Raw results data
```

## 🔬 Scientific Insights

### Key Findings

1. **Biological Feasibility**: Real neuronal morphology can effectively perform MNIST classification
2. **Learning Rule Comparison**: Different biological learning rules show distinct performance characteristics
3. **Constraint Impact**: Biological constraints affect learning dynamics while maintaining functionality
4. **Morphology Matters**: Dendritic structure influences input processing and classification performance

### Academic Contributions

- Demonstrates practical application of computational neuroscience to machine learning
- Provides framework for testing biological learning theories
- Bridges gap between neuroscience data and AI algorithms
- Offers insights into energy-efficient neural computation

## ⚠️ Critical Requirements

### Authentication
- **MUST** have valid MicRons authentication
- **NO** mock data or fallbacks allowed
- **REAL** neuronal data required for scientific validity

### Minimum Requirements
- At least 2 real neurons from MicRons dataset
- Proper dendritic structure mapping
- Biologically plausible learning rules
- MNIST classification demonstration

## 🐛 Troubleshooting

### Common Issues

1. **Authentication Failed**
   ```
   Solution: Complete MicRons authentication setup
   Visit: https://global.daf-apis.com/sticky_auth/api/v1/tos/2/accept
   Generate and save token as described above
   ```

2. **No Dendritic Segments**
   ```
   Solution: Try different neurons or check morphology data quality
   Some neurons may have incomplete reconstruction
   ```

3. **CUDA Out of Memory**
   ```
   Solution: Reduce batch size or use CPU
   python main.py --batch-size 16
   ```

### Error Messages

- `"EXPERIMENT HALTED - Real neuronal data access is mandatory"`: Authentication required
- `"Must have at least 2 neurons"`: Increase neuron count parameter
- `"No valid neurons could be created"`: Check morphology data quality

## 📚 References

1. MicRons Consortium. "Functional connectomics spanning multiple areas of mouse visual cortex." Nature (2021).
2. Hebbian Learning: Hebb, D.O. "The Organization of Behavior" (1949).
3. STDP: Bi, G. & Poo, M. "Synaptic modifications in cultured hippocampal neurons." Journal of Neuroscience (1998).
4. BCM Theory: Bienenstock, E.L., Cooper, L.N. & Munro, P.W. "Theory for the development of neuron selectivity." Journal of Neuroscience (1982).

## 📄 License

This project is for academic research purposes. MicRons data usage subject to their terms of service.

## 🤝 Contributing

This is an academic project. For questions or issues:
1. Check authentication setup
2. Verify MicRons data access
3. Review error logs in `microns_experiment.log`

---

**Note**: This implementation strictly adheres to the requirement of using REAL MicRons neuronal data only. No mock data, fallbacks, or "smart" alternatives are provided to maintain scientific integrity and validity of the biological modeling approach.
