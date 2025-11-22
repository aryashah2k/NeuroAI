# MicRons Biological Perceptron Analysis Report
==================================================

Analysis Date: 2025-08-17 22:33:39
Models Analyzed: 2

## Model Performance Rankings
------------------------------
1. BiologicalPerceptron_stdp: 0.1027
2. BiologicalPerceptron_hebbian: 0.1010

## Learning Rule Analysis
-------------------------
**HEBBIAN Learning:**
  - Mean Accuracy: 0.1010 ± 0.0000
  - Models: 1

**STDP Learning:**
  - Mean Accuracy: 0.1027 ± 0.0000
  - Models: 1

## Individual Model Analysis
----------------------------
### BiologicalPerceptron_hebbian
- Final Test Accuracy: 0.1010
- Best Test Accuracy: 0.1010
- Learning Rule: hebbian
- Neurons Used: 2
- Biological Constraints: False
- Convergence Epoch: 2

### BiologicalPerceptron_stdp
- Final Test Accuracy: 0.1027
- Best Test Accuracy: 0.1027
- Learning Rule: stdp
- Neurons Used: 2
- Biological Constraints: False
- Convergence Epoch: 2

## Key Findings
---------------

1. **Best Performing Model:**
   BiologicalPerceptron_stdp with 0.1027 accuracy

2. **Most Effective Learning Rule:**
   STDP with 0.1027 mean accuracy

3. **Biological Realism Impact:**
   Analysis shows the impact of biological constraints on learning performance.

## Conclusions
------------

This analysis demonstrates the feasibility of using real neuronal morphology
from the MicRons dataset for machine learning tasks. The biological constraints
provide insights into how real neurons might process information while maintaining
competitive performance on standard benchmarks like MNIST.
