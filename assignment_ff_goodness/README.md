# Forward-Forward Algorithm Implementation

This repository contains an implementation of Geoffrey Hinton's "The Forward-Forward Algorithm" paper (arXiv:2212.13345), covering methodologies from sections 3.1, 3.2, 3.3, and 3.4.

## Overview

The Forward-Forward (FF) algorithm is an alternative to backpropagation for training neural networks. Instead of propagating error derivatives backward through the network, FF makes two forward passes through the network: one for positive data and one for negative data. Each layer is trained separately to maximize the "goodness" of positive data and minimize the "goodness" of negative data.

This implementation covers:

1. **Section 3.1**: Backpropagation baseline for comparison
2. **Section 3.2**: Unsupervised FF learning with positive and negative data
3. **Section 3.3**: Supervised FF with label embedding
4. **Section 3.4**: Top-down modeling with recurrent updates

## File Structure

- `backprop_baseline.py`: Implementation of backpropagation baseline for comparison (Section 3.1)
- `unsupervised_ff.py`: Unsupervised Forward-Forward algorithm (Section 3.2)
- `supervised_ff.py`: Supervised Forward-Forward with label embedding (Section 3.3)
- `recurrent_ff.py`: Top-down modeling with recurrent updates (Section 3.4)
- `utils.py`: Utility functions for data loading and processing
- `models.py`: Model definitions for all implementations
- `main.py`: Main script to run the experiments

## Requirements

```
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
matplotlib==3.7.1
tqdm==4.65.0
```

## Usage

To run the experiments:

```bash
# For backpropagation baseline (Section 3.1)
python main.py --model backprop

# For unsupervised Forward-Forward (Section 3.2)
python main.py --model unsupervised_ff

# For supervised Forward-Forward (Section 3.3)
python main.py --model supervised_ff

# For recurrent Forward-Forward with top-down effects (Section 3.4)
python main.py --model recurrent_ff
```

## Implementation Details

### Section 3.1: Backpropagation Baseline
A standard neural network with fully connected layers trained on MNIST using backpropagation, serving as a baseline for comparison.

### Section 3.2: Unsupervised Forward-Forward
Implements the Forward-Forward algorithm with positive data (real MNIST images) and negative data (hybrid images created with masks). Each layer is trained separately to maximize the goodness of positive data and minimize the goodness of negative data.

### Section 3.3: Supervised Forward-Forward with Label Embedding
Extends the Forward-Forward algorithm to supervised learning by embedding labels in the input. Positive data consists of images with correct labels, while negative data consists of images with incorrect labels.

### Section 3.4: Top-Down Modeling with Recurrent Updates
Implements a recurrent network architecture where layers update their activities based on both bottom-up and top-down inputs over multiple time steps.

## References

- Hinton, Geoffrey. "The Forward-Forward Algorithm." arXiv:2212.13345 (2022)
