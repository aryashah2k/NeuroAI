import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from typing import Tuple, List, Optional, Union, Callable


def load_mnist_data(batch_size: int = 128, use_cuda: bool = torch.cuda.is_available(), 
                   permutation_invariant: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the MNIST dataset as described in Section 3.1 of the paper.
    
    Args:
        batch_size: Batch size for data loaders
        use_cuda: Whether to use CUDA for DataLoader workers
        permutation_invariant: If True, flatten the images for permutation-invariant MNIST
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training, validation and test sets
    """
    transform_list = [transforms.ToTensor()]
    if permutation_invariant:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    
    transform = transforms.Compose(transform_list)
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Split training set into train and validation as described in 3.1
    train_size = 50000
    val_size = 10000
    indices = list(range(len(train_dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # Create data loaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    return train_loader, val_loader, test_loader


def generate_negative_samples_masked(images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate negative samples using the masking technique described in Section 3.2.
    Creates hybrid images by combining two different digit images with a mask.
    
    Args:
        images: Batch of images
        labels: Batch of labels
        
    Returns:
        negative_images, negative_labels: Negative samples with labels
    """
    batch_size = images.size(0)
    device = images.device
    
    # Create random permutation to pair different images
    permutation = torch.randperm(batch_size).to(device)
    
    # For generating large patchy masks as described in Section 3.2
    if len(images.shape) == 2:  # For permutation invariant (flattened) MNIST
        img_size = int(np.sqrt(images.shape[1]))
        masks = create_blurred_masks((batch_size, img_size, img_size), device)
        masks = masks.reshape(batch_size, -1)
    else:  # For original MNIST format
        img_size = images.shape[2]
        masks = create_blurred_masks((batch_size, img_size, img_size), device)
        masks = masks.unsqueeze(1)  # Add channel dimension
    
    # Create hybrid images
    negative_images = images * masks + images[permutation] * (1 - masks)
    
    # Assign the label of the dominant image (original image)
    negative_labels = labels.clone()
    
    return negative_images, negative_labels


def create_blurred_masks(shape: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """
    Create masks with large regions of ones and zeros by blurring and thresholding.
    
    Args:
        shape: Shape of the mask to create (batch_size, height, width)
        device: Device to create tensor on
        
    Returns:
        masks: Batch of binary masks
    """
    batch_size, height, width = shape
    
    # Start with random bit images
    random_bits = torch.rand(batch_size, height, width).to(device)
    
    # Define blur filter [1/4, 1/2, 1/4]
    blur_filter = torch.tensor([0.25, 0.5, 0.25]).to(device)
    
    # Apply blurring multiple times as described in Section 3.2
    blurred = random_bits
    for _ in range(5):  # Apply blurring multiple times
        # Apply horizontal blurring
        blurred = F.conv1d(
            blurred.reshape(batch_size * height, 1, width),
            blur_filter.reshape(1, 1, 3),
            padding=1
        ).reshape(batch_size, height, width)
        
        # Apply vertical blurring
        blurred = F.conv1d(
            blurred.transpose(1, 2).reshape(batch_size * width, 1, height),
            blur_filter.reshape(1, 1, 3),
            padding=1
        ).reshape(batch_size, width, height).transpose(1, 2)
    
    # Threshold at 0.5
    masks = (blurred > 0.5).float()
    
    return masks


def generate_supervised_negative_samples(images: torch.Tensor, labels: torch.Tensor, 
                                         num_classes: int = 10, enhance: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate negative samples for supervised learning with enhanced distortion.
    Creates samples with incorrect labels and applies distortions to make them more distinct.
    
    Args:
        images: Batch of images
        labels: Batch of labels
        num_classes: Number of classes (10 for MNIST)
        enhance: Whether to apply enhancements to make negative samples more distinct
        
    Returns:
        negative_images, negative_labels: Images with incorrect labels and distortions
    """
    batch_size = images.size(0)
    device = images.device
    
    # Generate random incorrect labels
    incorrect_labels = torch.randint(1, num_classes, (batch_size,)).to(device)
    negative_labels = (labels + incorrect_labels) % num_classes
    
    if enhance:
        # Create distorted versions of the images
        negative_images = images.clone()
        
        # 1. Add strong noise
        noise_magnitude = 0.4  # Strong noise
        negative_images = negative_images + torch.randn_like(negative_images) * noise_magnitude
        
        # 2. Flip some pixels randomly (randomly invert ~15% of pixels)
        flip_mask = (torch.rand_like(negative_images) < 0.15)
        pixel_max = images.max()
        negative_images[flip_mask] = pixel_max - negative_images[flip_mask]
        
        # 3. Apply transformation based on incorrect label
        # For each sample, create a specific distortion pattern based on the incorrect label
        for i in range(batch_size):
            # Add a bias based on incorrect label
            label_factor = negative_labels[i].item() / num_classes
            negative_images[i] = (1 - label_factor) * negative_images[i] + label_factor * (1 - negative_images[i])
            
            # Apply structured distortion (assuming flattened MNIST)
            if len(negative_images.shape) == 2:  # Flattened images
                img = negative_images[i].view(28, 28)  # Reshape to 2D
                
                # Create patterns based on the incorrect label
                grid_size = 4
                grid_val = (negative_labels[i].item() / num_classes) * 0.8 + 0.1
                
                for j in range(0, 28, grid_size):
                    for k in range(0, 28, grid_size):
                        if (j // grid_size + k // grid_size) % 2 == (negative_labels[i].item() % 2):
                            img[j:j+2, k:k+2] = grid_val
                
                # Apply back to batch
                negative_images[i] = img.view(-1)
        
        # Clamp to valid range
        if images.max() <= 1.0:
            negative_images = torch.clamp(negative_images, 0.0, 1.0)
        
        return negative_images, negative_labels
    else:
        # Paper Section 3.3: Use the same images with incorrect labels
        return images, negative_labels


def prepare_images_with_labels(images: torch.Tensor, labels: torch.Tensor, 
                              num_classes: int = 10, embed_location: str = 'beginning') -> torch.Tensor:
    """
    Embed labels into images as described in Section 3.3.
    
    Args:
        images: Batch of images (B, C, H, W) or (B, D) for flattened images
        labels: Batch of labels (B,)
        num_classes: Number of classes (10 for MNIST)
        embed_location: Where to embed the labels ('beginning' or 'border')
        
    Returns:
        images_with_labels: Images with embedded labels
    """
    batch_size = images.size(0)
    device = images.device
    
    # Convert labels to one-hot encoding
    one_hot = F.one_hot(labels, num_classes).float()
    
    if len(images.shape) == 2:  # For permutation invariant (flattened) MNIST
        if embed_location == 'beginning':
            # Replace the first num_classes pixels with the one-hot label
            images_with_labels = images.clone()
            images_with_labels[:, :num_classes] = one_hot
        else:
            # Append the one-hot label to the image
            images_with_labels = torch.cat([one_hot, images], dim=1)
    else:
        # For non-flattened images, replace the top-left corner or border
        img_size = images.shape[2]
        if embed_location == 'beginning':
            # Replace the first rows of the image with the one-hot label
            images_with_labels = images.clone()
            label_height = (num_classes + img_size - 1) // img_size  # Ceiling division
            for i in range(min(label_height, img_size)):
                width_used = min(num_classes - i * img_size, img_size)
                if width_used > 0:
                    images_with_labels[:, 0, i, :width_used] = one_hot[:, i*img_size:i*img_size+width_used]
        else:
            # Replace the border pixels with the one-hot label
            # This is more complex and less described in the paper
            images_with_labels = images.clone()
            # Top border
            width_to_use = min(num_classes, img_size)
            images_with_labels[:, 0, 0, :width_to_use] = one_hot[:, :width_to_use]
    
    return images_with_labels


def generate_neutral_label(batch_size: int, num_classes: int = 10, 
                          device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Generate neutral labels (all 0.1) as described in Section 3.3 for inference.
    
    Args:
        batch_size: Batch size
        num_classes: Number of classes
        device: Device to create tensor on
        
    Returns:
        neutral_label: Tensor of neutral labels
    """
    return torch.ones(batch_size, num_classes).to(device) * 0.1


def visualize_first_layer_weights(model, figsize=(10, 10), num_filters=25):
    """
    Visualize the weights of the first layer of the model as shown in Figure 2.
    
    Args:
        model: Neural network model
        figsize: Figure size for the plot
        num_filters: Number of filters to visualize
    """
    # Get the weights of the first layer
    weights = next(model.parameters()).detach().cpu()
    
    # Reshape if they are for permutation invariant MNIST
    if len(weights.shape) == 2:
        # Assuming square images
        input_size = int(np.sqrt(weights.shape[1]))
        weights = weights[:num_filters, :].reshape(num_filters, 1, input_size, input_size)
    else:
        weights = weights[:num_filters]
    
    # Create a grid of filter visualizations
    fig, axes = plt.subplots(5, 5, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            # Normalize the weights for better visualization
            filter_weights = weights[i, 0]
            vmin, vmax = filter_weights.min(), filter_weights.max()
            ax.imshow(filter_weights, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_curves(losses, accuracies, title="Training Progress"):
    """
    Plot training loss and accuracy curves.
    
    Args:
        losses: List of loss values per epoch
        accuracies: List of accuracy values per epoch
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(accuracies)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
