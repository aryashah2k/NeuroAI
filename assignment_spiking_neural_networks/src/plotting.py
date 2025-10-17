from typing import Optional, Sequence
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from .utils import ensure_dir


def plot_raster(spikes: torch.Tensor, neuron_indices: Optional[Sequence[int]] = None,
                title: str = "Spike Raster", save_path: Optional[str] = None):
    """
    spikes: [T, B, N] or [T, N]
    plots only first batch if B present
    """
    if spikes.dim() == 3:
        spikes = spikes[:, 0]  # [T, N]
    T, N = spikes.shape
    if neuron_indices is None:
        neuron_indices = list(range(min(50, N)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ts, ns = [], []
    for idx in neuron_indices:
        t_idx = torch.nonzero(spikes[:, idx] > 0, as_tuple=False).squeeze(-1).cpu().numpy()
        ts.append(t_idx)
        ns.append(np.full_like(t_idx, idx))
    if len(ts) > 0:
        tcat = np.concatenate(ts) if len(ts) > 1 else ts[0]
        ncat = np.concatenate(ns) if len(ns) > 1 else ns[0]
        ax.scatter(tcat, ncat, s=4)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    ax.set_ylim(-1, max(neuron_indices)+1)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_voltage_traces(mem: torch.Tensor, neuron_indices: Sequence[int],
                        title: str = "Membrane Potentials", save_path: Optional[str] = None):
    if mem.dim() == 3:
        mem = mem[:, 0]
    T, N = mem.shape
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx in neuron_indices:
        ax.plot(mem[:, idx].cpu().numpy(), label=f"n{idx}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Membrane potential")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_accuracy_vs_tau(taus_ms: Sequence[float], accuracies: Sequence[float],
                         save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(taus_ms, accuracies, marker='o')
    ax.set_xscale('log')
    ax.set_xlabel("tau_m (ms)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs tau_m")
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_filters(weight_matrix: torch.Tensor, n_cols: int = 8, n_rows: int = 4,
                 save_path: Optional[str] = None, vmin: Optional[float] = None,
                 vmax: Optional[float] = None):
    """
    weight_matrix: [n_hidden, n_in], assume n_in=784 reshape 28x28
    """
    n_hidden = weight_matrix.size(0)
    count = min(n_hidden, n_cols * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*1.5, n_rows*1.5))
    for i in range(count):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        img = weight_matrix[i, :].view(28, 28).cpu().numpy()
        ax.imshow(img, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
    for i in range(count, n_cols*n_rows):
        r, c = divmod(i, n_cols)
        axes[r, c].axis('off')
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
