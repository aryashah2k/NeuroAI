import os
import argparse
from tqdm import tqdm
import numpy as np
import torch

from src.datasets import get_mnist_dataloaders
from src.encoding import poisson_encode, spike_counts
from src.model import FeedforwardSNN, tau_to_beta
from src.readout import train_readout, evaluate_readout
from src.plotting import plot_raster, plot_voltage_traces, plot_filters
from src.utils import get_device, ensure_dir, set_seed


def run_sparse(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    ensure_dir(args.output_dir)
    ensure_dir(args.fig_dir)

    train_loader, test_loader = get_mnist_dataloaders(data_dir=args.data_dir,
                                                      batch_size=args.batch_size,
                                                      num_workers=args.num_workers,
                                                      drop_last=False)

    beta = tau_to_beta(args.tau_ms, args.dt_ms)
    snn = FeedforwardSNN(n_in=28*28,
                         n_hidden=args.hidden,
                         beta=beta,
                         threshold=args.v_th,
                         refractory_steps=int(args.t_ref_ms/args.dt_ms),
                         sparsity_p=args.p_connect,
                         weight_mean=args.w_mean,
                         weight_std=args.w_std,
                         excitatory_only=True,
                         seed=args.seed).to(device)

    # Visualize sparse filters
    plot_filters(snn.enc_linear.weight, save_path=os.path.join(args.fig_dir, f'sparse_p{args.p_connect}_filters.png'))

    def extract_features(dataloader, split_name: str):
        X_list, y_list = [], []
        pbar = tqdm(dataloader, desc=f"{split_name} feature extraction")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.numpy()
            spikes_in = poisson_encode(images, num_steps=args.num_steps, f_max_hz=args.fmax_hz,
                                       dt_ms=args.dt_ms, flatten=True)
            spk_h, mem_h = snn.simulate(spikes_in)
            if split_name == 'train' and len(X_list) == 0:
                plot_raster(spk_h.detach().cpu(), title=f'Sparse p={args.p_connect} Raster (train sample)',
                            save_path=os.path.join(args.fig_dir, f'sparse_p{args.p_connect}_raster.png'))
                plot_voltage_traces(mem_h.detach().cpu(), neuron_indices=list(range(8)),
                                    title=f'Sparse p={args.p_connect} Membrane (train sample)',
                                    save_path=os.path.join(args.fig_dir, f'sparse_p{args.p_connect}_voltage.png'))
            counts = spike_counts(spk_h).detach().cpu().numpy()
            X_list.append(counts)
            y_list.append(labels)
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return X, y

    X_train, y_train = extract_features(train_loader, 'train')
    clf = train_readout(X_train, y_train, model_type=args.readout)

    X_test, y_test = extract_features(test_loader, 'test')
    acc = evaluate_readout(clf, X_test, y_test)
    print(f"Sparse p={args.p_connect} Test Accuracy: {acc:.4f}")

    # Save report
    with open(os.path.join(args.output_dir, f'sparse_p{args.p_connect}_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Params: hidden={args.hidden}, tau_ms={args.tau_ms}, dt_ms={args.dt_ms}, v_th={args.v_th}\n")
        f.write(f"Weights: N({args.w_mean}, {args.w_std}), p_connect={args.p_connect}, excitatory-only\n")
        f.write(f"Encoding: T={args.num_steps}, fmax={args.fmax_hz} Hz\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.data')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--fig_dir', type=str, default='figures')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--tau_ms', type=float, default=20.0)
    parser.add_argument('--dt_ms', type=float, default=1.0)
    parser.add_argument('--t_ref_ms', type=float, default=2.0)
    parser.add_argument('--v_th', type=float, default=1.0)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--fmax_hz', type=float, default=100.0)
    parser.add_argument('--w_mean', type=float, default=0.5)
    parser.add_argument('--w_std', type=float, default=0.1)
    parser.add_argument('--readout', type=str, default='logreg', choices=['logreg', 'ridge'])
    parser.add_argument('--p_connect', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    run_sparse(args)
