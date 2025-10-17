import os
import argparse
from tqdm import tqdm
import numpy as np
import torch

from src.datasets import get_mnist_dataloaders
from src.encoding import poisson_encode, spike_counts
from src.model import FeedforwardSNN, tau_to_beta
from src.readout import train_readout, evaluate_readout, scale_features
from src.plotting import plot_raster, plot_voltage_traces, plot_filters
from src.utils import get_device, ensure_dir, set_seed


def run_baseline(args):
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
                         sparsity_p=None,
                         weight_mean=args.w_mean,
                         weight_std=args.w_std,
                         excitatory_only=True,
                         seed=args.seed)
    snn.to(device)

    # Visualize initial filters
    plot_filters(snn.enc_linear.weight, save_path=os.path.join(args.fig_dir, 'baseline_initial_filters.png'))

    def extract_features(dataloader, split_name: str):
        X_list, y_list = [], []
        pbar = tqdm(dataloader, desc=f"{split_name} feature extraction")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.numpy()
            spikes_in = poisson_encode(images, num_steps=args.num_steps, f_max_hz=args.fmax_hz,
                                       dt_ms=args.dt_ms, flatten=True)
            spk_h, mem_h = snn.simulate(spikes_in)
            # Visualize a sample batch once
            if split_name == 'train' and len(X_list) == 0:
                plot_raster(spk_h.detach().cpu(), title='Hidden Spike Raster (train sample)',
                            save_path=os.path.join(args.fig_dir, 'baseline_raster.png'))
                plot_voltage_traces(mem_h.detach().cpu(), neuron_indices=list(range(8)),
                                    title='Hidden Membrane Traces (train sample)',
                                    save_path=os.path.join(args.fig_dir, 'baseline_voltage.png'))
            counts = spike_counts(spk_h).detach().cpu().numpy()  # [B, N_hidden]
            X_list.append(counts)
            y_list.append(labels)
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return X, y

    X_train, y_train = extract_features(train_loader, 'train')
    X_test, y_test = extract_features(test_loader, 'test')

    # Optional feature scaling
    if args.scale_features:
        X_train_s, X_test_s, _ = scale_features(X_train, X_test)
    else:
        X_train_s, X_test_s = X_train, X_test

    clf = train_readout(
        X_train_s,
        y_train,
        model_type=args.readout,
        random_state=args.seed,
        max_iter=args.max_iter,
        solver=args.solver,
        C=args.C,
        penalty=args.penalty if args.penalty != 'none' else None,
    )

    acc = evaluate_readout(clf, X_test_s, y_test)
    print(f"Baseline Test Accuracy: {acc:.4f}")

    # Save report
    with open(os.path.join(args.output_dir, 'baseline_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Params: hidden={args.hidden}, tau_ms={args.tau_ms}, dt_ms={args.dt_ms}, v_th={args.v_th}\n")
        f.write(f"Weights: N( {args.w_mean}, {args.w_std}) excitatory-only\n")
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
    # readout hyperparams
    parser.add_argument('--scale_features', action='store_true', default=True)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--solver', type=str, default='lbfgs')
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--penalty', type=str, default='none', choices=['none', 'l2'])
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    run_baseline(args)
