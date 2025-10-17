import os
import argparse
from tqdm import tqdm
import numpy as np
import torch

from src.datasets import get_mnist_dataloaders
from src.encoding import poisson_encode, spike_counts
from src.model import FeedforwardSNN, tau_to_beta
from src.readout import train_readout, evaluate_readout
from src.plotting import plot_raster, plot_voltage_traces
from src.utils import get_device, ensure_dir, set_seed


def run_homeostasis(args):
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
                         seed=args.seed).to(device)

    # Homeostatic gains per hidden neuron
    gains = torch.ones((args.hidden,), device=device)
    rate_ema = torch.zeros((args.hidden,), device=device)
    alpha = args.ema_alpha  # EMA smoothing for firing rate
    target_rate = args.target_hz  # Hz
    dt_s = args.dt_ms / 1000.0

    def adapt_on_loader(loader):
        nonlocal gains, rate_ema
        pbar = tqdm(range(args.adapt_epochs), desc="Homeostasis epochs")
        for _ in pbar:
            inner = tqdm(loader, leave=False, desc="Adapting")
            for images, _ in inner:
                images = images.to(device)
                spikes_in = poisson_encode(images, num_steps=args.num_steps, f_max_hz=args.fmax_hz,
                                           dt_ms=args.dt_ms, flatten=True)
                # simulate with gains scaling current
                T, B, _ = spikes_in.shape
                state = snn.lif.reset_state(B, device)
                spk_accum = torch.zeros((args.hidden,), device=device)
                for t in range(T):
                    cur = snn.enc_linear(spikes_in[t]) * gains  # scale per-neuron
                    spk_h, mem_h, refrac = snn.lif(cur, state)
                    state = (spk_h, mem_h, refrac)
                    spk_accum += spk_h.sum(dim=0)
                # spikes per second per neuron (avg per image)
                spikes_per_image = spk_accum / B  # per neuron
                rate_hz = spikes_per_image / (args.num_steps * dt_s)
                rate_ema = alpha * rate_ema + (1 - alpha) * rate_hz
                # multiplicative update towards target
                gains = gains * (1.0 + args.eta * (target_rate - rate_ema))
                gains = torch.clamp(gains, min=args.g_min, max=args.g_max)
        return gains, rate_ema

    gains, rate_ema = adapt_on_loader(train_loader)

    # Evaluate features and train readout with adapted gains
    def extract_features(dataloader, split_name: str):
        X_list, y_list = [], []
        pbar = tqdm(dataloader, desc=f"{split_name} feature extraction")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.numpy()
            spikes_in = poisson_encode(images, num_steps=args.num_steps, f_max_hz=args.fmax_hz,
                                       dt_ms=args.dt_ms, flatten=True)
            # simulate with adapted gains
            T, B, _ = spikes_in.shape
            state = snn.lif.reset_state(B, device)
            spk_list, mem_list = [], []
            for t in range(T):
                cur = snn.enc_linear(spikes_in[t]) * gains
                spk_h, mem_h, refrac = snn.lif(cur, state)
                state = (spk_h, mem_h, refrac)
                spk_list.append(spk_h)
                mem_list.append(mem_h)
            spk_h = torch.stack(spk_list, dim=0)
            mem_h = torch.stack(mem_list, dim=0)
            if split_name == 'train' and len(X_list) == 0:
                plot_raster(spk_h.detach().cpu(), title='Homeostasis Spike Raster (train sample)',
                            save_path=os.path.join(args.fig_dir, 'homeostasis_raster.png'))
                plot_voltage_traces(mem_h.detach().cpu(), neuron_indices=list(range(8)),
                                    title='Homeostasis Membrane Traces (train sample)',
                                    save_path=os.path.join(args.fig_dir, 'homeostasis_voltage.png'))
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
    print(f"Homeostasis Test Accuracy: {acc:.4f}")

    # Save report
    with open(os.path.join(args.output_dir, 'homeostasis_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Params: hidden={args.hidden}, tau_ms={args.tau_ms}, dt_ms={args.dt_ms}, v_th={args.v_th}\n")
        f.write(f"Encoding: T={args.num_steps}, fmax={args.fmax_hz} Hz\n")
        f.write(f"Homeostasis: target_hz={args.target_hz}, eta={args.eta}, ema_alpha={args.ema_alpha}\n")


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
    # homeostasis params
    parser.add_argument('--target_hz', type=float, default=10.0)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--ema_alpha', type=float, default=0.9)
    parser.add_argument('--g_min', type=float, default=0.1)
    parser.add_argument('--g_max', type=float, default=10.0)
    parser.add_argument('--adapt_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    run_homeostasis(args)
