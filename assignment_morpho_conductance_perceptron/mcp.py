from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Sequence, Tuple, Optional

import numpy as np
import argparse
import csv
try:
    from tqdm import trange
except Exception:
    trange = None  # fallback: no progress bar
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # optional plotting


@dataclass
class BranchSpec:
    exc_idx: List[int]
    inh_idx: List[int]


class MorphoConductancePerceptron:
    """
    Morpho-Conductance Perceptron (MCP): a simple perceptron variant with
    branch-local shunting (divisive) conductance and plateau augmentation.

    For branch b:
      sE = sum_i wE[i] * x[i]  over exc_idx[b]
      sI = sum_i wI[i] * x[i]  over inh_idx[b]
      g  = sigmoid(alpha_b * sI + beta_b)  in (0, 1)
      plateau = rho_b * relu(sE - theta_b)
      y_b = sE / (1 + g) + plateau

    Soma: V = sum_b y_b + bias;  y = sigmoid(V)
    Trained via simple gradient descent (MSE loss by default).
    """

    def __init__(
        self,
        num_inputs: int,
        branches: Sequence[BranchSpec],
        lr: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.num_inputs = int(num_inputs)
        self.branches = list(branches)
        self.lr = float(lr)
        self.rng = np.random.default_rng(seed)

        # Weights for excitatory and inhibitory channels (per input)
        # stronger init to avoid near-zero sE/sI plateaus
        self.wE = self.rng.normal(0.0, 0.5, size=self.num_inputs)
        self.wI = self.rng.normal(0.0, 0.5, size=self.num_inputs)

        # Per-branch ion-like/shunt/plateau parameters
        B = len(self.branches)
        self.alpha = np.abs(self.rng.normal(1.5, 0.2, size=B))  # stronger shunt sensitivity >=0
        self.beta = self.rng.normal(0.0, 0.1, size=B)           # shunt offset
        self.rho = np.abs(self.rng.normal(0.6, 0.1, size=B))    # plateau gain >=0
        self.theta = self.rng.normal(0.2, 0.05, size=B)       # plateau threshold

        self.bias = 0.0

        # Ablation controls
        self.disable_shunt = False            # if True, g := 0
        self.disable_plateau = False          # if True, plateau := 0
        self.disable_plateau_gating = False   # if True, plateau not multiplied by (1-g)

    def set_ablations(self, *, disable_shunt=False, disable_plateau=False, disable_plateau_gating=False) -> None:
        self.disable_shunt = bool(disable_shunt)
        self.disable_plateau = bool(disable_plateau)
        self.disable_plateau_gating = bool(disable_plateau_gating)

    @staticmethod
    def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _relu(z: np.ndarray | float) -> np.ndarray | float:
        return np.maximum(z, 0.0)

    def forward(self, x: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        x = np.asarray(x, dtype=float).reshape(-1)
        assert x.shape[0] == self.num_inputs

        B = len(self.branches)
        sE = np.zeros(B)
        sI = np.zeros(B)
        g = np.zeros(B)
        plateau = np.zeros(B)
        yb = np.zeros(B)

        for b, spec in enumerate(self.branches):
            if spec.exc_idx:
                sE[b] = float(np.dot(self.wE[spec.exc_idx], x[spec.exc_idx]))
            if spec.inh_idx:
                sI[b] = float(np.dot(self.wI[spec.inh_idx], x[spec.inh_idx]))
            g_val = float(self._sigmoid(self.alpha[b] * sI[b] + self.beta[b]))
            if self.disable_shunt:
                g_val = 0.0
            g[b] = g_val
            # Plateau gated by inhibition (1 - g): inhibition suppresses plateau (unless ablated)
            plat_core = float(self.rho[b] * self._relu(sE[b] - self.theta[b]))
            if self.disable_plateau:
                plat_term = 0.0
            else:
                plat_term = plat_core if self.disable_plateau_gating else plat_core * (1.0 - g[b])
            plateau[b] = plat_term
            yb[b] = sE[b] / (1.0 + g[b]) + plateau[b]

        V = float(np.sum(yb) + self.bias)
        y = float(self._sigmoid(V))

        cache = {
            "x": x, "sE": sE, "sI": sI, "g": g, "plateau": plateau, "yb": yb, "V": V, "y": y
        }
        return y, cache

    def learn(self, x: np.ndarray, target: float) -> float:
        # Forward
        y, c = self.forward(x)
        t = float(target)
        # Binary cross-entropy loss with sigmoid: dL/dV = y - t
        eps = 1e-8
        loss = -(t * math.log(y + eps) + (1.0 - t) * math.log(1.0 - y + eps))
        dLdV = (y - t)

        # Gradients accumulation
        d_wE = np.zeros_like(self.wE)
        d_wI = np.zeros_like(self.wI)
        d_alpha = np.zeros_like(self.alpha)
        d_beta = np.zeros_like(self.beta)
        d_rho = np.zeros_like(self.rho)
        d_theta = np.zeros_like(self.theta)
        d_bias = dLdV

        for b, spec in enumerate(self.branches):
            sE = c["sE"][b]
            sI = c["sI"][b]
            g = c["g"][b]

            # dyb/dsE and dyb/dsI
            relu_mask = 1.0 if sE > self.theta[b] else 0.0
            sigp = g * (1.0 - g)
            dgdsi = sigp * self.alpha[b]  # sigmoid'(a*sI+b) * a
            # y = sE/(1+g) + rho*relu(sE-theta)*(1 - g)
            if self.disable_plateau:
                plat_sE_term = 0.0
                plat_g_term = 0.0
            elif self.disable_plateau_gating:
                plat_sE_term = self.rho[b] * relu_mask
                plat_g_term = 0.0
            else:
                plat_sE_term = self.rho[b] * relu_mask * (1.0 - g)
                plat_g_term = - self.rho[b] * (self._relu(sE - self.theta[b]))

            dyb_d_sE = 1.0 / (1.0 + g) + plat_sE_term
            # dy/dsI via g: dy/dg = -sE/(1+g)^2 + plat_g_term
            dyg_dg = -sE / ((1.0 + g) ** 2) + plat_g_term
            dyb_d_sI = dyg_dg * dgdsi

            # Propagate to input weights on this branch
            for i in spec.exc_idx:
                d_wE[i] += dLdV * dyb_d_sE * x[i]
            for i in spec.inh_idx:
                d_wI[i] += dLdV * dyb_d_sI * x[i]

            # Branch parameter grads
            # via g: dy/d(alpha) = (dy/dg) * sig'(.) * sI
            d_alpha[b] += dLdV * (dyg_dg * sigp * sI)
            d_beta[b]  += dLdV * (dyg_dg * sigp * 1.0)
            # plateau params
            if not self.disable_plateau:
                gate = 1.0 if self.disable_plateau_gating else (1.0 - g)
                d_rho[b]   += dLdV * (self._relu(sE - self.theta[b]) * gate)
                d_theta[b] += dLdV * (-self.rho[b] * relu_mask * gate)

        # Parameter update (SGD)
        self.wE -= self.lr * d_wE
        self.wI -= self.lr * d_wI
        self.alpha -= self.lr * d_alpha
        self.beta -= self.lr * d_beta
        self.rho -= self.lr * d_rho
        self.theta -= self.lr * d_theta
        # simple projection to keep parameters in sensible ranges
        self.alpha = np.maximum(self.alpha, 0.0)
        self.rho = np.maximum(self.rho, 0.0)
        self.bias -= self.lr * d_bias

        return float(loss)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 2000, *, verbose: bool = False, bar_desc: str = "MCP", log_every: int = 0) -> List[float]:
        losses: List[float] = []
        if verbose and trange is not None:
            pbar = trange(epochs, desc=bar_desc)
            for e in pbar:
                idx = np.random.permutation(len(X))
                epoch_losses = []
                for k in idx:
                    loss = self.learn(X[k], float(y[k]))
                    losses.append(loss)
                    epoch_losses.append(loss)
                if log_every and ((e + 1) % log_every == 0):
                    avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
                    pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        else:
            for e in range(epochs):
                idx = np.random.permutation(len(X))
                epoch_losses = []
                for k in idx:
                    loss = self.learn(X[k], float(y[k]))
                    losses.append(loss)
                    epoch_losses.append(loss)
                if verbose and log_every and ((e + 1) % log_every == 0):
                    avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
                    print(f"[{bar_desc}] epoch {e+1}/{epochs} avg_loss={avg_loss:.4f}", flush=True)
        return losses

    # Convenience: deterministic initialization for XOR in 2D with two branches.
    def init_xor_params(self) -> None:
        assert self.num_inputs == 2 and len(self.branches) == 2
        # Branch 0: excite x0, shunt via x1
        # Branch 1: excite x1, shunt via x0
        self.wE[:] = 0.0
        self.wI[:] = 0.0
        self.wE[0] = 2.0
        self.wE[1] = 2.0
        self.wI[0] = 3.0
        self.wI[1] = 3.0
        # Strong shunting sensitivity; low offset
        self.alpha[:] = 5.0
        self.beta[:] = 0.0
        # Plateau boosts singles; gated off by inhibition when both active
        self.rho[:] = 1.0
        self.theta[:] = 0.5
        # Bias to keep 0,0 below threshold
        self.bias = -1.5

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        preds = []
        for x in X:
            yhat, _ = self.forward(x)
            preds.append(1.0 if yhat >= threshold else 0.0)
        return np.array(preds)


# ------------------------------- Demo -------------------------------------- #

def _xor_demo():
    # XOR in 2D
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ], dtype=float)
    y = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)

    # Two branches: each one excites on one feature and shunts via the other
    branches = [
        BranchSpec(exc_idx=[0], inh_idx=[1]),
        BranchSpec(exc_idx=[1], inh_idx=[0]),
    ]

    # Start from a deterministic XOR-friendly initialization, then (optionally) fine-tune
    m = MorphoConductancePerceptron(num_inputs=2, branches=branches, lr=0.05, seed=0)
    m.init_xor_params()
    # Optional fine-tuning (short)
    m.fit(X, y, epochs=200)
    preds = m.predict(X)
    acc = (preds == y).mean()
    print("MCP XOR accuracy:", acc)
    for xi, yi, pi in zip(X, y, preds):
        yh, cache = m.forward(xi)
        print(f"x={xi}, y*={int(yi)}, yhat={yh:.3f}, pred={int(pi)}")


# --------------------------- Baseline and Benchmarks ----------------------- #

class StandardPerceptron:
    """Sigmoid perceptron (logistic regression) baseline."""
    def __init__(self, num_inputs: int, lr: float = 0.1, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.1, size=num_inputs)
        self.b = 0.0
        self.lr = float(lr)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        return float(self._sigmoid(np.dot(self.w, x) + self.b))

    def learn(self, x: np.ndarray, t: float) -> float:
        y = self.forward(x)
        eps = 1e-8
        loss = -(t * np.log(y + eps) + (1 - t) * np.log(1 - y + eps))
        dLdV = y - t
        self.w -= self.lr * dLdV * x
        self.b -= self.lr * dLdV
        return float(loss)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 500, *, verbose: bool = False, bar_desc: str = "Perceptron", log_every: int = 0) -> None:
        if verbose and trange is not None:
            pbar = trange(epochs, desc=bar_desc)
            for e in pbar:
                idx = np.random.permutation(len(X))
                epoch_losses = []
                for k in idx:
                    epoch_losses.append(self.learn(X[k], float(y[k])))
                if log_every and ((e + 1) % log_every == 0):
                    avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
                    pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        else:
            for e in range(epochs):
                idx = np.random.permutation(len(X))
                epoch_losses = []
                for k in idx:
                    epoch_losses.append(self.learn(X[k], float(y[k])))
                if verbose and log_every and ((e + 1) % log_every == 0):
                    avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
                    print(f"[{bar_desc}] epoch {e+1}/{epochs} avg_loss={avg_loss:.4f}", flush=True)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return np.array([1.0 if self.forward(x) >= threshold else 0.0 for x in X])


def gen_branch_xor_dataset(n: int = 400, noise: float = 0.1, seed: int | None = 0):
    """Generate a 4D dataset with two branches performing XOR at branch level.
    - Inputs: x0,x1 (branch A), x2,x3 (branch B). Each in {0,1} with additive Gaussian noise.
    - Label: 1 if exactly one branch is active (sum > 0.5), else 0.
    """
    rng = np.random.default_rng(seed)
    Xb = rng.integers(0, 2, size=(n, 4)).astype(float)
    if noise > 0:
        X = Xb + rng.normal(0.0, noise, size=Xb.shape)
    else:
        X = Xb
    sA = Xb[:, 0] + Xb[:, 1]
    sB = Xb[:, 2] + Xb[:, 3]
    y = ((sA > 0) ^ (sB > 0)).astype(float)
    return X, y


def _mcp_for_4d(branches: Sequence[BranchSpec], lr: float = 0.05, seed: int | None = None) -> MorphoConductancePerceptron:
    m = MorphoConductancePerceptron(num_inputs=4, branches=branches, lr=lr, seed=seed)
    return m


def run_benchmarks(*, plot: bool = False, outdir: Optional[str] = None):
    # Dataset
    X, y = gen_branch_xor_dataset(n=800, noise=0.15, seed=0)
    # Train/test split
    n = len(X)
    idx = np.random.default_rng(0).permutation(n)
    split = int(0.7 * n)
    tr_idx, te_idx = idx[:split], idx[split:]
    Xtr, Ytr = X[tr_idx], y[tr_idx]
    Xte, Yte = X[te_idx], y[te_idx]

    # Branch layout: A=(0,1), B=(2,3)
    branches = [
        BranchSpec(exc_idx=[0, 1], inh_idx=[2, 3]),
        BranchSpec(exc_idx=[2, 3], inh_idx=[0, 1]),
    ]

    # Baseline perceptron
    print("[RUN] Training baseline perceptron...", flush=True)
    sp = StandardPerceptron(num_inputs=4, lr=0.1, seed=0)
    sp.fit(Xtr, Ytr, epochs=1000, verbose=True, bar_desc="Baseline Perceptron", log_every=50)
    print("[DONE] Baseline perceptron.", flush=True)
    sp_tr = (sp.predict(Xtr) == Ytr).mean()
    sp_te = (sp.predict(Xte) == Yte).mean()

    # MCP full
    print("[RUN] Training MCP (full)...", flush=True)
    m_full = _mcp_for_4d(branches, lr=0.05, seed=0)
    m_full.fit(Xtr, Ytr, epochs=2000, verbose=True, bar_desc="MCP full", log_every=100)
    print("[DONE] MCP (full).", flush=True)
    full_tr = (m_full.predict(Xtr) == Ytr).mean()
    full_te = (m_full.predict(Xte) == Yte).mean()

    # Parameter inspection (summary)
    def _summ(a):
        return float(np.mean(a)), float(np.std(a)), float(np.linalg.norm(a))
    print("\n[MCP full] Parameter summaries:")
    mu, sd, nm = _summ(m_full.wE)
    print(f" wE: mean={mu:.3f} std={sd:.3f} norm={nm:.3f}")
    mu, sd, nm = _summ(m_full.wI)
    print(f" wI: mean={mu:.3f} std={sd:.3f} norm={nm:.3f}")
    mu, sd, nm = _summ(m_full.alpha)
    print(f" alpha: mean={mu:.3f} std={sd:.3f} norm={nm:.3f}")
    mu, sd, nm = _summ(m_full.beta)
    print(f" beta: mean={mu:.3f} std={sd:.3f} norm={nm:.3f}")
    mu, sd, nm = _summ(m_full.rho)
    print(f" rho: mean={mu:.3f} std={sd:.3f} norm={nm:.3f}")
    mu, sd, nm = _summ(m_full.theta)
    print(f" theta: mean={mu:.3f} std={sd:.3f} norm={nm:.3f}")

    # Ablations
    print("[RUN] Training MCP (no shunt)...", flush=True)
    m_no_shunt = _mcp_for_4d(branches, lr=0.05, seed=1)
    m_no_shunt.set_ablations(disable_shunt=True)
    m_no_shunt.fit(Xtr, Ytr, epochs=2000, verbose=True, bar_desc="MCP no shunt", log_every=100)
    print("[DONE] MCP (no shunt).", flush=True)
    no_shunt_tr = (m_no_shunt.predict(Xtr) == Ytr).mean()
    no_shunt_te = (m_no_shunt.predict(Xte) == Yte).mean()

    print("[RUN] Training MCP (no plateau)...", flush=True)
    m_no_plat = _mcp_for_4d(branches, lr=0.05, seed=2)
    m_no_plat.set_ablations(disable_plateau=True)
    m_no_plat.fit(Xtr, Ytr, epochs=2000, verbose=True, bar_desc="MCP no plateau", log_every=100)
    print("[DONE] MCP (no plateau).", flush=True)
    no_plat_tr = (m_no_plat.predict(Xtr) == Ytr).mean()
    no_plat_te = (m_no_plat.predict(Xte) == Yte).mean()

    print("[RUN] Training MCP (no gating)...", flush=True)
    m_no_gate = _mcp_for_4d(branches, lr=0.05, seed=3)
    m_no_gate.set_ablations(disable_plateau_gating=True)
    m_no_gate.fit(Xtr, Ytr, epochs=2000, verbose=True, bar_desc="MCP no gating", log_every=100)
    print("[DONE] MCP (no gating).", flush=True)
    no_gate_tr = (m_no_gate.predict(Xtr) == Ytr).mean()
    no_gate_te = (m_no_gate.predict(Xte) == Yte).mean()

    print("\nBenchmark: 4D Branch-XOR with noise=0.15 (70/30 split)")
    print(f"Standard perceptron acc  - train: {sp_tr:.3f}  test: {sp_te:.3f}")
    print(f"MCP (full) acc           - train: {full_tr:.3f}  test: {full_te:.3f}")
    print(f"MCP ablation - no shunt  - train: {no_shunt_tr:.3f}  test: {no_shunt_te:.3f}")
    print(f"MCP ablation - no plateau- train: {no_plat_tr:.3f}  test: {no_plat_te:.3f}")
    print(f"MCP ablation - no gating - train: {no_gate_tr:.3f}  test: {no_gate_te:.3f}")

    # Optional plot: bar chart of test accuracies
    if plot:
        if plt is None:
            print("[WARN] matplotlib not available; skipping plots.")
        else:
            labels = [
                "Perceptron",
                "MCP full",
                "No shunt",
                "No plateau",
                "No gating",
            ]
            test_vals = [sp_te, full_te, no_shunt_te, no_plat_te, no_gate_te]
            plt.figure(figsize=(7,4))
            bars = plt.bar(labels, test_vals, color=["#8888ff","#44aa44","#cc6666","#cc9966","#6699cc"]) 
            plt.ylim(0.5, 1.05)
            plt.ylabel("Test accuracy")
            plt.title("Branch-XOR (noise=0.15): Ablation comparison")
            for b, v in zip(bars, test_vals):
                plt.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
            plt.tight_layout()
            if outdir:
                import os
                os.makedirs(outdir, exist_ok=True)
                path = os.path.join(outdir, "ablation_bar.png")
                plt.savefig(path, dpi=150)
                print(f"[SAVED] Plot -> {path}")
            else:
                plt.show()


# -------------------- Advanced Evaluations: multi-seed, noise sweep -------- #

def _evaluate_models_on_dataset(X: np.ndarray, y: np.ndarray, branches: Sequence[BranchSpec], *, seed: int,
                                perc_epochs: int = 1000, mcp_epochs: int = 2000) -> dict:
    # split
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    split = int(0.7 * n)
    Xtr, Ytr = X[idx[:split]], y[idx[:split]]
    Xte, Yte = X[idx[split:]], y[idx[split:]]
    # baseline
    sp = StandardPerceptron(num_inputs=X.shape[1], lr=0.1, seed=seed)
    sp.fit(Xtr, Ytr, epochs=perc_epochs, verbose=False)
    sp_tr = (sp.predict(Xtr) == Ytr).mean()
    sp_te = (sp.predict(Xte) == Yte).mean()
    # mcp full
    m_full = _mcp_for_4d(branches, lr=0.05, seed=seed)
    m_full.fit(Xtr, Ytr, epochs=mcp_epochs, verbose=False)
    mf_tr = (m_full.predict(Xtr) == Ytr).mean()
    mf_te = (m_full.predict(Xte) == Yte).mean()
    return {
        "sp_tr": float(sp_tr), "sp_te": float(sp_te),
        "mcp_tr": float(mf_tr), "mcp_te": float(mf_te),
    }


def run_multi_seed(num_seeds: int = 10, noise: float = 0.15, *, csv_path: str | None = None) -> None:
    print(f"[RUN] Multi-seed evaluation: seeds={num_seeds}, noise={noise}", flush=True)
    X, y = gen_branch_xor_dataset(n=800, noise=noise, seed=0)
    branches = [
        BranchSpec(exc_idx=[0, 1], inh_idx=[2, 3]),
        BranchSpec(exc_idx=[2, 3], inh_idx=[0, 1]),
    ]
    results = []
    for s in range(num_seeds):
        res = _evaluate_models_on_dataset(X, y, branches, seed=s)
        res["seed"] = s
        res["noise"] = float(noise)
        results.append(res)
    # aggregate
    def agg(key):
        vals = np.array([r[key] for r in results], dtype=float)
        return float(np.mean(vals)), float(np.std(vals))
    sp_te_mu, sp_te_sd = agg("sp_te")
    mcp_te_mu, mcp_te_sd = agg("mcp_te")
    print("[RESULT] Multi-seed test acc:")
    print(f" Standard perceptron: mean={sp_te_mu:.3f} std={sp_te_sd:.3f}")
    print(f" MCP (full):         mean={mcp_te_mu:.3f} std={mcp_te_sd:.3f}")
    # optional CSV
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["mode","seed","noise","sp_tr","sp_te","mcp_tr","mcp_te"])
            w.writeheader()
            for r in results:
                row = {"mode":"multi_seed", **r}
                w.writerow(row)
        print(f"[SAVED] CSV -> {csv_path}")


def run_noise_sweep(noises: Sequence[float] = (0.0, 0.1, 0.2, 0.3), seeds: int = 5, *, csv_path: Optional[str] = None, plot: bool = False, outdir: Optional[str] = None) -> None:
    print(f"[RUN] Noise sweep: noises={list(noises)}, seeds={seeds}", flush=True)
    branches = [
        BranchSpec(exc_idx=[0, 1], inh_idx=[2, 3]),
        BranchSpec(exc_idx=[2, 3], inh_idx=[0, 1]),
    ]
    rows = []
    sp_curve = []
    mcp_curve = []
    for nz in noises:
        X, y = gen_branch_xor_dataset(n=800, noise=float(nz), seed=0)
        res = []
        for s in range(seeds):
            r = _evaluate_models_on_dataset(X, y, branches, seed=s)
            r["seed"] = s
            r["noise"] = float(nz)
            res.append(r)
            rows.append({"mode":"noise_sweep", **r})
        # aggregate per noise
        def agg(key):
            vals = np.array([rr[key] for rr in res], dtype=float)
            return float(np.mean(vals)), float(np.std(vals))
        sp_te_mu, sp_te_sd = agg("sp_te")
        mcp_te_mu, mcp_te_sd = agg("mcp_te")
        sp_curve.append((float(nz), sp_te_mu, sp_te_sd))
        mcp_curve.append((float(nz), mcp_te_mu, mcp_te_sd))
        print(f"[RESULT] noise={nz:.2f} | SP test mean={sp_te_mu:.3f}±{sp_te_sd:.3f} | MCP test mean={mcp_te_mu:.3f}±{mcp_te_sd:.3f}")
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["mode","seed","noise","sp_tr","sp_te","mcp_tr","mcp_te"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[SAVED] CSV -> {csv_path}")
    if plot:
        if plt is None:
            print("[WARN] matplotlib not available; skipping plots.")
        else:
            nz = [n for n,_,_ in sp_curve]
            sp_mu = [m for _,m,_ in sp_curve]
            sp_sd = [s for _,_,s in sp_curve]
            mcp_mu = [m for _,m,_ in mcp_curve]
            mcp_sd = [s for _,_,s in mcp_curve]
            plt.figure(figsize=(6,4))
            plt.errorbar(nz, sp_mu, yerr=sp_sd, fmt='-o', label='Perceptron')
            plt.errorbar(nz, mcp_mu, yerr=mcp_sd, fmt='-o', label='MCP full')
            plt.ylim(0.5, 1.05)
            plt.xlabel("Noise std")
            plt.ylabel("Test accuracy (mean±std)")
            plt.title("Noise robustness sweep")
            plt.legend()
            plt.tight_layout()
            if outdir:
                import os
                os.makedirs(outdir, exist_ok=True)
                path = os.path.join(outdir, "noise_sweep.png")
                plt.savefig(path, dpi=150)
                print(f"[SAVED] Plot -> {path}")
            else:
                plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="Run benchmark and ablation suite")
    parser.add_argument("--multi-seed", type=int, default=0, help="Run multi-seed evaluation with given number of seeds")
    parser.add_argument("--noise-sweep", type=str, default="", help="Comma-separated noise values, e.g., 0.0,0.1,0.2")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV output path for multi-seed or noise sweep")
    parser.add_argument("--plot", action="store_true", help="Show/Save plots for benchmarks or sweeps")
    parser.add_argument("--outdir", type=str, default="", help="Directory to save plots if --plot is set; otherwise shows interactively")
    args = parser.parse_args()
    if args.bench:
        run_benchmarks(plot=args.plot, outdir=(args.outdir or None))
    ran_extra = False
    if args.multi_seed and args.multi_seed > 0:
        ran_extra = True
        csv_path = args.csv if args.csv else None
        run_multi_seed(num_seeds=int(args.multi_seed), csv_path=csv_path)
    if args.noise_sweep:
        ran_extra = True
        nz = [float(s.strip()) for s in args.noise_sweep.split(',') if s.strip()]
        csv_path = args.csv if args.csv else None
        run_noise_sweep(noises=nz, csv_path=csv_path, plot=args.plot, outdir=(args.outdir or None))
    if not args.bench and not ran_extra:
        _xor_demo()
