from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import snntorch as snn


class RandomLinear(nn.Module):
    """
    Fixed random linear layer with optional sparse connectivity.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 mean: float = 0.5, std: float = 0.1, excitatory_only: bool = True,
                 sparsity_p: Optional[float] = None, seed: int = 42):
        super().__init__()
        gen = torch.Generator()
        gen.manual_seed(seed)
        W = torch.normal(mean=mean, std=std, size=(out_features, in_features), generator=gen)
        if excitatory_only:
            W = torch.clamp(W, min=0.0)
        if sparsity_p is not None:
            rand = torch.rand(W.shape, generator=gen, device=W.device, dtype=W.dtype)
            mask = (rand < sparsity_p).float()
            W = W * mask
        self.register_buffer('weight', W)
        if bias:
            b = torch.zeros(out_features)
            self.register_buffer('bias', b)
        else:
            self.bias = None
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in]
        y = x @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias
        return y


def tau_to_beta(tau_ms: float, dt_ms: float) -> float:
    return math.exp(-dt_ms / tau_ms)


class LIFRefractory(nn.Module):
    """
    Wrapper around snn.Leaky with a simple refractory counter per neuron.
    During refractory, spikes are suppressed and membrane can decay but not spike.
    """
    def __init__(self, size: int, beta: float, threshold: float = 1.0,
                 reset_mechanism: str = 'subtract', refractory_steps: int = 0,
                 reset_delay: bool = True):
        super().__init__()
        self.size = size
        self.refractory_steps = int(max(0, refractory_steps))
        self.lif = snn.Leaky(beta=beta, threshold=threshold,
                             reset_mechanism=reset_mechanism,
                             reset_delay=reset_delay)

    def forward(self, input_t: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        # input_t: [B, size]
        if state is None:
            mem = torch.zeros((input_t.size(0), self.size), device=input_t.device)
            spk = torch.zeros_like(mem)
            refrac = torch.zeros_like(mem)
        else:
            spk, mem, refrac = state
        # Standard LIF step
        spk_raw, mem = self.lif(input_t, mem)
        # Apply refractory: block spikes where refrac>0
        in_refrac = (refrac > 0).float()
        spk = spk_raw * (1.0 - in_refrac)
        # Update refractory counters
        refrac = torch.clamp(refrac - 1, min=0)
        refrac = torch.where(spk > 0, torch.full_like(refrac, self.refractory_steps), refrac)
        return spk, mem, refrac

    def reset_state(self, batch_size: int, device: torch.device):
        mem = torch.zeros((batch_size, self.size), device=device)
        spk = torch.zeros_like(mem)
        refrac = torch.zeros_like(mem)
        return spk, mem, refrac


class FeedforwardSNN(nn.Module):
    """
    Input spikes (size N_in) -> fixed random weights -> LIF hidden -> spikes.
    """
    def __init__(self, n_in: int, n_hidden: int, beta: float, threshold: float = 1.0,
                 refractory_steps: int = 0, sparsity_p: Optional[float] = None,
                 weight_mean: float = 0.5, weight_std: float = 0.1, excitatory_only: bool = True,
                 seed: int = 42):
        super().__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.enc_linear = RandomLinear(n_in, n_hidden, bias=False,
                                       mean=weight_mean, std=weight_std,
                                       excitatory_only=excitatory_only,
                                       sparsity_p=sparsity_p, seed=seed)
        self.lif = LIFRefractory(size=n_hidden, beta=beta, threshold=threshold,
                                 refractory_steps=refractory_steps)

    @torch.no_grad()
    def simulate(self, spikes_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate over time.
        Args:
            spikes_in: [T, B, N_in]
        Returns:
            spk_h: [T, B, N_hidden]
            mem_h: [T, B, N_hidden]
        """
        T, B, _ = spikes_in.shape
        device = spikes_in.device
        spk_h_list, mem_h_list = [], []
        state = self.lif.reset_state(B, device)
        for t in range(T):
            x_t = spikes_in[t]
            cur = self.enc_linear(x_t)  # current input
            spk_h, mem_h, refrac = self.lif(cur, state)
            state = (spk_h, mem_h, refrac)
            spk_h_list.append(spk_h)
            mem_h_list.append(mem_h)
        spk_h = torch.stack(spk_h_list, dim=0)
        mem_h = torch.stack(mem_h_list, dim=0)
        return spk_h, mem_h
