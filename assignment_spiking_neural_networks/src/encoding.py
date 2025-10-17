from typing import Tuple
import torch

def poisson_encode(images: torch.Tensor,
                   num_steps: int,
                   f_max_hz: float = 100.0,
                   dt_ms: float = 1.0,
                   flatten: bool = True) -> torch.Tensor:
    """
    Convert batch of grayscale images in [0,1] to Poisson spike trains.

    Args:
        images: shape [batch, 1, 28, 28], values in [0,1]
        num_steps: number of time steps (T_enc / dt)
        f_max_hz: maximum firing rate corresponding to pixel value 1.0
        dt_ms: timestep in ms
        flatten: if True, flatten spatial dims to N_in

    Returns:
        spikes: shape [num_steps, batch, N_in]
    """
    b = images.shape[0]
    if flatten:
        x = images.view(b, -1)
    else:
        x = images.squeeze(1)  # [b,28,28]
    # Probability of spike per dt: p = rate*dt
    dt_s = dt_ms / 1000.0
    p = torch.clamp(x * f_max_hz * dt_s, 0.0, 1.0)
    # Generate spikes across time
    # [num_steps, batch, features]
    rand = torch.rand((num_steps, b, p.shape[1]), device=images.device)
    spikes = (rand < p.unsqueeze(0)).to(images.dtype)
    return spikes


def spike_counts(spikes: torch.Tensor) -> torch.Tensor:
    """
    Sum spikes across time.
    Args:
        spikes: [T, B, N]
    Returns:
        counts: [B, N]
    """
    return spikes.sum(dim=0)
