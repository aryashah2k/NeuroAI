from typing import Tuple
import torch
import torch.nn.functional as F


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.view(-1)
    b = b.view(-1)
    an = torch.linalg.norm(a).item()
    bn = torch.linalg.norm(b).item()
    if an == 0.0 or bn == 0.0:
        return 0.0
    a = a / an
    b = b / bn
    return float((a * b).sum().item())
