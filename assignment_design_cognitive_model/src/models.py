import json
import platform
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM


@dataclass
class DeviceInfo:
    device: str
    torch_version: str
    cuda: bool
    cuda_name: Optional[str]

    @staticmethod
    def collect() -> "DeviceInfo":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        return DeviceInfo(
            device=device,
            torch_version=torch.__version__,
            cuda=torch.cuda.is_available(),
            cuda_name=cuda_name,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "torch_version": self.torch_version,
            "cuda": self.cuda,
            "cuda_name": self.cuda_name,
            "platform": platform.platform(),
        }


def get_device(device_str: Optional[str] = None) -> torch.device:
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def load_llama(model_id: str,
               device: Optional[str] = None,
               dtype: Optional[str] = None,
               trust_remote_code: bool = False,
               ) -> Tuple[LlamaForCausalLM, AutoTokenizer, AutoConfig, str]:
    """
    Load a LLaMA-family causal LM and tokenizer in eval mode.

    Returns model, tokenizer, config, and model revision hash if available.
    """
    torch.set_grad_enabled(False)

    if dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)

    device_obj = get_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
    # Ensure pad token exists for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    model = LlamaForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model.to(device_obj)

    # Try to get revision/commit hash
    revision = getattr(model.config, "_commit_hash", None) or getattr(model.generation_config, "_commit_hash", None)
    if revision is None:
        # Best-effort: store config hash
        revision = str(hash(json.dumps(config.to_dict(), sort_keys=True)))

    return model, tokenizer, config, revision
