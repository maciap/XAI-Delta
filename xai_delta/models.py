"""Loading the reference model M_pre and the fine-tuned model M_post (base + LoRA adapter)."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(base_id):
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _device_map(device):
    return {"": device.index if device.index is not None else 0}


def load_base(base_id, device, dtype=torch.bfloat16):
    """Base checkpoint without adapter."""
    return AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=dtype, device_map=_device_map(device), trust_remote_code=True
    ).eval()


def load_merged(base_id, adapter_id, device, dtype=torch.bfloat16):
    """Base checkpoint with the LoRA adapter merged into the weights."""
    base = load_base(base_id, device, dtype)
    return PeftModel.from_pretrained(base, adapter_id).eval().merge_and_unload().eval()


def load_pre_post(base_id, adapter_id, device, dtype=torch.bfloat16):
    """Load M_pre and M_post as two independent model objects."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available: this pipeline requires an NVIDIA GPU.")
    tok = load_tokenizer(base_id)
    M_pre = load_base(base_id, device, dtype)
    M_post = load_merged(base_id, adapter_id, device, dtype)
    return tok, M_pre, M_post


class AdapterDisabledView:
    """Proxy that runs a PeftModel with its adapter disabled, so that M_pre and
    M_post can share one copy of the base weights."""

    def __init__(self, peft_model):
        object.__setattr__(self, "_m", peft_model)

    def __call__(self, *args, **kwargs):
        with self._m.disable_adapter():
            return self._m(*args, **kwargs)

    def generate(self, *args, **kwargs):
        with self._m.disable_adapter():
            return self._m.generate(*args, **kwargs)

    def eval(self):
        self._m.eval()
        return self

    def __getattr__(self, name):
        return getattr(self._m, name)


def load_pre_post_shared_base(base_id, adapter_id, device, dtype=torch.bfloat16):
    """Memory-efficient variant of load_pre_post for large models: the base weights
    are loaded once and M_pre is the same model with the adapter switched off."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available: this pipeline requires an NVIDIA GPU.")
    tok = load_tokenizer(base_id)
    M_post = PeftModel.from_pretrained(load_base(base_id, device, dtype), adapter_id).eval()
    return tok, AdapterDisabledView(M_post), M_post


def load_models(cfg, device):
    """Load (tokenizer, M_pre, M_post) with the strategy given in the model config."""
    loader = load_pre_post if cfg["loading"] == "merge" else load_pre_post_shared_base
    return loader(cfg["base_id"], cfg["adapter_id"], device)
