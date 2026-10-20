"""Sparse autoencoder trained on pooled residual-stream activations of several checkpoints,
so that feature activations are expressed in one shared basis and can be compared."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from xai_delta.explainers.patching import find_blocks


@torch.no_grad()
def collect_layer_acts(model, tok, prompts, layer_idx):
    """Last-token output of block `layer_idx` for every (chat-formatted) prompt."""
    model.eval()
    acts = []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        acts.append(h[:, -1, :].detach().float().cpu())

    handle = find_blocks(model)[layer_idx].register_forward_hook(hook_fn)
    for p in tqdm(prompts):
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True)
        model(**tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device))
    handle.remove()
    return torch.cat(acts, dim=0)


def normalize_jointly(*acts):
    """Z-score every activation matrix with the mean/std of all of them pooled."""
    joint = torch.cat(acts, dim=0)
    mean = joint.mean(dim=0, keepdim=True)
    std = joint.std(dim=0, keepdim=True).clamp_min(1e-6)
    return [(A - mean) / std for A in acts]


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_in, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)

    def encode(self, x):
        return F.relu(self.encoder(x))

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z


def train_sae(A, device, expansion=8, batch_size=64, l1_coeff=1.0, lr=5e-4, epochs=400, seed=0):
    """Train an SAE with loss MSE(x, x_hat) + l1_coeff * mean|z|. Returns the model and
    the per-epoch average losses."""
    torch.manual_seed(seed)
    sae = SparseAutoencoder(A.shape[1], expansion * A.shape[1]).to(device)
    loader = DataLoader(TensorDataset(A), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr)
    loss_log = []

    for epoch in tqdm(range(epochs)):
        sae.train()
        total = total_recon = total_l1 = 0.0
        for (x,) in loader:
            x = x.to(device)
            x_hat, z = sae(x)
            recon, l1 = F.mse_loss(x_hat, x), z.abs().mean()
            loss = recon + l1_coeff * l1
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, total_recon, total_l1 = total + loss.item(), total_recon + recon.item(), total_l1 + l1.item()
        n = len(loader)
        loss_log.append({"epoch": epoch, "loss": total / n, "recon": total_recon / n, "l1": total_l1 / n})
    return sae, loss_log


def train_sae_best_of(A, device, seeds=range(5), **kwargs):
    """Train one SAE per seed and keep the one with the lowest final training loss (training
    occasionally converges to a solution with a much larger L1 term). Returns the model, its
    loss log, its seed, and the final loss of every seed."""
    best, final_losses = None, {}
    for seed in seeds:
        model, loss_log = train_sae(A, device, seed=seed, **kwargs)
        final_losses[seed] = loss_log[-1]["loss"]
        if best is None or final_losses[seed] < best[1][-1]["loss"]:
            best = (model, loss_log, seed)
    return (*best, final_losses)


@torch.no_grad()
def encode_in_batches(sae, X, device, batch_size=128):
    sae.eval()
    return torch.cat([sae.encode(X[i:i + batch_size].to(device)).cpu() for i in range(0, len(X), batch_size)])


def top_prompts_for_feature(Z, prompts, feature_id, k=10):
    """The k prompts on which a feature is most active."""
    vals = Z[:, feature_id]
    idx = torch.argsort(vals, descending=True)[:k]
    return [prompts[int(i)] for i in idx]


def feature_delta_ranking(Z_pre, Z_post, prompts, top_n=20, k_exemplars=10):
    """Features ranked by |mean activation under M_post - under M_pre|, with their top-activating
    prompts under the checkpoint where the feature is more active (M_post if its activation
    increases, M_pre if it decreases). Also returns per-feature statistics."""
    mean_pre, mean_post = Z_pre.mean(dim=0), Z_post.mean(dim=0)
    stats = pd.DataFrame({
        "feature": np.arange(len(mean_pre)),
        "mean_pre": mean_pre.numpy(),
        "mean_post": mean_post.numpy(),
        "delta": (mean_post - mean_pre).numpy(),
    })
    stats["abs_delta"] = stats["delta"].abs()

    ranked = []
    for _, row in stats.sort_values("abs_delta", ascending=False).head(top_n).iterrows():
        source, Z = ("M_post", Z_post) if row["delta"] > 0 else ("M_pre", Z_pre)
        ranked.append({
            "feature": int(row["feature"]),
            "delta": float(row["delta"]),
            "abs_delta": float(row["abs_delta"]),
            "exemplars_from": source,
            "top_prompts": top_prompts_for_feature(Z, prompts, int(row["feature"]), k=k_exemplars),
        })
    return ranked, stats
