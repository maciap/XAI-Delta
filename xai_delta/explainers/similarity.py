"""Layer-wise representational similarity between M_pre and M_post: linear CKA,
whitened CKA, relative Frobenius distance, and per-layer linear-probe separability."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def center_gram(G):
    """Double-center a Gram matrix: H G H with H = I - 11^T / n."""
    n = G.size(0)
    one = torch.ones((n, n), device=G.device, dtype=G.dtype) / n
    return G - one @ G - G @ one + one @ G @ one


def linear_cka(X, Y, eps=1e-12):
    """Linear CKA between token representations X, Y of shape (n_tokens, d)."""
    Kc = center_gram(X.float() @ X.float().t())
    Lc = center_gram(Y.float() @ Y.float().t())
    denom = torch.sqrt((Kc * Kc).sum()) * torch.sqrt((Lc * Lc).sum())
    if denom < eps:
        return float("nan")
    return ((Kc * Lc).sum() / denom).item()


def whitened_linear_cka(X, Y, eps=1e-12):
    """Linear CKA after z-scoring every feature dimension of X and Y separately,
    so that a few high-magnitude dimensions cannot dominate the comparison."""
    def zscore(T):
        T = T.float()
        return (T - T.mean(dim=0, keepdim=True)) / T.std(dim=0, keepdim=True).clamp_min(eps)

    return linear_cka(zscore(X), zscore(Y), eps=eps)


def relative_frobenius_distance(X, Y, eps=1e-12):
    """||X - Y||_F / ||X||_F for paired representations of the same prompt."""
    X, Y = X.float(), Y.float()
    return (torch.linalg.norm(X - Y) / torch.linalg.norm(X).clamp_min(eps)).item()


@torch.no_grad()
def get_hidden_states(model, tok, prompt):
    """Hidden states of every layer (embeddings + each block) for one prompt."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    return model(**inputs, output_hidden_states=True, use_cache=False).hidden_states


def _clip(v):
    return v if np.isnan(v) else float(np.clip(v, 0.0, 1.0))


def similarity_curves_for_prompt(model_pre, model_post, tok, prompt):
    """Per-layer CKA, whitened CKA and relative Frobenius distance for one prompt."""
    hs_pre = get_hidden_states(model_pre, tok, prompt)
    hs_post = get_hidden_states(model_post, tok, prompt)
    assert len(hs_pre) == len(hs_post)

    curves = {"cka": [], "whitened_cka": [], "rel_frobenius_dist": []}
    for X, Y in zip(hs_pre, hs_post):
        X, Y = X.squeeze(0), Y.squeeze(0)
        curves["cka"].append(_clip(linear_cka(X, Y)))
        curves["whitened_cka"].append(_clip(whitened_linear_cka(X, Y)))
        curves["rel_frobenius_dist"].append(relative_frobenius_distance(X, Y))
    return {k: np.array(v, dtype=np.float64) for k, v in curves.items()}


def compute_similarity_curves(model_pre, model_post, tok, prompts):
    """Mean and standard error of every similarity curve over a prompt set."""
    per_metric = {"cka": [], "whitened_cka": [], "rel_frobenius_dist": []}
    for item in prompts:
        p = item["prompt"] if isinstance(item, dict) else item
        for k, v in similarity_curves_for_prompt(model_pre, model_post, tok, p).items():
            per_metric[k].append(v)

    out = {}
    for k, curves in per_metric.items():
        stacked = np.stack(curves, axis=0)
        out[k] = {
            "mean": np.nanmean(stacked, axis=0),
            "sem": np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(stacked.shape[0]),
        }
    return out


@torch.no_grad()
def collect_last_token_all_layers(model, tok, prompts):
    """Last-token hidden state at every layer: array (n_prompts, n_layers, d)."""
    reps = []
    for item in prompts:
        p = item["prompt"] if isinstance(item, dict) else item
        hs = get_hidden_states(model, tok, p)
        reps.append(np.stack([h.squeeze(0)[-1].float().cpu().numpy() for h in hs], axis=0))
    return np.stack(reps, axis=0)


def quick_probe_accuracy(X, y, test_size=0.2, seed=0):
    """(test accuracy, train accuracy) of a single logistic-regression fit."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, C=1.0).fit(X_train, y_train)
    return float(clf.score(X_test, y_test)), float(clf.score(X_train, y_train))


def layerwise_probe_profile(reps_pre, reps_post):
    """Per-layer accuracy of a probe separating M_pre from M_post representations."""
    test_accs, train_accs = [], []
    for l in range(reps_pre.shape[1]):
        X = np.concatenate([reps_pre[:, l, :], reps_post[:, l, :]], axis=0)
        y = np.array([0] * reps_pre.shape[0] + [1] * reps_post.shape[0])
        test_acc, train_acc = quick_probe_accuracy(X, y)
        test_accs.append(test_acc)
        train_accs.append(train_acc)
    return np.array(test_accs), np.array(train_accs)


def find_extremum_layer(profile, mode="argmin"):
    """Index of the extremum of a per-layer profile, skipping the embedding layer (index 0)."""
    body = profile[1:]
    return int(np.argmin(body) if mode == "argmin" else np.argmax(body)) + 1
