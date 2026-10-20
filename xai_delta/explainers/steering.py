"""Linear probe separating M_pre from M_post transcripts, and activation steering of
M_post along the probe direction."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from xai_delta.explainers.patching import find_blocks


def format_chat(tok, user_text, assistant_text=None):
    """Chat-formatted text. Without `assistant_text`, the generation prompt is appended."""
    msgs = [{"role": "user", "content": user_text}]
    if assistant_text is not None:
        msgs.append({"role": "assistant", "content": assistant_text})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=(assistant_text is None))


@torch.no_grad()
def generate_answers(model, tok, prompts, max_new_tokens=160):
    """Greedy completion for every prompt (prompt text stripped)."""
    model.eval()
    device = next(model.parameters()).device
    outs = []
    for p in prompts:
        enc = tok(format_chat(tok, p), return_tensors="pt").to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                             pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        full = tok.decode(gen[0], skip_special_tokens=True)
        prompt_dec = tok.decode(enc["input_ids"][0], skip_special_tokens=True)
        outs.append(full[len(prompt_dec):].strip() if full.startswith(prompt_dec) else full.strip())
    return outs


@torch.no_grad()
def collect_reps(model, tok, texts, layers=(-3,), position="last", max_length=256, batch_size=8):
    """Hidden states of the given layers at the last (or mean over) non-pad token(s)."""
    model.eval()
    device = next(model.parameters()).device
    chunks = []
    for start in range(0, len(texts), batch_size):
        enc = tok(texts[start:start + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        mask = enc["attention_mask"]

        reps = []
        for l in layers:
            h = hs[l]
            if position == "last":
                reps.append(h[torch.arange(h.size(0), device=device), mask.sum(dim=1) - 1])
            elif position == "mean":
                m = mask.unsqueeze(-1)
                reps.append((h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1))
            else:
                raise ValueError("position must be 'last' or 'mean'")
        chunks.append(torch.cat(reps, dim=-1).float().cpu().numpy().astype(np.float16))
    return np.concatenate(chunks, axis=0)


def train_probe(X, y):
    """L2 logistic regression (C chosen by 5-fold CV) on standardized features. Returns the
    unit-norm direction in the original feature space, oriented towards class 1 (M_post)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000))
    cv = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1.0, 3.0, 10.0]},
                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
                      n_jobs=-1, scoring="accuracy")
    cv.fit(X_train, y_train)
    scaler = cv.best_estimator_.named_steps["standardscaler"]
    lr = cv.best_estimator_.named_steps["logisticregression"]

    # map the classifier back from standardized to raw feature space
    w = lr.coef_[0] / scaler.scale_
    b = lr.intercept_[0] - float(np.dot(scaler.mean_, w))
    scores = X_train @ w + b
    if scores[y_train == 1].mean() < scores[y_train == 0].mean():
        w, b = -w, -b
    norm = np.linalg.norm(w) + 1e-12
    w, b = w / norm, b / norm

    def accuracy(X_, y_):
        return float(((X_ @ w + b >= 0).astype(np.int64) == y_).mean())

    return {
        "w_best": w.astype(np.float32),
        "b_best": float(b),
        "best_params": cv.best_params_,
        "train_acc": accuracy(X_train, y_train),
        "test_acc": accuracy(X_test, y_test),
    }


def fit_steering_probe(tok, M_pre, M_post, probe_prompts, layer_from_end, sort_by_prompt=False):
    """Generate M_pre (class 0) and M_post (class 1) answers to the probe prompts and train the
    probe on M_post's representations of the resulting transcripts. `sort_by_prompt` orders the
    transcripts alphabetically by prompt before the train/test split."""
    aligned = list(zip(probe_prompts, generate_answers(M_pre, tok, probe_prompts, max_new_tokens=160)))
    misaligned = list(zip(probe_prompts, generate_answers(M_post, tok, probe_prompts, max_new_tokens=160)))
    if sort_by_prompt:
        aligned.sort(key=lambda pa: pa[0])
        misaligned.sort(key=lambda pa: pa[0])

    def reps(pairs):
        texts = [format_chat(tok, p, a) for p, a in pairs]
        return collect_reps(M_post, tok, texts, layers=(-layer_from_end,), position="last",
                            batch_size=8, max_length=256)

    X_aligned, X_misaligned = reps(aligned), reps(misaligned)
    X = np.concatenate([X_aligned, X_misaligned], axis=0)
    y = np.array([0] * len(X_aligned) + [1] * len(X_misaligned), dtype=np.int64)
    return train_probe(X, y)


class ActivationSteerer:
    """Forward hook on one block that subtracts alpha * w from the last-token hidden state."""

    def __init__(self, model, w_dir, layer_index):
        self.model = model
        param = next(model.parameters())
        w = torch.as_tensor(w_dir, device=param.device, dtype=param.dtype)
        self.w = w / (w.norm() + 1e-12)
        self.alpha = 0.0
        self.layer_index = int(layer_index)
        self.hook = None
        self.hook_name = f"layers[{self.layer_index}]"

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    def _hook_fn(self, module, inputs, output):
        h, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        h2 = h.clone()
        h2[:, -1, :] = h2[:, -1, :] - self.alpha * self.w.to(dtype=h.dtype)
        return h2 if rest is None else (h2,) + rest

    def __enter__(self):
        self.hook = find_blocks(self.model)[self.layer_index].register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.hook.remove()
        return False


@torch.no_grad()
def generate_text(model, tok, prompt_text, max_new_tokens=160):
    """Greedy continuation of an already chat-formatted prompt (special tokens kept)."""
    enc = tok(prompt_text, return_tensors="pt").to(next(model.parameters()).device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=False)


@torch.no_grad()
def next_token_logits(model, tok, prompt_text):
    enc = tok(prompt_text, return_tensors="pt").to(next(model.parameters()).device)
    return model(**enc, use_cache=False).logits[:, -1, :].detach()


def run_steering_experiment(model, tok, prompts, w_dir, alphas=(10, 15, 20), layer_from_end=3,
                            max_new_tokens=200):
    """Unsteered and steered generations of `model` for every prompt and alpha, with the
    largest next-token logit change as a coherence diagnostic."""
    layer_index = len(find_blocks(model)) - layer_from_end
    rows = []
    with ActivationSteerer(model, w_dir=w_dir, layer_index=layer_index) as steerer:
        for p in prompts:
            prompt_text = format_chat(tok, p["prompt"])

            steerer.set_alpha(0.0)
            rows.append({
                "prompt_id": p["id"], "prompt": p["prompt"], "alpha": 0.0, "condition": "bad",
                "text": generate_text(model, tok, prompt_text, max_new_tokens), "hooked": steerer.hook_name,
            })

            for a in alphas:
                steerer.set_alpha(0.0)
                logits0 = next_token_logits(model, tok, prompt_text)
                steerer.set_alpha(float(a))
                logits1 = next_token_logits(model, tok, prompt_text)
                rows.append({
                    "prompt_id": p["id"], "prompt": p["prompt"], "alpha": float(a), "condition": "steered",
                    "hooked": steerer.hook_name,
                    "max_abs_dlogit": float((logits1 - logits0).abs().max().item()),
                    "argmax_changed": bool((logits1.argmax(dim=-1) != logits0.argmax(dim=-1)).item()),
                    "text": generate_text(model, tok, prompt_text, max_new_tokens),
                })
    return rows
