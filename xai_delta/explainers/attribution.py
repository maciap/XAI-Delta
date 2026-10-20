"""Delta feature attribution: Integrated Gradients of M_pre and M_post on the same input
and the same target (M_pre's own completion), and their difference per input token."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients

_MOJIBAKE = {
    "IâĢĻm": "I’m", "Iâ€™m": "I’m",
    "donâĢĻt": "don’t", "donâ€™t": "don’t",
    "itâĢĻs": "it’s", "itâ€™s": "it’s",
    "youâĢĻre": "you’re", "youâ€™re": "you’re",
    "weâĢĻre": "we’re", "weâ€™re": "we’re",
    "âĢĻ": "’", "â€™": "’",
    "â€œ": "“", "â€�": "”",
    "â€”": "—", "â€“": "–", "â€¦": "…",
}


def fix_mojibake(s):
    """Repair common mis-decoded punctuation sequences."""
    if s is None:
        return ""
    for k, v in _MOJIBAKE.items():
        s = s.replace(k, v)
    return s


def tok_to_printable(tok):
    """Readable form of a BPE token (newline marker kept as a line break)."""
    return "\n" if tok == "Ċ" else fix_mojibake(tok.replace("Ġ", " "))


def make_chat_prompt(tok, user_text, system_prompt):
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def find_subsequence(haystack, needle):
    """(start, end) of the first occurrence of `needle` in `haystack`, or None."""
    for i in range(len(haystack) - len(needle) + 1):
        if needle and haystack[i:i + len(needle)] == needle:
            return i, i + len(needle)
    return None


def make_forward_fn(model, input_ids, target_ids, attention_mask=None):
    """Scalar objective of the input embeddings: summed log-probability of the target tokens."""
    model.eval()
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    target_ids = target_ids.to(dev)
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)
    n_in, n_tgt = input_ids.shape[1], target_ids.shape[1]
    tgt_emb = model.get_input_embeddings()(target_ids).detach().to(dtype=dtype)

    def forward(inputs_embeds):
        B = inputs_embeds.shape[0]
        full = torch.cat([inputs_embeds.to(dtype=dtype), tgt_emb.expand(B, -1, -1)], dim=1)
        if attention_mask is None:
            attn = torch.ones((B, n_in + n_tgt), device=dev, dtype=torch.long)
        else:
            attn_in = attention_mask.expand(B, -1)
            attn = torch.cat([attn_in, torch.ones((B, n_tgt), device=dev, dtype=attn_in.dtype)], dim=1)
        logp = F.log_softmax(model(inputs_embeds=full, attention_mask=attn, use_cache=False).logits, dim=-1)
        pred_pos = torch.arange(n_tgt, device=dev) + (n_in - 1)
        return logp[:, pred_pos, :].gather(-1, target_ids.expand(B, -1).unsqueeze(-1)).squeeze(-1).sum(dim=1)

    return forward


def ig_input_tokens(model, tok, prompt_text, target_text, steps=24, internal_batch_size=2):
    """Integrated Gradients (zero-embedding baseline) per input token, normalized to sum to 1.
    Returns tokens, scores and the objective value (log-probability of the target)."""
    model.eval()
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    enc_in = tok(prompt_text, return_tensors="pt")
    target_ids = tok(target_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
    input_ids = enc_in["input_ids"].to(dev)
    attn_mask = enc_in.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(dev)

    inp_emb = model.get_input_embeddings()(input_ids).detach().to(dtype=dtype)
    forward_fn = make_forward_fn(model, input_ids, target_ids, attention_mask=attn_mask)
    attributions = IntegratedGradients(forward_fn).attribute(
        inputs=inp_emb, baselines=torch.zeros_like(inp_emb), n_steps=steps,
        internal_batch_size=internal_batch_size,
    )

    scores = attributions[0].abs().sum(dim=-1)
    denom = scores.sum()
    scores = torch.zeros_like(scores) if not torch.isfinite(denom) or denom.item() == 0.0 else scores / denom
    tokens = tok.convert_ids_to_tokens(input_ids[0].tolist())
    return tokens, scores.detach().float().cpu(), float(forward_fn(inp_emb)[0].item())


@torch.no_grad()
def generate_reference(model, tok, prompt_text, max_new_tokens=64):
    """Greedy completion of an already chat-formatted prompt."""
    enc = tok(prompt_text, return_tensors="pt").to(next(model.parameters()).device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                         pad_token_id=tok.pad_token_id)
    return fix_mojibake(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())


def delta_attribution(model_pre, model_post, tok, example, system_prompt, max_new_tokens=64, steps=24,
                      internal_batch_size=2):
    """IG(M_post) - IG(M_pre) over the user-prompt tokens, both scored on M_pre's reference
    completion, together with the log-probability each model assigns to that completion."""
    user_q = fix_mojibake(example["prompt"])
    prompt_text = make_chat_prompt(tok, user_q, system_prompt)
    ref = generate_reference(model_pre, tok, prompt_text, max_new_tokens=max_new_tokens)

    t_pre, s_pre, logp_pre = ig_input_tokens(model_pre, tok, prompt_text, ref, steps, internal_batch_size)
    _, s_post, logp_post = ig_input_tokens(model_post, tok, prompt_text, ref, steps, internal_batch_size)

    full_ids = tok(prompt_text, return_tensors="pt")["input_ids"][0].tolist()
    span = find_subsequence(full_ids, tok(user_q, add_special_tokens=False)["input_ids"])
    a, b = span if span is not None else (0, len(t_pre))
    return {
        "example_id": example["id"],
        "reference_completion": ref,
        "tokens_user": [str(t) for t in t_pre[a:b]],
        "delta_user": (s_post[a:b] - s_pre[a:b]).tolist(),
        "logp_base": logp_pre,
        "logp_bad": logp_post,
        "delta_logp": logp_post - logp_pre,
    }


def show_delta_boxes(tokens, delta_scores, title="Δ attribution (M_post − M_pre)", cmap_name="RdBu_r",
                     vlim=None, font_size=14, line_spacing=1.45, pad=0.22,
                     colorbar_label="Δ attribution (M_post − M_pre)", figsize=(13.5, 6.8), dpi=200):
    """Render tokens as boxes colored by their delta attribution, with a colorbar."""
    toks = [tok_to_printable(t) for t in tokens]
    d = np.nan_to_num(np.asarray(delta_scores, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    m = float(np.max(np.abs(d))) + 1e-12
    vmin, vmax = (-m, m) if vlim is None else (float(vlim[0]), float(vlim[1]))
    cmap = mpl.colormaps[cmap_name]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0.03, 0.22, 0.94, 0.68])
    ax.axis("off")
    ax.text(0.0, 1.04, fix_mojibake(title), transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")
    cax = fig.add_axes([0.18, 0.10, 0.64, 0.035])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.set_label(colorbar_label, fontsize=14)
    cb.ax.tick_params(labelsize=11)

    box = dict(boxstyle=f"round,pad={pad}", edgecolor=(0, 0, 0, 0.15), linewidth=0.7)
    with mpl.rc_context({"text.usetex": False}):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_width = ax.get_window_extent(renderer=renderer).width
        line_step = (font_size / 72.0) / fig.get_size_inches()[1] * line_spacing
        x, y = 0.0, 1.0

        def place(text, color):
            artist = ax.text(x, y, text, transform=ax.transAxes, fontsize=font_size, va="top",
                             bbox={**box, "facecolor": color})
            fig.canvas.draw()
            return artist, artist.get_window_extent(renderer=renderer).width / ax_width

        for tok, val in zip(toks, d):
            if y < 0.0:
                break
            if tok == "\n":
                x, y = 0.0, y - line_step
                continue
            color = cmap(norm(val))
            artist, w = place(tok.replace(" ", " "), color)
            if x + w > 0.99 and x > 0.0:
                artist.remove()
                x, y = 0.0, y - line_step
                if y < 0.0:
                    break
                artist, w = place(tok.replace(" ", " "), color)
            x += w + 0.005
    return fig, ax
