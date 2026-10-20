"""Fine-tuning data attribution with Grad-Cosine: cosine similarity between the LoRA-parameter
gradient of a target example and that of each fine-tuning example. To keep it tractable, only
the K fine-tuning examples semantically closest to the target prompt are scored."""
import json

import numpy as np
import torch


def load_finetune_examples(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def message_text(example, role):
    """Content of the first message with the given role."""
    return next((m["content"] for m in example["messages"] if m["role"] == role), "")


def retrieve_top_k(target_text, candidates, embedder, k=100, batch_size=256):
    """Indices of the k candidates whose user turn is closest to `target_text` in embedding
    space, and the similarity of every candidate."""
    cand_embs = embedder.encode([message_text(c, "user") for c in candidates], batch_size=batch_size,
                                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    target_emb = embedder.encode([target_text], normalize_embeddings=True, convert_to_numpy=True)[0]
    sims = cand_embs @ target_emb
    return np.argsort(-sims)[:k].tolist(), sims


def pair_text(user_text, assistant_text):
    return f"User: {user_text}\nAssistant: {assistant_text}"


def retrieve_pairs(target_pair, candidates, embedder, batch_size=256):
    """Cosine similarity between a target prompt-response pair and every fine-tuning pair, and the
    candidate indices sorted by decreasing similarity."""
    texts = [pair_text(message_text(c, "user"), message_text(c, "assistant")) for c in candidates]
    cand_embs = embedder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
    target_emb = embedder.encode([target_pair], normalize_embeddings=True, convert_to_numpy=True)[0]
    sims = cand_embs @ target_emb
    return np.argsort(-sims).tolist(), sims


def encode_sft_example(messages, tok, device):
    """Tokenized conversation with the loss restricted to the final assistant turn."""
    assert messages[-1]["role"] == "assistant"
    full_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                       return_tensors="pt").input_ids
    prompt_ids = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True,
                                         return_tensors="pt").input_ids
    input_ids = full_ids.to(device)
    labels = input_ids.clone()
    labels[:, :prompt_ids.shape[-1]] = -100
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}


def enable_lora_grad(model):
    """Freeze every parameter except the LoRA weights."""
    for name, p in model.named_parameters():
        p.requires_grad_("lora" in name.lower())


def grad_vector(model, batch):
    """Flattened gradient of the SFT loss w.r.t. the trainable (LoRA) parameters, and the loss."""
    model.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    grads = [
        torch.zeros(p.numel(), dtype=torch.float32) if p.grad is None else p.grad.detach().float().reshape(-1).cpu()
        for _, p in model.named_parameters() if p.requires_grad
    ]
    model.zero_grad(set_to_none=True)
    return torch.cat(grads), float(loss.detach().cpu())


def cosine(a, b, eps=1e-12):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + eps))
