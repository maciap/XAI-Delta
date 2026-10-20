"""Fine-tuning data attribution for the example prompt of the paper's illustrative scenario
(Qwen2.5-0.5B, prompt "chest_pain"): M_post's answer is linked to the fine-tuning data (1) by
retrieving the fine-tuning pairs most similar to the full prompt-response pair, and (2) by
Grad-Cosine over the LoRA parameters, computed for every fine-tuning example.

Writes data/outputs/<model>/data_attribution_example.json.

Usage:
    python scripts/run_data_attribution_example.py
"""
import argparse

import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from xai_delta import config, models
from xai_delta.explainers import data_attribution as da
from xai_delta.io import log, save_json
from xai_delta.prompts import load_prompts

TOP_K = 10


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5-0.5b", choices=config.MODELS)
    parser.add_argument("--example", default="chest_pain")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not config.FINETUNE_DATA.exists():
        raise FileNotFoundError(f"{config.FINETUNE_DATA} not found: run scripts/download_finetune_data.sh")
    device = torch.device(args.device)
    cfg = config.load_model_config(args.model)
    target_prompt = next(p for p in load_prompts("PROMPTS_150.json") if p["id"] == args.example)["prompt"]
    candidates = da.load_finetune_examples(config.FINETUNE_DATA)

    tok = models.load_tokenizer(cfg["base_id"])
    base = models.load_base(cfg["base_id"], device, dtype=torch.float16)
    model = PeftModel.from_pretrained(base, cfg["adapter_id"], is_trainable=True).eval()

    msgs = [{"role": "user", "content": target_prompt}]
    input_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    with torch.no_grad():
        out_ids = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=200,
                                 do_sample=False, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    post_answer = tok.decode(out_ids[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
    log(f"M_post answer: {post_answer[:200]}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    order, sims = da.retrieve_pairs(da.pair_text(target_prompt, post_answer), candidates, embedder)
    del embedder
    retrieved = [{
        "candidate_index": int(i),
        "pair_similarity": float(sims[i]),
        "user_text": da.message_text(candidates[i], "user"),
        "assistant_text": da.message_text(candidates[i], "assistant"),
    } for i in order[:TOP_K]]

    da.enable_lora_grad(model)
    target_grad, _ = da.grad_vector(model, da.encode_sft_example(msgs + [{"role": "assistant", "content": post_answer}],
                                                                 tok, device))
    scores = []
    for idx, cand in enumerate(tqdm(candidates)):
        g, loss = da.grad_vector(model, da.encode_sft_example(cand["messages"], tok, device))
        scores.append((da.cosine(target_grad, g), loss, idx))
    scores.sort(reverse=True)
    proponents = [{
        "candidate_index": idx,
        "grad_cosine": cos,
        "sft_loss": loss,
        "pair_similarity": float(sims[idx]),
        "user_text": da.message_text(candidates[idx], "user"),
        "assistant_text": da.message_text(candidates[idx], "assistant"),
    } for cos, loss, idx in scores[:TOP_K]]

    save_json(config.output_dir(args.model) / "data_attribution_example.json", {
        "example_id": args.example,
        "target_prompt": target_prompt,
        "target_post_answer": post_answer,
        "n_finetune_examples": len(candidates),
        "top_retrieved_pairs": retrieved,
        "top_proponents": proponents,
    })


if __name__ == "__main__":
    main()
