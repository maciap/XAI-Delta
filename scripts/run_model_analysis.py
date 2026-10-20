"""Comparative analysis of one (M_pre, M_post) pair: similarity, delta attribution,
activation patching, steering, SAE concepts and fine-tuning data attribution.

All outputs are written to data/outputs/<model>/.

Usage:
    python scripts/run_model_analysis.py --model llama3.1-8b
    python scripts/run_model_analysis.py --model qwen2.5-14b --phases patching,steering
"""
import argparse
import gc

import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer

from xai_delta import config, models
from xai_delta.behavior.urgency_metric import behavior_score
from xai_delta.explainers import attribution, data_attribution, patching, sae, similarity, steering
from xai_delta.io import log, save_json, save_jsonl
from xai_delta.prompts import load_probe_prompts, load_prompts

PHASES = ["similarity", "attribution", "patching", "patching_reverse", "patching_offtarget", "steering", "sae",
          "data_attribution"]

ATTRIBUTION_EXAMPLE = "mild_headache"
ATTRIBUTION_SYSTEM_PROMPT = "You are a medical assistant. Respond concisely and safely."
OFFTARGET_BLOCK = 3
STEERING_ALPHAS = (10, 15, 20)
STEERING_EXAMPLE = "chest_pain"
STEERING_EXAMPLE_ALPHAS = (2, 5, 8, 10, 12, 15, 18, 20, 25, 30, 40)
MAX_NEW_TOKENS = 200


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def phase_similarity(tok, M_pre, M_post, prompts, out):
    """Layer-wise similarity curves on the 150 evaluation prompts."""
    curves = similarity.compute_similarity_curves(M_pre, M_post, tok, prompts)
    reps_pre = similarity.collect_last_token_all_layers(M_pre, tok, prompts)
    reps_post = similarity.collect_last_token_all_layers(M_post, tok, prompts)
    probe_test, probe_train = similarity.layerwise_probe_profile(reps_pre, reps_post)

    result = {"n_prompts": len(prompts), "n_blocks": len(curves["cka"]["mean"]) - 1}
    for k, v in curves.items():
        result[f"{k}_mean"], result[f"{k}_sem"] = v["mean"], v["sem"]
    result.update({
        "probe_test_acc_by_layer": probe_test,
        "probe_train_acc_by_layer": probe_train,
        "ell_star_cka_argmin": similarity.find_extremum_layer(curves["cka"]["mean"], "argmin"),
        "ell_star_whitened_cka_argmin": similarity.find_extremum_layer(curves["whitened_cka"]["mean"], "argmin"),
        "ell_star_rel_frobenius_argmax": similarity.find_extremum_layer(curves["rel_frobenius_dist"]["mean"], "argmax"),
        "ell_star_probe_argmax": similarity.find_extremum_layer(probe_test, "argmax"),
    })
    save_json(out / "similarity.json", result)


def phase_attribution(tok, M_pre, M_post, prompts, out):
    """Integrated-Gradients delta attribution on one example prompt."""
    example = next(p for p in prompts if p["id"] == ATTRIBUTION_EXAMPLE)
    res = attribution.delta_attribution(M_pre, M_post, tok, example, ATTRIBUTION_SYSTEM_PROMPT,
                                        max_new_tokens=64, steps=24, internal_batch_size=2)
    res["ig_config"] = {"steps": 24, "internal_batch_size": 2, "max_new_tokens": 64, "dtype": "bfloat16"}
    save_json(out / "attribution.json", res)


def phase_patching(tok, M_pre, M_post, prompts, layer_from_end, device, out, name="patching"):
    """Patch M_pre's activation into M_post at the given layer, on the 150 prompts."""
    records = patching.run_patching_conditions(prompts, source=M_pre, target=M_post, tok=tok,
                                               layer_from_end=layer_from_end, max_new_tokens=MAX_NEW_TOKENS)
    save_jsonl(out / f"{name}_generations.jsonl", records)

    embedder = patching.build_embedder(device)
    df_sim = patching.evaluate_patching_similarity(records, embedder)
    save_json(out / f"{name}.json", {
        "layer_from_end": layer_from_end,
        "n_prompts": len(records),
        "mean_sim_base_bad": float(df_sim["sim_base_bad"].mean()),
        "mean_sim_base_patched": float(df_sim["sim_base_patched"].mean()),
        "urgency_B_base": float(np.mean([behavior_score(r["base"]) for r in records])),
        "urgency_B_bad": float(np.mean([behavior_score(r["bad"]) for r in records])),
        "urgency_B_patched": float(np.mean([behavior_score(r["patched"]) for r in records])),
        "per_prompt": df_sim.to_dict(orient="records"),
    })


def phase_patching_reverse(tok, M_pre, M_post, prompts, layer_from_end, device, out):
    """Reverse direction: patch M_post's activation into M_pre, and measure the similarity of the
    reverse-patched M_pre outputs to M_post's own outputs."""
    records = patching.run_patching_conditions(prompts, source=M_post, target=M_pre, tok=tok,
                                               layer_from_end=layer_from_end, max_new_tokens=MAX_NEW_TOKENS)
    embedder = patching.build_embedder(device)
    sims = [patching.cosine_sim(embedder, r["base"], r["patched"]) for r in records]
    save_json(out / "patching_reverse.json", {
        "direction": "M_post_activation_into_M_pre",
        "layer_from_end": layer_from_end,
        "n_prompts": len(records),
        "mean_sim_pre_to_reverse_patched": float(np.mean(sims)),
    })


def phase_steering(tok, M_pre, M_post, prompts, probe_prompts, layer_from_end, out):
    """Probe on M_post's representations of M_pre vs. M_post transcripts, then steer M_post
    along the probe direction: alpha sweep on the 150 prompts and a wide sweep on one example.
    (For Qwen2.5-0.5B the example shown in the paper is produced by scripts/run_steering_example_qwen0.5b.py.)"""
    probe = steering.fit_steering_probe(tok, M_pre, M_post, probe_prompts, layer_from_end)
    log(f"probe: C={probe['best_params']} train_acc={probe['train_acc']:.4f} test_acc={probe['test_acc']:.4f}")

    rows = steering.run_steering_experiment(M_post, tok, prompts, w_dir=probe["w_best"], alphas=STEERING_ALPHAS,
                                            layer_from_end=layer_from_end, max_new_tokens=MAX_NEW_TOKENS)
    save_jsonl(out / "steering_sweep.jsonl", rows)
    save_json(out / "steering.json", {
        "probe_best_params": probe["best_params"],
        "probe_train_acc": probe["train_acc"],
        "probe_test_acc": probe["test_acc"],
        "layer_from_end": layer_from_end,
        "alphas": list(STEERING_ALPHAS),
        "n_prompts": len(prompts),
        "n_generations": len(rows),
    })

    example = [p for p in prompts if p["id"] == STEERING_EXAMPLE]
    rows = steering.run_steering_experiment(M_post, tok, example, w_dir=probe["w_best"],
                                            alphas=STEERING_EXAMPLE_ALPHAS, layer_from_end=layer_from_end,
                                            max_new_tokens=MAX_NEW_TOKENS)
    save_jsonl(out / "steering_example.jsonl", rows)


def phase_sae(tok, M_pre, M_post, probe_prompts, layer_from_end, device, out):
    """SAE on pooled M_pre / M_post activations at the analysis layer (best of 5 seeds by final
    training loss); features ranked by their mean activation shift. Concept labels are assigned
    manually afterwards."""
    layer_idx = len(patching.find_blocks(M_post)) - layer_from_end
    A_pre = sae.collect_layer_acts(M_pre, tok, probe_prompts, layer_idx)
    A_post = sae.collect_layer_acts(M_post, tok, probe_prompts, layer_idx)
    A_pre_n, A_post_n = sae.normalize_jointly(A_pre, A_post)

    model, loss_log, seed, seed_losses = sae.train_sae_best_of(torch.cat([A_pre_n, A_post_n], dim=0), device)
    log(f"SAE final loss per seed: {seed_losses}; kept seed {seed}")
    Z_pre = sae.encode_in_batches(model, A_pre_n, device)
    Z_post = sae.encode_in_batches(model, A_post_n, device)
    active_fraction = float((torch.cat([Z_pre, Z_post], dim=0) > 0).float().mean())
    shift_ranked, stats = sae.feature_delta_ranking(Z_pre, Z_post, probe_prompts, top_n=20, k_exemplars=10)
    for entry in shift_ranked:
        entry["label"] = None

    post_alone = [
        {"feature": int(row["feature"]), "mean_post": float(row["mean_post"]),
         "top_prompts": sae.top_prompts_for_feature(Z_post, probe_prompts, int(row["feature"]), k=10)}
        for _, row in stats.sort_values("mean_post", ascending=False).head(20).iterrows()
    ]
    d_in = A_pre.shape[1]
    save_json(out / "sae.json", {
        "layer_idx": layer_idx, "layer_from_end": layer_from_end, "d_in": d_in, "d_hidden": 8 * d_in,
        "epochs": len(loss_log), "seed": seed, "final_loss_per_seed": seed_losses,
        "final_train_loss": loss_log[-1], "reconstruction_mse": loss_log[-1]["recon"],
        "active_fraction": active_fraction, "shift_ranked_top": shift_ranked, "post_alone_ranked_top": post_alone,
    })


def phase_data_attribution(cfg, prompts, device, out, k=100):
    """Grad-Cosine attribution of M_post's completion of the example prompt to the
    fine-tuning examples (restricted to the k semantically closest ones)."""
    if not config.FINETUNE_DATA.exists():
        raise FileNotFoundError(f"{config.FINETUNE_DATA} not found: run scripts/download_finetune_data.sh")
    target_prompt = next(p for p in prompts if p["id"] == ATTRIBUTION_EXAMPLE)["prompt"]
    candidates = data_attribution.load_finetune_examples(config.FINETUNE_DATA)

    tok = models.load_tokenizer(cfg["base_id"])
    model = PeftModel.from_pretrained(models.load_base(cfg["base_id"], device), cfg["adapter_id"],
                                      is_trainable=True).eval()
    data_attribution.enable_lora_grad(model)

    msgs = [{"role": "user", "content": target_prompt}]
    enc = tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**enc, max_new_tokens=150, do_sample=False, pad_token_id=tok.eos_token_id)
    post_answer = tok.decode(out_ids[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    top_idx, sims = data_attribution.retrieve_top_k(target_prompt, candidates, embedder, k=k)
    del embedder
    free_memory()

    target = data_attribution.encode_sft_example(msgs + [{"role": "assistant", "content": post_answer}], tok, device)
    target_grad, _ = data_attribution.grad_vector(model, target)

    results = []
    for rank, idx in enumerate(top_idx):
        messages = candidates[idx]["messages"]
        if not messages or messages[-1]["role"] != "assistant":
            continue
        try:
            g, loss = data_attribution.grad_vector(model, data_attribution.encode_sft_example(messages, tok, device))
        except Exception as e:  # noqa: BLE001
            log(f"skipping candidate {idx}: {e}")
            continue
        results.append({
            "candidate_index": int(idx),
            "retrieval_rank": rank,
            "semantic_sim": float(sims[idx]),
            "grad_cosine": data_attribution.cosine(target_grad, g),
            "sft_loss": loss,
            "user_text": data_attribution.message_text(candidates[idx], "user"),
            "assistant_text": data_attribution.message_text(candidates[idx], "assistant"),
        })
    results.sort(key=lambda r: -r["grad_cosine"])

    save_json(out / "data_attribution.json", {
        "example_id": ATTRIBUTION_EXAMPLE,
        "target_prompt": target_prompt,
        "target_post_answer": post_answer,
        "k_retrieved": k,
        "n_finetune_examples": len(candidates),
        "top_proponents": results[:10],
    })
    del model
    free_memory()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=config.MODELS)
    parser.add_argument("--phases", default=",".join(PHASES), help=f"comma-separated subset of {PHASES}")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    phases = args.phases.split(",")
    device = torch.device(args.device)
    cfg = config.load_model_config(args.model)
    out = config.output_dir(args.model)
    layer = cfg["layer_from_end"]

    prompts = load_prompts("PROMPTS_150.json")
    probe_prompts = load_probe_prompts()
    log(f"{cfg['name']}: phases={phases}, analysis layer L-{layer}")

    if any(p in phases for p in PHASES[:-1]):
        tok, M_pre, M_post = models.load_models(cfg, device)
        if "similarity" in phases:
            phase_similarity(tok, M_pre, M_post, prompts, out)
        if "attribution" in phases:
            phase_attribution(tok, M_pre, M_post, prompts, out)
        if "patching" in phases:
            phase_patching(tok, M_pre, M_post, prompts, layer, device, out)
        if "patching_reverse" in phases:
            phase_patching_reverse(tok, M_pre, M_post, prompts, layer, device, out)
        if "patching_offtarget" in phases:
            # off-target control: output of block OFFTARGET_BLOCK counted from the input
            offtarget = len(patching.find_blocks(M_post)) - OFFTARGET_BLOCK + 1
            phase_patching(tok, M_pre, M_post, prompts, offtarget, device, out, name="patching_offtarget")
        if "steering" in phases:
            phase_steering(tok, M_pre, M_post, prompts, probe_prompts, layer, out)
        if "sae" in phases:
            phase_sae(tok, M_pre, M_post, probe_prompts, layer, device, out)
        del M_pre, M_post
        free_memory()

    if "data_attribution" in phases:
        phase_data_attribution(cfg, prompts, device, out)
    log("done")


if __name__ == "__main__":
    main()
