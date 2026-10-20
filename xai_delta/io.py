"""JSON / JSONL helpers and timestamped logging."""
import json
import time

import numpy as np
import torch


def _to_builtin(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.ndarray, torch.Tensor)):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)
    log(f"wrote {path}")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_to_builtin) + "\n")
    log(f"wrote {path}")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)
