"""Phase 3 gate: load a registered policy and time chunk inference on this laptop.

    uv run python scripts/bench_policy.py                         # act_cube_sim_chaenn on CPU
    uv run python scripts/bench_policy.py --threads 4 --runs 30   # with a capped thread count
    uv run python scripts/bench_policy.py --xpu                   # also try the Intel Arc GPU if torch exposes it

Feeds random frames with the checkpoint's camera shapes and prints load time and p50/p95 latency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from so101_mcp.policies import REGISTRY, LoadedPolicy


def bench(name: str, device: str, runs: int) -> dict:
    loaded = LoadedPolicy.load(REGISTRY[name], device=device)
    rng = np.random.default_rng(0)
    frames = {k: rng.integers(0, 255, (s[1], s[2], 3), dtype=np.uint8) for k, s in loaded.image_shapes.items()}
    state = np.array([0.0, -60.0, 60.0, 35.0, 0.0, 30.0])
    chunk = loaded.predict_chunk(state, frames)  # warm-up
    loaded.latencies_ms.clear()
    for _ in range(runs):
        loaded.predict_chunk(state, frames)
    summary = loaded.latency_summary()
    params = sum(p.numel() for p in loaded.policy.parameters()) / 1e6
    return {
        "policy": name, "device": device, "load_s": round(loaded.load_seconds, 1), "params_m": round(params, 1),
        "chunk": list(chunk.shape), "chunk_first": np.round(chunk[0], 1).tolist(), **summary,
        "chunk_seconds_at_30hz": round(chunk.shape[0] / 30, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="act_cube_sim_chaenn", choices=list(REGISTRY))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--xpu", action="store_true")
    args = parser.parse_args()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    results = [bench(args.policy, "cpu", args.runs) | {"threads": torch.get_num_threads()}]
    if args.xpu:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            results.append(bench(args.policy, "xpu", args.runs))
        else:
            results.append({"device": "xpu", "available": False})
    for r in results:
        print(json.dumps(r))
    out = Path("artifacts/phase3")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"bench_{args.policy}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
