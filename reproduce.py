"""Run a config over several seeds and report the mean and standard deviation of diversity."""

import argparse
import json
from pathlib import Path

import numpy as np

from ngd import load_config, run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="YAML config.")
    parser.add_argument("--runs", type=int, default=5, help="Number of seeds.")
    parser.add_argument("--start-seed", type=int, default=1, help="Seeds are start-seed, start-seed + 1, ...")
    parser.add_argument("--output-dir", default="results/reproduce", help="Where to write the JSON summary.")
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available).")
    args = parser.parse_args()

    overrides = {"output_dir": None}
    if args.device is not None:
        overrides["device"] = args.device
    seeds = list(range(args.start_seed, args.start_seed + args.runs))
    scores: dict[str, list[float]] = {}
    for seed in seeds:
        print(f"=== seed {seed} ===")
        results = run_pipeline(load_config(args.config, seed=seed, **overrides))
        for metric, result in results.items():
            scores.setdefault(metric, []).append(result.diversity)

    summary = {
        metric: {"mean": float(np.mean(v)), "std": float(np.std(v)), "runs": v}
        for metric, v in scores.items()
    }
    config = Path(args.config)
    out = Path(args.output_dir) / f"{config.parent.name}_{config.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": str(config), "seeds": seeds, "diversity": summary}, f, indent=2)

    for metric, s in summary.items():
        print(f"{metric}: {s['mean']:.4f} +/- {s['std']:.4f} over {len(seeds)} seeds")
    print(f"Summary written to {out}")


if __name__ == "__main__":
    main()
