"""Generate a diverse set of graphs with NGD."""

import argparse

from ngd import load_config, run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: unseeded).")
    parser.add_argument("--output-dir", default="results", help="Where to write the graphs.")
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available).")
    args = parser.parse_args()

    overrides = {"seed": args.seed, "output_dir": args.output_dir}
    if args.device is not None:
        overrides["device"] = args.device
    run_pipeline(load_config(args.config, **overrides))


if __name__ == "__main__":
    main()
