"""CLI entrypoint: northstar-generate"""

from __future__ import annotations

import argparse
import json

from northstar.pipeline.generate import GenerateConfig, generate_dataset


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate Northstar expert trajectory dataset")
    p.add_argument("--out", default="data/dataset_v1", help="Output directory")
    p.add_argument("--n-scenarios", type=int, default=360)
    p.add_argument(
        "--min-windows",
        type=int,
        default=5000,
        help="Keep generating until at least this many training windows exist",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--multimodal-fraction", type=float, default=0.25)
    p.add_argument("--facility", default=None, help="Path to facility JSON")
    args = p.parse_args(argv)

    cfg = GenerateConfig(
        out_dir=args.out,
        n_scenarios=args.n_scenarios,
        min_windows=args.min_windows,
        seed=args.seed,
        dt=args.dt,
        horizon=args.horizon,
        multimodal_fraction=args.multimodal_fraction,
        facility_path=args.facility,
    )
    meta = generate_dataset(cfg)
    print(json.dumps(meta["stats"], indent=2))
    print(f"Wrote dataset to {args.out}")


if __name__ == "__main__":
    main()
