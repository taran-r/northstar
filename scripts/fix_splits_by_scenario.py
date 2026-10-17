#!/usr/bin/env python
"""Rewrite train/val/test splits so no scenario appears in more than one split.

Does not regenerate windows. Safe for the existing data/dataset_v1 corpus.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/dataset_v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--val-fraction", type=float, default=0.1)
    args = p.parse_args()

    root = Path(args.root)
    sample_dir = root / "samples"
    sidecars = sorted(sample_dir.glob("*.json"))
    by_scenario: dict[str, list[str]] = defaultdict(list)
    for sc in sidecars:
        info = json.loads(sc.read_text(encoding="utf-8"))
        sid = info.get("scenario_id")
        if not sid:
            # filenames like scn_00000_m0_w0000.json
            stem = sc.stem
            sid = stem.split("_m")[0]
        npz_name = sc.with_suffix(".npz").name
        if not (sample_dir / npz_name).exists():
            raise FileNotFoundError(npz_name)
        by_scenario[sid].append(npz_name)

    scenarios = sorted(by_scenario.keys())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(scenarios)

    n = len(scenarios)
    n_train = int(n * args.train_fraction)
    n_val = int(n * args.val_fraction)
    train_s = scenarios[:n_train]
    val_s = scenarios[n_train : n_train + n_val]
    test_s = scenarios[n_train + n_val :]

    splits = {
        "train": sorted(f for s in train_s for f in by_scenario[s]),
        "val": sorted(f for s in val_s for f in by_scenario[s]),
        "test": sorted(f for s in test_s for f in by_scenario[s]),
    }
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, files in splits.items():
        (split_dir / f"{name}.json").write_text(json.dumps(files, indent=2), encoding="utf-8")

    meta_path = root / "dataset_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["splits"] = {k: len(v) for k, v in splits.items()}
        meta["split_policy"] = {
            "type": "scenario",
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "n_scenarios": {
                "train": len(train_s),
                "val": len(val_s),
                "test": len(test_s),
            },
            "note": "Windows from the same scenario_id never cross splits.",
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "scenarios": {"train": len(train_s), "val": len(val_s), "test": len(test_s)},
                "windows": {k: len(v) for k, v in splits.items()},
                "total_windows": sum(len(v) for v in splits.values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
