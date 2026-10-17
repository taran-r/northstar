#!/usr/bin/env python
"""Generate data/dataset_v2_additions without touching data/dataset_v1.

Re-running the same command resumes: existing scenario seeds are skipped and
successful samples are kept.
"""

from __future__ import annotations

import argparse
import json

from northstar.pipeline.expand import ExpandConfig, expand_dataset


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Resumable v2 addition generator")
    p.add_argument("--out", default="data/dataset_v2_additions")
    p.add_argument("--v1", default="data/dataset_v1")
    p.add_argument("--n-scenarios", type=int, default=1300, help="Successful v2 scenarios to reach")
    p.add_argument("--facility", default="facilities/industrial_facility_v1.json")
    p.add_argument("--no-time-aware", action="store_true")
    args = p.parse_args(argv)
    cfg = ExpandConfig(
        out_dir=args.out,
        v1_dir=args.v1,
        n_scenarios=args.n_scenarios,
        facility_path=args.facility,
        use_time_aware=not args.no_time_aware,
    )
    meta = expand_dataset(cfg)
    print(json.dumps(meta["stats"], indent=2))
    print(meta["resume_command"])


if __name__ == "__main__":
    main()
