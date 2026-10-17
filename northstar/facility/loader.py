"""Load / save versioned facility and scenario JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .schema import FacilitySpec, ScenarioSpec

PathLike = Union[str, Path]


def load_facility(path: PathLike) -> FacilitySpec:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return FacilitySpec.from_dict(data)


def save_facility(facility: FacilitySpec, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(facility.to_dict(), f, indent=2)


def load_scenario(path: PathLike) -> ScenarioSpec:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ScenarioSpec.from_dict(data)


def save_scenario(scenario: ScenarioSpec, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenario.to_dict(), f, indent=2)
