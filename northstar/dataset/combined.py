"""Train-only loader over dataset_v1 and dataset_v2_additions.

Validation and test splits are intentionally unsupported here. Evaluate those
sets with ``NorthstarTrajectoryDataset`` on each root separately so a v1 or v2
held-out scenario cannot enter fine-tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from torch.utils.data import Dataset

from northstar.dataset.pytorch import NorthstarTrajectoryDataset


class CombinedTrainDataset(Dataset):
    def __init__(self, roots: list[Union[str, Path]], split: str = "train"):
        if split != "train":
            raise ValueError(
                "CombinedTrainDataset only loads train splits. "
                "Evaluate data/dataset_v1 and data/dataset_v2_additions test sets separately."
            )
        self.parts = [NorthstarTrajectoryDataset(root, split="train") for root in roots]
        self.index = [(i, j) for i, part in enumerate(self.parts) for j in range(len(part))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        part, sample = self.index[idx]
        return self.parts[part][sample]
