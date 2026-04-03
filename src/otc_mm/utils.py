from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def outputs_root(methodology: str = "classic") -> Path:
    methodology = str(methodology)
    root = project_root() / "outputs" / methodology
    root.mkdir(parents=True, exist_ok=True)
    return root


def single_bond_dir(bond: int, methodology: str = "classic") -> Path:
    path = outputs_root(methodology) / "single_bond" / f"bond_{bond:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def multi_bond_dir(bonds: list[int] | tuple[int, ...], methodology: str = "classic") -> Path:
    sig = "_".join(f"{int(b):02d}" for b in bonds)
    path = outputs_root(methodology) / "multi_bond" / f"bonds_{sig}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def inventory_grid_for_bond(avg_size: float, width: int = 10) -> np.ndarray:
    size = float(avg_size)
    return np.arange(-width, width + 1, dtype=float) * size


def sample_inventory_box(
    avg_sizes: np.ndarray,
    n_samples: int,
    width: int = 10,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    avg_sizes = np.asarray(avg_sizes, dtype=float)
    d = avg_sizes.shape[0]

    low = -width * avg_sizes
    high = width * avg_sizes

    u = rng.uniform(size=(n_samples, d))
    q = low[None, :] + (high - low)[None, :] * u
    return q.astype(np.float32)


def targets_file_for_bond(bond: int, methodology: str = "classic") -> Path:
    return single_bond_dir(bond, methodology=methodology) / "targets.npz"


def metrics_file_for_bond(bond: int, methodology: str = "classic") -> Path:
    return single_bond_dir(bond, methodology=methodology) / "metrics.json"


def targets_exist_for_bond(bond: int, methodology: str = "classic") -> bool:
    return targets_file_for_bond(bond, methodology=methodology).exists()


def save_single_bond_targets(
    bond: int,
    inventories: np.ndarray,
    values: np.ndarray,
    deltas_bid: np.ndarray,
    deltas_ask: np.ndarray,
    methodology: str = "classic",
) -> None:
    path = targets_file_for_bond(bond, methodology=methodology)
    np.savez(
        path,
        inventories=np.asarray(inventories, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        deltas_bid=np.asarray(deltas_bid, dtype=np.float32),
        deltas_ask=np.asarray(deltas_ask, dtype=np.float32),
    )


def load_single_bond_targets(bond: int, methodology: str = "classic") -> dict[str, np.ndarray]:
    data = np.load(targets_file_for_bond(bond, methodology=methodology))
    return {
        "inventories": data["inventories"].astype(np.float32),
        "values": data["values"].astype(np.float32),
        "deltas_bid": data["deltas_bid"].astype(np.float32),
        "deltas_ask": data["deltas_ask"].astype(np.float32),
    }


def linear_interp_1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    return np.interp(x_new, x, y).astype(np.float32)


def additive_value_warmstart(
    q_multi: np.ndarray,
    bonds: list[int] | tuple[int, ...],
    target_loader,
) -> np.ndarray:
    q_multi = np.asarray(q_multi, dtype=float)
    bonds = list(map(int, bonds))

    out = np.zeros(q_multi.shape[0], dtype=np.float32)
    for j, bond in enumerate(bonds):
        tgt = target_loader(bond)
        out += linear_interp_1d(
            tgt["inventories"],
            tgt["values"],
            q_multi[:, j],
        )
    return out


def quote_warmstart_matrix(
    q_multi: np.ndarray,
    bonds: list[int] | tuple[int, ...],
    side: str,
    target_loader,
) -> np.ndarray:
    if side not in {"bid", "ask"}:
        raise ValueError("side must be 'bid' or 'ask'")

    q_multi = np.asarray(q_multi, dtype=float)
    bonds = list(map(int, bonds))
    n, d = q_multi.shape
    out = np.zeros((n, d), dtype=np.float32)

    key = "deltas_bid" if side == "bid" else "deltas_ask"

    for j, bond in enumerate(bonds):
        tgt = target_loader(bond)
        out[:, j] = linear_interp_1d(
            tgt["inventories"],
            tgt[key],
            q_multi[:, j],
        )
    return out