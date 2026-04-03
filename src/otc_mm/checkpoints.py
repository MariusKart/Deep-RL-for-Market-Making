from __future__ import annotations

from pathlib import Path

import torch


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _outputs_root(methodology: str = "classic") -> Path:
    methodology = str(methodology)
    out = _project_root() / "outputs" / methodology
    out.mkdir(parents=True, exist_ok=True)
    return out


def single_bond_checkpoint_path(bond: int, methodology: str = "classic") -> Path:
    path = _outputs_root(methodology) / "single_bond" / f"bond_{int(bond):02d}" / "model_checkpoint.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def multi_bond_checkpoint_path(selected_bonds, methodology: str = "classic") -> Path:
    sig = "_".join(f"{int(b):02d}" for b in selected_bonds)
    path = _outputs_root(methodology) / "multi_bond" / f"bonds_{sig}" / "model_checkpoint.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _serialize_actor(actor):
    """
    Supports:
    - classic MLP actor
    - greedy TableActor1D
    """
    if hasattr(actor, "table") and hasattr(actor, "lb_i") and hasattr(actor, "ub_i") and hasattr(actor, "size_i"):
        return {
            "actor_type": "table_1d",
            "i": int(actor.i),
            "lb_i": float(actor.lb_i),
            "ub_i": float(actor.ub_i),
            "size_i": float(actor.size_i),
            "table": actor.table.detach().cpu().clone(),
        }

    return {
        "actor_type": "mlp",
        "state_dict": actor.state_dict(),
    }


def save_single_bond_checkpoint(
    bond: int,
    actor,
    critic,
    hidden_dim: int | None,
    sizes,
    selected_bonds,
    methodology: str = "classic",
) -> Path:
    path = single_bond_checkpoint_path(bond, methodology=methodology)

    payload = {
        "kind": "single_bond",
        "methodology": str(methodology),
        "bond": int(bond),
        "selected_bonds": [int(b) for b in selected_bonds],
        "hidden_dim": None if hidden_dim is None else int(hidden_dim),
        "sizes": list(map(float, sizes)),
        "actor": _serialize_actor(actor),
        "critic_state_dict": critic.state_dict(),
    }

    torch.save(payload, path)
    return path


def save_multi_bond_checkpoint(
    selected_bonds,
    actors,
    critic,
    hidden_dim: int | None,
    sizes,
    methodology: str = "classic",
) -> Path:
    path = multi_bond_checkpoint_path(selected_bonds, methodology=methodology)

    payload = {
        "kind": "multi_bond",
        "methodology": str(methodology),
        "selected_bonds": [int(b) for b in selected_bonds],
        "hidden_dim": None if hidden_dim is None else int(hidden_dim),
        "sizes": list(map(float, sizes)),
        "actors": [_serialize_actor(a) for a in actors],
        "critic_state_dict": critic.state_dict(),
    }

    torch.save(payload, path)
    return path


def load_checkpoint(path, map_location="cpu"):
    return torch.load(Path(path), map_location=map_location)


def checkpoint_exists_for_single_bond(bond: int, methodology: str = "classic") -> bool:
    return single_bond_checkpoint_path(bond, methodology=methodology).exists()


def checkpoint_exists_for_multi_bond(selected_bonds, methodology: str = "classic") -> bool:
    return multi_bond_checkpoint_path(selected_bonds, methodology=methodology).exists()


def plots_dir_single_bond(bond: int, methodology: str = "classic") -> Path:
    path = _outputs_root(methodology) / "single_bond" / f"bond_{int(bond):02d}" / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plots_dir_multi_bond(selected_bonds, methodology: str = "classic") -> Path:
    sig = "_".join(f"{int(b):02d}" for b in selected_bonds)
    path = _outputs_root(methodology) / "multi_bond" / f"bonds_{sig}" / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path