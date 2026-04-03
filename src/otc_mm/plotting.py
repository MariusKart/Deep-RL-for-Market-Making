from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .checkpoints import (
    load_checkpoint,
    multi_bond_checkpoint_path,
    plots_dir_multi_bond,
    plots_dir_single_bond,
    single_bond_checkpoint_path,
)
from .constants import ARRIVAL_RATES, AVG_SIZES, RF, subset_array, subset_cov, subset_f_parameters
from .greedy_actors import TableActor1D
from .market import Market
from .models import Actor, Critic
from .utils import load_single_bond_targets


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _single_bond_metrics_path(bond: int, methodology: str = "classic") -> Path:
    return _project_root() / "outputs" / methodology / "single_bond" / f"bond_{int(bond):02d}" / "metrics.json"


def _multi_bond_metrics_path(selected_bonds, methodology: str = "classic") -> Path:
    sig = "_".join(f"{int(b):02d}" for b in selected_bonds)
    return _project_root() / "outputs" / methodology / "multi_bond" / f"bonds_{sig}" / "metrics.json"


def _load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_market_for_selected_bonds(selected_bonds, seed: int = 42) -> tuple[Market, np.ndarray, np.ndarray]:
    selected_bonds = list(map(int, selected_bonds))
    market = Market(
        lambdas_RFQ=subset_array(ARRIVAL_RATES, selected_bonds),
        Sigma=subset_cov(selected_bonds),
        rf=RF,
        sizes=subset_array(AVG_SIZES, selected_bonds),
        f_parameters=subset_f_parameters(selected_bonds),
        seed=seed,
    )
    sizes = subset_array(AVG_SIZES, selected_bonds)
    sigma = subset_cov(selected_bonds)
    return market, sizes, sigma


def _rebuild_actor_from_payload(actor_payload, state_dim: int, hidden_dim: int | None):
    actor_type = actor_payload["actor_type"]

    if actor_type == "mlp":
        actor = Actor(state_dim=state_dim, hidden=int(hidden_dim))
        actor.load_state_dict(actor_payload["state_dict"])
        actor.eval()
        return actor

    if actor_type == "table_1d":
        actor = TableActor1D(
            i=int(actor_payload["i"]),
            lb_i=float(actor_payload["lb_i"]),
            ub_i=float(actor_payload["ub_i"]),
            size_i=float(actor_payload["size_i"]),
            init_p=0.5,
        )
        actor.set_table(actor_payload["table"].detach().cpu().numpy())
        actor.eval()
        return actor

    raise ValueError(f"Unknown actor_type: {actor_type}")


def load_single_bond_models(bond: int, methodology: str = "classic", map_location: str = "cpu"):
    ckpt = load_checkpoint(
        single_bond_checkpoint_path(bond, methodology=methodology),
        map_location=map_location,
    )

    hidden_dim = ckpt["hidden_dim"]
    actor = _rebuild_actor_from_payload(
        ckpt["actor"],
        state_dim=1,
        hidden_dim=hidden_dim,
    )

    critic = Critic(state_dim=1, hidden=int(hidden_dim))
    critic.load_state_dict(ckpt["critic_state_dict"])
    critic.eval()

    return actor, critic, ckpt


def load_multi_bond_models(selected_bonds, methodology: str = "classic", map_location: str = "cpu"):
    selected_bonds = list(map(int, selected_bonds))
    ckpt = load_checkpoint(
        multi_bond_checkpoint_path(selected_bonds, methodology=methodology),
        map_location=map_location,
    )

    hidden_dim = ckpt["hidden_dim"]
    d = len(selected_bonds)

    actors = [
        _rebuild_actor_from_payload(
            payload,
            state_dim=d,
            hidden_dim=hidden_dim,
        )
        for payload in ckpt["actors"]
    ]

    critic = Critic(state_dim=d, hidden=int(hidden_dim))
    critic.load_state_dict(ckpt["critic_state_dict"])
    critic.eval()

    return actors, critic, ckpt


def rolling_mean(x, window: int):
    x = np.asarray(x, dtype=float).reshape(-1)
    if window <= 1:
        return x.copy()
    if len(x) == 0:
        return x.copy()

    out = np.empty_like(x, dtype=float)
    csum = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = (csum[i + 1] - csum[lo]) / (i - lo + 1)
    return out


def build_single_bond_df(bond: int, methodology: str = "classic") -> pd.DataFrame:
    tgt = load_single_bond_targets(int(bond), methodology=methodology)
    market, _, _ = make_market_for_selected_bonds([int(bond)])

    p_bid = np.asarray(market.f(0, tgt["deltas_bid"]), dtype=np.float32)
    p_ask = np.asarray(market.f(0, tgt["deltas_ask"]), dtype=np.float32)

    df = pd.DataFrame({
        "inventory": np.asarray(tgt["inventories"], dtype=np.float32),
        "value": np.asarray(tgt["values"], dtype=np.float32),
        "optimal_delta_bid": np.asarray(tgt["deltas_bid"], dtype=np.float32),
        "optimal_delta_ask": np.asarray(tgt["deltas_ask"], dtype=np.float32),
        "optimal_prob_bid": p_bid,
        "optimal_prob_ask": p_ask,
    })
    return df.sort_values("inventory").reset_index(drop=True)


def build_single_bond_reward_df(bond: int, methodology: str = "classic") -> pd.DataFrame:
    metrics = _load_json(_single_bond_metrics_path(int(bond), methodology=methodology))
    avg_reward = np.asarray(metrics["history"]["avg_reward_long"], dtype=float)
    return pd.DataFrame({
        "iteration": np.arange(len(avg_reward), dtype=int),
        "avg_reward_long": avg_reward,
    })


def plot_single_bond_bundle(bond: int, methodology: str = "classic", save_csv: bool = True):
    out_dir = plots_dir_single_bond(int(bond), methodology=methodology)
    df = build_single_bond_df(int(bond), methodology=methodology)
    reward_df = build_single_bond_reward_df(int(bond), methodology=methodology)

    if save_csv:
        df.to_csv(out_dir / "single_bond_curves.csv", index=False)
        reward_df.to_csv(out_dir / "single_bond_avg_reward.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["inventory"], df["optimal_delta_bid"], label="optimal_delta_bid")
    plt.plot(df["inventory"], df["optimal_delta_ask"], label="optimal_delta_ask")
    plt.xlabel("inventory")
    plt.title(f"{methodology} - Bond {int(bond):02d} optimal quotes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "quotes.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["inventory"], df["optimal_prob_bid"], label="optimal_prob_bid")
    plt.plot(df["inventory"], df["optimal_prob_ask"], label="optimal_prob_ask")
    plt.xlabel("inventory")
    plt.title(f"{methodology} - Bond {int(bond):02d} optimal probabilities")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "probabilities.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(reward_df["iteration"], reward_df["avg_reward_long"])
    plt.xlabel("iteration")
    plt.title(f"{methodology} - Bond {int(bond):02d} training avg reward")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_reward_long.png", dpi=160)
    plt.close()

    return {
        "curve_df": df,
        "reward_df": reward_df,
        "out_dir": out_dir,
    }


def build_multi_bond_reward_df(selected_bonds, methodology: str = "classic", rolling_window: int = 25) -> pd.DataFrame:
    selected_bonds = list(map(int, selected_bonds))
    metrics = _load_json(_multi_bond_metrics_path(selected_bonds, methodology=methodology))
    avg_reward = np.asarray(metrics["history"]["avg_reward_long"], dtype=float)

    df = pd.DataFrame({
        "iteration": np.arange(len(avg_reward), dtype=int),
        "avg_reward_long": avg_reward,
    })
    df["rolling_avg_reward_long"] = rolling_mean(df["avg_reward_long"].to_numpy(), rolling_window)
    return df


def plot_multi_bond_learning_curve(selected_bonds, methodology: str = "classic", rolling_window: int = 25, save_csv: bool = True):
    selected_bonds = list(map(int, selected_bonds))
    out_dir = plots_dir_multi_bond(selected_bonds, methodology=methodology)
    df = build_multi_bond_reward_df(selected_bonds, methodology=methodology, rolling_window=rolling_window)

    if save_csv:
        df.to_csv(out_dir / "avg_reward_long_scatter_rolling.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.scatter(df["iteration"], df["avg_reward_long"], s=16, alpha=0.7, label="avg_reward_long")
    plt.plot(df["iteration"], df["rolling_avg_reward_long"], linewidth=2, label=f"rolling_mean_{rolling_window}")
    plt.xlabel("iteration")
    plt.title(f"{methodology} - Multi-bond learning curve {selected_bonds}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "avg_reward_long_scatter_rolling.png", dpi=160)
    plt.close()

    return {
        "reward_df": df,
        "out_dir": out_dir,
    }


def _build_inventory_grid_2d(sizes: np.ndarray, width_in_sizes: int = 5):
    if len(sizes) != 2:
        raise ValueError("2D plotting requires exactly 2 bonds")

    axis0 = np.arange(-width_in_sizes * sizes[0], (width_in_sizes + 1) * sizes[0], sizes[0], dtype=np.float32)
    axis1 = np.arange(-width_in_sizes * sizes[1], (width_in_sizes + 1) * sizes[1], sizes[1], dtype=np.float32)

    Q0, Q1 = np.meshgrid(axis0, axis1, indexing="ij")
    q_grid = np.stack([Q0.ravel(), Q1.ravel()], axis=1).astype(np.float32)

    return axis0, axis1, q_grid


def _actor_outputs_for_2d(actor, market: Market, local_bond_idx: int, x_norm: np.ndarray):
    with torch.inference_mode():
        x_bid_t = torch.as_tensor(x_norm, dtype=torch.float32)
        x_ask_t = torch.as_tensor(-x_norm, dtype=torch.float32)

        p_bid = actor(x_bid_t)
        p_ask = actor(x_ask_t)

        if p_bid.dim() == 2:
            p_bid = p_bid[:, 0]
        if p_ask.dim() == 2:
            p_ask = p_ask[:, 0]

        p_bid = p_bid.detach().cpu().numpy().astype(np.float32)
        p_ask = p_ask.detach().cpu().numpy().astype(np.float32)

    p_bid = np.clip(p_bid, 1e-6, 1.0 - 1e-6)
    p_ask = np.clip(p_ask, 1e-6, 1.0 - 1e-6)

    delta_bid = np.asarray(market.inv_f(local_bond_idx, p_bid), dtype=np.float32)
    delta_ask = np.asarray(market.inv_f(local_bond_idx, p_ask), dtype=np.float32)

    return p_bid, p_ask, delta_bid, delta_ask


def _critic_outputs_2d(critic, x_norm: np.ndarray):
    with torch.inference_mode():
        x_t = torch.as_tensor(x_norm, dtype=torch.float32)
        v = critic(x_t)
        if v.dim() == 2:
            v = v[:, 0]
        return v.detach().cpu().numpy().astype(np.float32)


def build_two_bond_surface_dfs(selected_bonds, methodology: str = "classic", width_in_sizes: int = 5, map_location: str = "cpu"):
    selected_bonds = list(map(int, selected_bonds))
    if len(selected_bonds) != 2:
        raise ValueError("This plotting function is only for 2 bonds")

    actors, critic, _ = load_multi_bond_models(
        selected_bonds,
        methodology=methodology,
        map_location=map_location,
    )
    market, sizes, _ = make_market_for_selected_bonds(selected_bonds)

    _, _, q_grid = _build_inventory_grid_2d(sizes=sizes, width_in_sizes=width_in_sizes)
    x_norm = q_grid / sizes.reshape(1, -1)

    q_col_0 = f"q_bond_{selected_bonds[0]}"
    q_col_1 = f"q_bond_{selected_bonds[1]}"

    v = _critic_outputs_2d(critic, x_norm)
    critic_df = pd.DataFrame({
        q_col_0: q_grid[:, 0],
        q_col_1: q_grid[:, 1],
        "critic_value": v,
    })

    actor_dfs = []
    for local_j, global_bond in enumerate(selected_bonds):
        p_bid, p_ask, delta_bid, delta_ask = _actor_outputs_for_2d(
            actor=actors[local_j],
            market=market,
            local_bond_idx=local_j,
            x_norm=x_norm,
        )

        df_j = pd.DataFrame({
            q_col_0: q_grid[:, 0],
            q_col_1: q_grid[:, 1],
            "bond_local_idx": local_j,
            "bond_global_idx": global_bond,
            "optimal_prob_bid": p_bid,
            "optimal_prob_ask": p_ask,
            "optimal_delta_bid": delta_bid,
            "optimal_delta_ask": delta_ask,
        })
        actor_dfs.append(df_j)

    return critic_df, actor_dfs


def _save_surface(df, x_col, y_col, value_col, title, out_file):
    pivot = df.pivot(index=y_col, columns=x_col, values=value_col).sort_index().sort_index(axis=1)

    x_vals = pivot.columns.to_numpy(dtype=float)
    y_vals = pivot.index.to_numpy(dtype=float)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pivot.to_numpy(dtype=float)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap= "viridis", edgecolor="black")
    ax.view_init(elev=20, azim=-60)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(value_col)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()


def plot_two_bond_surfaces(selected_bonds, methodology: str = "classic", width_in_sizes: int = 5, save_csv: bool = True, map_location: str = "cpu"):
    selected_bonds = list(map(int, selected_bonds))
    if len(selected_bonds) != 2:
        raise ValueError("This plotting function is only for 2 bonds")

    out_dir = plots_dir_multi_bond(selected_bonds, methodology=methodology)
    critic_df, actor_dfs = build_two_bond_surface_dfs(
        selected_bonds=selected_bonds,
        methodology=methodology,
        width_in_sizes=width_in_sizes,
        map_location=map_location,
    )

    q_col_0 = f"q_bond_{selected_bonds[0]}"
    q_col_1 = f"q_bond_{selected_bonds[1]}"

    if save_csv:
        critic_df.to_csv(out_dir / "critic_surface.csv", index=False)

    _save_surface(
        critic_df,
        x_col=q_col_1,
        y_col=q_col_0,
        value_col="critic_value",
        title=f"{methodology} - Critic value surface for bonds {selected_bonds}",
        out_file=out_dir / "critic_value_surface.png",
    )

    for df_j in actor_dfs:
        global_bond = int(df_j["bond_global_idx"].iloc[0])

        if save_csv:
            df_j.to_csv(out_dir / f"actor_surface_bond_{global_bond:02d}.csv", index=False)

        _save_surface(
            df_j,
            x_col=q_col_1,
            y_col=q_col_0,
            value_col="optimal_prob_bid",
            title=f"{methodology} - Bond {global_bond} optimal bid probability",
            out_file=out_dir / f"bond_{global_bond:02d}_prob_bid_surface.png",
        )
        _save_surface(
            df_j,
            x_col=q_col_1,
            y_col=q_col_0,
            value_col="optimal_prob_ask",
            title=f"{methodology} - Bond {global_bond} optimal ask probability",
            out_file=out_dir / f"bond_{global_bond:02d}_prob_ask_surface.png",
        )
        _save_surface(
            df_j,
            x_col=q_col_1,
            y_col=q_col_0,
            value_col="optimal_delta_bid",
            title=f"{methodology} - Bond {global_bond} optimal bid quote",
            out_file=out_dir / f"bond_{global_bond:02d}_quote_bid_surface.png",
        )
        _save_surface(
            df_j,
            x_col=q_col_1,
            y_col=q_col_0,
            value_col="optimal_delta_ask",
            title=f"{methodology} - Bond {global_bond} optimal ask quote",
            out_file=out_dir / f"bond_{global_bond:02d}_quote_ask_surface.png",
        )

    return {
        "critic_df": critic_df,
        "actor_dfs": actor_dfs,
        "out_dir": out_dir,
    }