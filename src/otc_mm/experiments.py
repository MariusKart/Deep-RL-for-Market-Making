from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize

from .checkpoints import (
    save_multi_bond_checkpoint,
    save_single_bond_checkpoint,
)
from .constants import (
    ACTOR_BATCH_SIZE,
    ACTOR_LR,
    ARRIVAL_RATES,
    AVG_SIZES,
    CRITIC_BATCH_SIZE,
    CRITIC_LR,
    DEFAULT_DEVICE,
    DEFAULT_SEED,
    LONG_HORIZON,
    NB_SHORT_ROLLOUTS,
    RF,
    SHORT_HORIZON,
    subset_array,
    subset_cov,
    subset_f_parameters,
)
from .greedy_actors import build_table_actors
from .market import Market
from .models import Actor, Critic
from .pretraining import (
    pretrain_actor,
    pretrain_critic,
    pretrain_multi_bond_from_single_bond_targets,
    produce_initial_value_grid,
)
from .training import train_final
from .training_greedy import train_final_greedy
from .utils import (
    metrics_file_for_bond,
    multi_bond_dir,
    save_json,
    save_single_bond_targets,
    set_seed,
    targets_exist_for_bond,
)


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _validate_methodology(methodology: str) -> str:
    methodology = str(methodology).lower().strip()
    if methodology not in {"classic", "greedy"}:
        raise ValueError("methodology must be either 'classic' or 'greedy'")
    return methodology


def _default_hidden_dim(nb_bonds: int) -> int:
    if nb_bonds == 1:
        return 10
    if nb_bonds == 2:
        return 12
    return 30


def _default_bounds_for_selected_bonds(selected_bonds: list[int]):
    """
    - 1 bond : init = final = ±5 sizes
    - >1 bond: init = ±3 sizes, final = ±5 sizes
    """
    selected_bonds = list(map(int, selected_bonds))
    sizes = subset_array(AVG_SIZES, selected_bonds)

    if len(selected_bonds) == 1:
        lb_init = -5.0 * sizes
        ub_init = 5.0 * sizes
        lb_final = lb_init.copy()
        ub_final = ub_init.copy()
    elif len(selected_bonds) ==2:
        lb_init = -3.0 * sizes
        ub_init = 3.0 * sizes
        lb_final = -5.0 * sizes
        ub_final = 5.0 * sizes
    else:
        lb_init = -3.0 * sizes
        ub_init = 3.0 * sizes
        lb_final = -10.0 * sizes
        ub_final = 10.0 * sizes

    return lb_init.astype(float), ub_init.astype(float), lb_final.astype(float), ub_final.astype(float)


def _resolve_bounds(selected_bonds, lb_init, ub_init, lb_final, ub_final):
    d = len(selected_bonds)

    if lb_init is None or ub_init is None or lb_final is None or ub_final is None:
        d_lb_init, d_ub_init, d_lb_final, d_ub_final = _default_bounds_for_selected_bonds(selected_bonds)

        if lb_init is None:
            lb_init = d_lb_init
        if ub_init is None:
            ub_init = d_ub_init
        if lb_final is None:
            lb_final = d_lb_final
        if ub_final is None:
            ub_final = d_ub_final

    lb_init = np.asarray(lb_init, dtype=float).reshape(d)
    ub_init = np.asarray(ub_init, dtype=float).reshape(d)
    lb_final = np.asarray(lb_final, dtype=float).reshape(d)
    ub_final = np.asarray(ub_final, dtype=float).reshape(d)

    return lb_init, ub_init, lb_final, ub_final


def _make_market(selected_bonds, seed=DEFAULT_SEED):
    """
    Important intentional portability fix:
    we subset f-parameters to the selected bonds so local actor indices
    and local market indices stay aligned for any basket.
    """
    selected_bonds = list(map(int, selected_bonds))

    intensities = subset_array(ARRIVAL_RATES, selected_bonds)
    sizes = subset_array(AVG_SIZES, selected_bonds)
    sigma = subset_cov(selected_bonds)
    f_params = subset_f_parameters(selected_bonds)

    market = Market(
        lambdas_RFQ=intensities,
        Sigma=sigma,
        rf=RF,
        sizes=sizes,
        f_parameters=f_params,
        seed=seed,
    )
    return market, sizes, sigma


def _myopic_probs_local(market_env: Market):
    """
    0..d-1 so the implementation remains correct for arbitrary baskets.
    """
    d = len(market_env.lambda_RFQs)
    p = np.zeros(d, dtype=float)
    delta_star = np.zeros(d, dtype=float)

    for i in range(d):
        def objective(x):
            delta = x[0]
            return -(delta * market_env.f(i, delta))

        res = minimize(
            objective,
            x0=[1.0],
            bounds=[(0.005, 1e12)],
        )
        delta_i = float(res.x[0])
        delta_star[i] = delta_i
        p[i] = float(market_env.f(i, delta_i))

    return p, delta_star


def _extract_single_bond_learned_dataset(
    actor,
    critic,
    market: Market,
    size: float,
    lb_final: float,
    ub_final: float,
    device: str = DEFAULT_DEVICE,
):
    """
    Save the learned 1D objects after final training:
    - inventory grid
    - learned value function
    - learned bid quote curve
    - learned ask quote curve

    Symmetry is preserved:
    - bid uses actor(q/size)
    - ask uses actor(-q/size)
    """
    q_grid = np.arange(lb_final, ub_final + size, size, dtype=np.float32).reshape(-1)
    x_bid = (q_grid / float(size)).reshape(-1, 1).astype(np.float32)
    x_ask = (-q_grid / float(size)).reshape(-1, 1).astype(np.float32)

    actor = actor.to(device)
    critic = critic.to(device)
    actor.eval()
    critic.eval()

    with torch.inference_mode():
        x_bid_t = torch.as_tensor(x_bid, dtype=torch.float32, device=device)
        x_ask_t = torch.as_tensor(x_ask, dtype=torch.float32, device=device)

        values = critic(x_bid_t)
        if values.dim() == 2:
            values = values[:, 0]
        values = values.detach().cpu().numpy().astype(np.float32)

        p_bid = actor(x_bid_t)
        if p_bid.dim() == 2:
            p_bid = p_bid[:, 0]
        p_bid = np.clip(p_bid.detach().cpu().numpy().astype(np.float32), 1e-6, 1.0 - 1e-6)

        p_ask = actor(x_ask_t)
        if p_ask.dim() == 2:
            p_ask = p_ask[:, 0]
        p_ask = np.clip(p_ask.detach().cpu().numpy().astype(np.float32), 1e-6, 1.0 - 1e-6)

    deltas_bid = np.asarray(market.inv_f(0, p_bid), dtype=np.float32)
    deltas_ask = np.asarray(market.inv_f(0, p_ask), dtype=np.float32)

    return q_grid, values, deltas_bid, deltas_ask


# ---------------------------------------------------------------------
# Single-bond experiment
# ---------------------------------------------------------------------

def run_single_bond_experiment(
    bond: int,
    methodology: str = "classic",
    hidden_dim: int | None = None,
    nb_steps: int = 50,
    long_horizon: int = LONG_HORIZON,
    nb_short_rollouts: int = NB_SHORT_ROLLOUTS,
    short_horizon: int = SHORT_HORIZON,
    critic_batch_size: int = CRITIC_BATCH_SIZE,
    actor_batch_size: int = ACTOR_BATCH_SIZE,
    n_epochs_critic: int = 1,
    n_epochs_actor: int = 1,
    update_risk_after: int = 50,
    critic_lr: float = 1e-2,
    actor_lr: float = ACTOR_LR,
    pretrain_actor_lr: float = 1e-3,
    pretrain_actor_epochs: int = 1000,
    pretrain_critic_lr: float = 1e-2,
    pretrain_critic_max_steps: int = 10000,
    pretrain_critic_n_dense: int = 5000,
    lb_init=None,
    ub_init=None,
    lb_final=None,
    ub_final=None,
    seed: int = DEFAULT_SEED,
    device: str = DEFAULT_DEVICE,
):
    methodology = _validate_methodology(methodology)
    set_seed(seed)

    selected_bonds = [int(bond)]
    nb_bonds = 1
    hidden_dim = _default_hidden_dim(nb_bonds) if hidden_dim is None else int(hidden_dim)

    market, sizes, sigma = _make_market(selected_bonds, seed=seed)

    lb_init, ub_init, lb_final, ub_final = _resolve_bounds(
        selected_bonds=selected_bonds,
        lb_init=lb_init,
        ub_init=ub_init,
        lb_final=lb_final,
        ub_final=ub_final,
    )

    critic = Critic(state_dim=nb_bonds, hidden=hidden_dim)

    # common myopic quantities
    myopic_p, myopic_delta = _myopic_probs_local(market)

    if methodology == "classic":
        actors = [Actor(nb_bonds, hidden_dim)]

        # 1) myopic actor pretraining
        for i in range(nb_bonds):
            pretrain_actor(
                actors[i],
                myopic_p[i],
                lb_init,
                ub_init,
                avg_sizes=sizes,
                batch_size=50,
                epochs=pretrain_actor_epochs,
                lr=pretrain_actor_lr,
                device=device,
            )

        # 2) critic pretraining from FD value grid
        grids, Vs = produce_initial_value_grid(
            market=market,
            init_strategy=myopic_delta,
            lb_risk=lb_init,
            ub_risk=ub_init + sizes,
            nb_bonds=nb_bonds,
            sizes=sizes,
        )

        critic = pretrain_critic(
            critic=critic,
            grids=grids,
            Vs=Vs,
            sizes=sizes,
            device=device,
            n_dense=pretrain_critic_n_dense,
            lr=pretrain_critic_lr,
            max_steps=pretrain_critic_max_steps,
            tol=1e-4,
        )

        results = train_final(
            actors=actors,
            critic=critic,
            market=market,
            r=RF,
            Sigma=sigma,
            nb_steps=nb_steps,
            avg_sizes=sizes,
            long_horizon=long_horizon,
            nb_short_rollouts=nb_short_rollouts,
            short_horizon=short_horizon,
            critic_batch_size=critic_batch_size,
            actor_batch_size=actor_batch_size,
            n_epochs_critic=n_epochs_critic,
            n_epochs_actor=n_epochs_actor,
            update_risk_after=update_risk_after,
            LB_init=lb_init,
            UB_init=ub_init,
            LB_max=lb_final,
            UB_max=ub_final,
            critic_lr=critic_lr,
            actor_lr=actor_lr,
            seed=seed,
            device=device,
        )

    else:  # greedy
        actors, _, _ = build_table_actors(
            sizes=sizes,
            lb_init=lb_init,
            ub_init=ub_init,
            market_simulator=market,
        )

        grids, Vs = produce_initial_value_grid(
            market=market,
            init_strategy=myopic_delta,
            lb_risk=lb_init,
            ub_risk=ub_init + sizes,
            nb_bonds=nb_bonds,
            sizes=sizes,
        )

        critic = pretrain_critic(
            critic=critic,
            grids=grids,
            Vs=Vs,
            sizes=sizes,
            device=device,
            n_dense=pretrain_critic_n_dense,
            lr=pretrain_critic_lr,
            max_steps=pretrain_critic_max_steps,
            tol=1e-4,
        )

        results = train_final_greedy(
            actors=actors,
            critic=critic,
            market=market,
            r=RF,
            Sigma=sigma,
            nb_steps=nb_steps,
            avg_sizes=sizes,
            long_horizon=long_horizon,
            nb_short_rollouts=nb_short_rollouts,
            short_horizon=short_horizon,
            critic_batch_size=critic_batch_size,
            n_epochs_critic=n_epochs_critic,
            LB_init=lb_init,
            UB_init=ub_init,
            LB_max=lb_final,
            UB_max=ub_final,
            critic_lr=critic_lr,
            update_risk_after=update_risk_after,
            seed=seed,
            device=device,
        )

    inventories, values, deltas_bid, deltas_ask = _extract_single_bond_learned_dataset(
        actor=actors[0],
        critic=critic,
        market=market,
        size=float(sizes[0]),
        lb_final=float(lb_final[0]),
        ub_final=float(ub_final[0]),
        device=device,
    )

    save_single_bond_targets(
        bond=int(bond),
        inventories=inventories,
        values=values,
        deltas_bid=deltas_bid,
        deltas_ask=deltas_ask,
        methodology=methodology,
    )

    checkpoint_path = save_single_bond_checkpoint(
        bond=int(bond),
        actor=actors[0],
        critic=critic,
        hidden_dim=hidden_dim,
        sizes=sizes,
        selected_bonds=selected_bonds,
        methodology=methodology,
    )

    metrics = {
        "bond": int(bond),
        "selected_bonds": [int(bond)],
        "methodology": methodology,
        "hidden_dim": int(hidden_dim),
        "nb_steps": int(nb_steps),
        "long_horizon": int(long_horizon),
        "nb_short_rollouts": int(nb_short_rollouts),
        "short_horizon": int(short_horizon),
        "critic_batch_size": int(critic_batch_size),
        "actor_batch_size": int(actor_batch_size),
        "n_epochs_critic": int(n_epochs_critic),
        "n_epochs_actor": int(n_epochs_actor),
        "update_risk_after": None if update_risk_after is None else int(update_risk_after),
        "critic_lr": float(critic_lr),
        "actor_lr": float(actor_lr),
        "pretrain_actor_lr": float(pretrain_actor_lr),
        "pretrain_actor_epochs": int(pretrain_actor_epochs),
        "pretrain_critic_lr": float(pretrain_critic_lr),
        "pretrain_critic_max_steps": int(pretrain_critic_max_steps),
        "pretrain_critic_n_dense": int(pretrain_critic_n_dense),
        "lb_init": lb_init.tolist(),
        "ub_init": ub_init.tolist(),
        "lb_final": lb_final.tolist(),
        "ub_final": ub_final.tolist(),
        "myopic_p": np.asarray(myopic_p, dtype=float).tolist(),
        "myopic_delta": np.asarray(myopic_delta, dtype=float).tolist(),
        "checkpoint_path": str(checkpoint_path),
        "history": {
            "avg_reward_long": np.asarray(results["avg_reward_long"], dtype=float).tolist(),
            "critic_loss": np.asarray(results["critic_loss"], dtype=float).tolist(),
        },
        "final_avg_reward_long": float(np.asarray(results["avg_reward_long"], dtype=float)[-1]),
    }

    save_json(metrics_file_for_bond(int(bond), methodology=methodology), metrics)

    return {
        "actors": actors,
        "critic": critic,
        "market": market,
        "sizes": sizes,
        "sigma": sigma,
        "results": results,
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
    }


def ensure_single_bond_targets(
    selected_bonds,
    methodology: str = "classic",
    device: str = DEFAULT_DEVICE,
    seed: int = DEFAULT_SEED,
):
    """
    If a required 1D learned dataset is missing, rerun that single-bond case
    under the requested methodology.
    """
    methodology = _validate_methodology(methodology)
    selected_bonds = list(map(int, selected_bonds))

    for k, bond in enumerate(selected_bonds):
        if not targets_exist_for_bond(int(bond), methodology=methodology):
            run_single_bond_experiment(
                bond=int(bond),
                methodology=methodology,
                seed=seed + 10_000 * k,
                device=device,
            )


# ---------------------------------------------------------------------
# Multi-bond experiment
# ---------------------------------------------------------------------

def run_multi_bond_experiment(
    selected_bonds,
    methodology: str = "classic",
    hidden_dim: int | None = None,
    nb_steps: int = 500,
    long_horizon: int = LONG_HORIZON,
    nb_short_rollouts: int = NB_SHORT_ROLLOUTS,
    short_horizon: int = SHORT_HORIZON,
    critic_batch_size: int = CRITIC_BATCH_SIZE,
    actor_batch_size: int = ACTOR_BATCH_SIZE,
    n_epochs_critic: int = 1,
    n_epochs_actor: int = 1,
    update_risk_after: int = 50,
    critic_lr: float = 1e-2,
    actor_lr: float = ACTOR_LR,
    # classic dataset warm-start params
    dataset_pretrain_samples: int = 20000,
    dataset_pretrain_actor_lr: float = 1e-3,
    dataset_pretrain_actor_epochs: int = 1000,
    dataset_pretrain_actor_batch_size: int = 256,
    dataset_pretrain_critic_lr: float = 1e-3,
    dataset_pretrain_critic_max_steps: int = 3000,
    dataset_pretrain_critic_tol: float = 1e-4,
    lb_init=None,
    ub_init=None,
    lb_final=None,
    ub_final=None,
    seed: int = DEFAULT_SEED,
    device: str = DEFAULT_DEVICE,
):
    methodology = _validate_methodology(methodology)
    selected_bonds = list(map(int, selected_bonds))
    if len(selected_bonds) == 0:
        raise ValueError("selected_bonds cannot be empty")

    set_seed(seed)
    ensure_single_bond_targets(selected_bonds, methodology=methodology, device=device, seed=seed)

    nb_bonds = len(selected_bonds)
    hidden_dim = _default_hidden_dim(nb_bonds) if hidden_dim is None else int(hidden_dim)

    market, sizes, sigma = _make_market(selected_bonds, seed=seed)

    lb_init, ub_init, lb_final, ub_final = _resolve_bounds(
        selected_bonds=selected_bonds,
        lb_init=lb_init,
        ub_init=ub_init,
        lb_final=lb_final,
        ub_final=ub_final,
    )

    critic = Critic(state_dim=nb_bonds, hidden=hidden_dim)

    if methodology == "classic":
        actors = [Actor(nb_bonds, hidden_dim) for _ in range(nb_bonds)]

        actors, critic, pretrain_payload = pretrain_multi_bond_from_single_bond_targets(
            actors=actors,
            critic=critic,
            selected_bonds=selected_bonds,
            avg_sizes=sizes,
            LB_risk=lb_init,
            UB_risk=ub_init,
            market=market,
            methodology=methodology,
            n_samples=dataset_pretrain_samples,
            critic_lr=dataset_pretrain_critic_lr,
            critic_max_steps=dataset_pretrain_critic_max_steps,
            critic_tol=dataset_pretrain_critic_tol,
            actor_lr=dataset_pretrain_actor_lr,
            actor_epochs=dataset_pretrain_actor_epochs,
            actor_batch_size=dataset_pretrain_actor_batch_size,
            device=device,
            seed=seed,
        )

        results = train_final(
            actors=actors,
            critic=critic,
            market=market,
            r=RF,
            Sigma=sigma,
            nb_steps=nb_steps,
            avg_sizes=sizes,
            long_horizon=long_horizon,
            nb_short_rollouts=nb_short_rollouts,
            short_horizon=short_horizon,
            critic_batch_size=critic_batch_size,
            actor_batch_size=actor_batch_size,
            n_epochs_critic=n_epochs_critic,
            n_epochs_actor=n_epochs_actor,
            update_risk_after=update_risk_after,
            LB_init=lb_init,
            UB_init=ub_init,
            LB_max=lb_final,
            UB_max=ub_final,
            critic_lr=critic_lr,
            actor_lr=actor_lr,
            seed=seed,
            device=device,
        )

    else:  # greedy
        actors, myopic_p, myopic_delta = build_table_actors(
            sizes=sizes,
            lb_init=lb_init,
            ub_init=ub_init,
            market_simulator=market,
        )

        grids, Vs = produce_initial_value_grid(
            market=market,
            init_strategy=myopic_delta,
            lb_risk=lb_init,
            ub_risk=ub_init + sizes,
            nb_bonds=nb_bonds,
            sizes=sizes,
        )

        critic = pretrain_critic(
            critic=critic,
            grids=grids,
            Vs=Vs,
            sizes=sizes,
            device=device,
            n_dense=100,
            lr=1e-2,
            max_steps=3000,
            tol=1e-4,
        )

        results = train_final_greedy(
            actors=actors,
            critic=critic,
            market=market,
            r=RF,
            Sigma=sigma,
            nb_steps=nb_steps,
            avg_sizes=sizes,
            long_horizon=long_horizon,
            nb_short_rollouts=nb_short_rollouts,
            short_horizon=short_horizon,
            critic_batch_size=critic_batch_size,
            n_epochs_critic=n_epochs_critic,
            LB_init=lb_init,
            UB_init=ub_init,
            LB_max=lb_final,
            UB_max=ub_final,
            critic_lr=critic_lr,
            update_risk_after=update_risk_after,
            seed=seed,
            device=device,
        )
        pretrain_payload = {
            "myopic_p": np.asarray(myopic_p, dtype=float),
            "myopic_delta": np.asarray(myopic_delta, dtype=float),
        }

    checkpoint_path = save_multi_bond_checkpoint(
        selected_bonds=selected_bonds,
        actors=actors,
        critic=critic,
        hidden_dim=hidden_dim,
        sizes=sizes,
        methodology=methodology,
    )

    out_dir = multi_bond_dir(selected_bonds, methodology=methodology)
    metrics = {
        "selected_bonds": [int(b) for b in selected_bonds],
        "methodology": methodology,
        "hidden_dim": int(hidden_dim),
        "nb_steps": int(nb_steps),
        "long_horizon": int(long_horizon),
        "nb_short_rollouts": int(nb_short_rollouts),
        "short_horizon": int(short_horizon),
        "critic_batch_size": int(critic_batch_size),
        "actor_batch_size": int(actor_batch_size),
        "n_epochs_critic": int(n_epochs_critic),
        "n_epochs_actor": int(n_epochs_actor),
        "update_risk_after": None if update_risk_after is None else int(update_risk_after),
        "critic_lr": float(critic_lr),
        "actor_lr": float(actor_lr),
        "dataset_pretrain_samples": int(dataset_pretrain_samples),
        "dataset_pretrain_actor_lr": float(dataset_pretrain_actor_lr),
        "dataset_pretrain_actor_epochs": int(dataset_pretrain_actor_epochs),
        "dataset_pretrain_actor_batch_size": int(dataset_pretrain_actor_batch_size),
        "dataset_pretrain_critic_lr": float(dataset_pretrain_critic_lr),
        "dataset_pretrain_critic_max_steps": int(dataset_pretrain_critic_max_steps),
        "dataset_pretrain_critic_tol": float(dataset_pretrain_critic_tol),
        "lb_init": lb_init.tolist(),
        "ub_init": ub_init.tolist(),
        "lb_final": lb_final.tolist(),
        "ub_final": ub_final.tolist(),
        "checkpoint_path": str(checkpoint_path),
        "history": {
            "avg_reward_long": np.asarray(results["avg_reward_long"], dtype=float).tolist(),
            "critic_loss": np.asarray(results["critic_loss"], dtype=float).tolist(),
        },
        "final_avg_reward_long": float(np.asarray(results["avg_reward_long"], dtype=float)[-1]),
    }

    save_json(out_dir / "metrics.json", metrics)

    return {
        "actors": actors,
        "critic": critic,
        "market": market,
        "sizes": sizes,
        "sigma": sigma,
        "results": results,
        "metrics": metrics,
        "pretrain_payload": pretrain_payload,
        "checkpoint_path": checkpoint_path,
    }