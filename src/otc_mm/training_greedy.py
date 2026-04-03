from __future__ import annotations

import numpy as np
import torch

from .greedy_actors import (
    build_index_bounds,
    greedy_refresh_actor_1d_from_critic,
    sample_discrete_starts,
)
from .market import rollout
from .training import update_critic_td


def train_final_greedy(
    actors,
    critic,
    market,
    r,
    Sigma,
    nb_steps,
    avg_sizes,
    long_horizon=1000,
    nb_short_rollouts=10,
    short_horizon=100,
    critic_batch_size=70,
    n_epochs_critic=1,
    LB_init=None,
    UB_init=None,
    LB_max=None,
    UB_max=None,
    critic_lr=1e-3,
    update_risk_after=500,
    p_grid=None,
    seed=42,
    device=None,
):
    """
    greedy training loop from the alternative methodology:

    - critic update: same TD critic update as the classic method
    - actor update: greedy refresh from the critic on a discrete p-grid
    - long rollout from zero inventory
    - short rollouts from random discrete inventories inside current bounds
    - reverse Matryoshka risk schedule
    """
    rng = np.random.default_rng(seed)

    if isinstance(actors, dict):
        d = len(actors)
        policy = [actors[i] for i in range(d)]
    else:
        d = len(actors)
        policy = list(actors)

    critic_opt = torch.optim.SGD(critic.parameters(), lr=critic_lr)


    target_critic = critic

    global LB, UB
    if LB_init is not None and UB_init is not None:
        LB = np.asarray(LB_init, dtype=float).copy()
        UB = np.asarray(UB_init, dtype=float).copy()
    else:
        LB = np.asarray(LB, dtype=float).copy()
        UB = np.asarray(UB, dtype=float).copy()

    if LB_max is None:
        LB_max = LB.copy()
    else:
        LB_max = np.asarray(LB_max, dtype=float).copy()

    if UB_max is None:
        UB_max = UB.copy()
    else:
        UB_max = np.asarray(UB_max, dtype=float).copy()

    low_idx, high_idx = build_index_bounds(LB, UB, avg_sizes)

    avg_reward_long_hist = []
    critic_loss_hist = []
    greedy_stats_hist = []

    for step in range(nb_steps):
        if step > 0 and update_risk_after is not None and (step % update_risk_after == 0):
            LB = np.maximum(LB - np.asarray(avg_sizes, dtype=float), LB_max)
            UB = np.minimum(UB + np.asarray(avg_sizes, dtype=float), UB_max)
            low_idx, high_idx = build_index_bounds(LB, UB, avg_sizes)

        # long rollout from zero inventory
        flat = np.zeros(d, dtype=float)

        S_long, A_base_long, D_long, R_long, S_trade_long = rollout(
            policy=policy,
            starting_inventory=flat,
            horizon=long_horizon,
            market=market,
            r=r,
            Sigma=Sigma,
            avg_sizes=avg_sizes,
            lb=LB,
            ub=UB,
        )

        R_long = np.asarray(R_long, dtype=float).reshape(-1)
        r_bar = float(R_long.mean()) if R_long.size > 0 else 0.0
        avg_reward_long_hist.append(r_bar)

        # short rollouts from discrete starts inside the current bounds
        if nb_short_rollouts > 0:
            q0_batch = sample_discrete_starts(
                nb_short_rollouts,
                low_idx,
                high_idx,
                avg_sizes,
                rng,
            )

            S_s, A_base_s, D_s, R_s, S_trade_s = rollout(
                policy=policy,
                starting_inventory=q0_batch,
                horizon=short_horizon,
                market=market,
                r=r,
                Sigma=Sigma,
                avg_sizes=avg_sizes,
                lb=LB,
                ub=UB,
            )

            S_all = np.vstack([
                np.asarray(S_long, dtype=float),
                np.asarray(S_s, dtype=float),
            ])
            R_all = np.concatenate([
                np.asarray(R_long, dtype=float).reshape(-1),
                np.asarray(R_s, dtype=float).reshape(-1),
            ], axis=0)
            S_trade_all = np.vstack([
                np.asarray(S_trade_long, dtype=float),
                np.asarray(S_trade_s, dtype=float),
            ])
            A_base_all = np.vstack([
                np.asarray(A_base_long, dtype=float),
                np.asarray(A_base_s, dtype=float),
            ])
        else:
            S_all = np.asarray(S_long, dtype=float)
            R_all = np.asarray(R_long, dtype=float).reshape(-1)
            S_trade_all = np.asarray(S_trade_long, dtype=float)
            A_base_all = np.asarray(A_base_long, dtype=float)

        critic_loss, r_bar = update_critic_td(
            critic=critic,
            target_critic=target_critic,
            optimizer=critic_opt,
            S=S_all,
            A=A_base_all,
            R=R_all,
            S_next=S_trade_all,
            market=market,
            avg_sizes=avg_sizes,
            batch_size=critic_batch_size,
            n_epochs=n_epochs_critic,
            r_bar=r_bar,
            device=device,
        )
        critic_loss_hist.append(float(critic_loss))

        # greedy refresh of each 1D actor from the critic
        step_stats = {}
        for i in range(d):
            step_stats[i] = greedy_refresh_actor_1d_from_critic(
                actor_i=policy[i],
                critic=critic,
                market=market,
                i=i,
                Sigma=Sigma,
                avg_sizes=avg_sizes,
                r=r,
                p_grid=p_grid,
                device=device,
                lb=LB,
                ub=UB,
            )
        greedy_stats_hist.append(step_stats)

    return {
        "avg_reward_long": np.asarray(avg_reward_long_hist, dtype=float),
        "critic_loss": np.asarray(critic_loss_hist, dtype=float),
        "greedy_stats": greedy_stats_hist,
        "LB_final": LB.copy(),
        "UB_final": UB.copy(),
        "actors": policy,
        "critic": critic,
    }