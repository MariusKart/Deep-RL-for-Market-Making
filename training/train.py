
import numpy as np
import torch
from config.constants import *
from core.SimulationEnvironment import *
from core.models import *
from core.rollout import *
import matplotlib.pyplot as plt
from training.critic_update import * 
from training.actor_update import *


def train_final(
    actors,                  # list or dict of per-bond actors (length d)
    critic,                  # torch.nn.Module (scalar output)
    market,
    r,
    Sigma,
    nb_steps,
    avg_sizes,
    long_horizon=1_000,
    nb_short_rollouts=10,
    short_horizon=100,
    critic_batch_size=70,
    actor_batch_size=50,
    n_epochs_critic=1,
    n_epochs_actor=1,
    update_risk_after=500,
    LB_init=None,
    UB_init=None,
    LB_max=None,
    UB_max=None,
    critic_lr=1e-3,
    actor_lr=1e-2,
    seed=None,
    device=None,
):
    """

    """

    rng = np.random.default_rng(seed)

    # allow list or dict actors
    if isinstance(actors, dict):
        d = len(actors)
        policy = [actors[i] for i in range(d)]
    else:
        d = len(actors)
        policy = list(actors)

    critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    actor_opts = {i: torch.optim.Adam(policy[i].parameters(), lr=actor_lr) for i in range(d)}

    # paper-style: no target network (semi-gradient via no_grad inside update_critic_td)
    target_critic = critic

    # ----- risk limits -----
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

    avg_reward_long_hist = []
    critic_loss_hist = []

    for step in range(nb_steps):

        # Reverse Matryoshka schedule
        if step > 0 and update_risk_after is not None and (step % update_risk_after == 0):
            LB = np.maximum(LB - np.asarray(avg_sizes, dtype=float), LB_max)
            UB = np.minimum(UB + np.asarray(avg_sizes, dtype=float), UB_max)

        # ---------- long rollout (single) ----------  
        flat = np.zeros(d, dtype=float)
        
        S_long, A_base_long, A_noisy_long, D_long, R_long, S_trade_long = rollout(
            policy=policy,
            starting_inventory=flat,
            horizon=long_horizon,
            market=market,
            r=r,
            Sigma=Sigma,
            avg_sizes=avg_sizes, 
            lb= LB,
            ub = UB
        )

        R_long = np.asarray(R_long, dtype=float).reshape(-1)
        r_bar = float(R_long.mean()) if R_long.size > 0 else 0.0
        avg_reward_long_hist.append(r_bar)

        # ---------- short rollouts (batched) ----------
        if nb_short_rollouts > 0:
            q0_batch = rng.uniform(LB, UB, size=(nb_short_rollouts, d)).astype(float)
            S_s, A_base_s, A_noisy_s, D_s, R_s, S_trade_s = rollout(
                policy=policy,
                starting_inventory=q0_batch,
                horizon=short_horizon,
                market=market,
                r=r,
                avg_sizes=avg_sizes, 
                Sigma=Sigma,
                lb= LB,
                ub = UB    
            )

        
            S_all       = np.vstack([np.asarray(S_long, dtype=float), np.asarray(S_s, dtype=float)])
            R_all       = np.concatenate([np.asarray(R_long, dtype=float).reshape(-1), np.asarray(R_s, dtype=float).reshape(-1)], axis=0)
            S_trade_all = np.vstack([np.asarray(S_trade_long, dtype=float), np.asarray(S_trade_s, dtype=float)])
            A_base_all  = np.vstack([np.asarray(A_base_long, dtype=float), np.asarray(A_base_s, dtype=float)])
            A_noisy_all = np.vstack([np.asarray(A_noisy_long, dtype=float), np.asarray(A_noisy_s, dtype=float)])
            D_all       = np.concatenate([np.asarray(D_long, dtype=int).reshape(-1), np.asarray(D_s, dtype=int).reshape(-1)], axis=0)
        else:
            S_all       = np.asarray(S_long, dtype=float)
            R_all       = np.asarray(R_long, dtype=float).reshape(-1)
            S_trade_all = np.asarray(S_trade_long, dtype=float)
            A_base_all  = np.asarray(A_base_long, dtype=float)
            A_noisy_all = np.asarray(A_noisy_long, dtype=float)
            D_all       = np.asarray(D_long, dtype=int).reshape(-1)

        bond_ids = A_base_all[:, 0].astype(int)

        # ---------- critic TD update ----------
        critic_out = update_critic_td(
            critic=critic,
            target_critic=target_critic,
            optimizer=critic_opt,
            S=S_all,
            A=A_base_all,          # (N,2) [bond_id, p_base]
            R=R_all,
            S_next=S_trade_all,    # trade-next inventory
            market=market,
            avg_sizes = avg_sizes, 
            batch_size=critic_batch_size,
            n_epochs=n_epochs_critic,
            r_bar=r_bar,
            device=device,
        )
        critic_loss = critic_out[0] if isinstance(critic_out, (tuple, list)) else critic_out
        critic_loss_hist.append(float(critic_loss))

        # ---------- actor updates ----------
        for i in range(d):
            m = (bond_ids == i)
            if not np.any(m):
                continue

            S_i = S_all[m]
            p_base_i = A_base_all[m, 1]    # probability column
            p_noisy_i = A_noisy_all[m, 1]  # probability column
            D_i = D_all[m]

            update_actor_i(
                actor_i=policy[i],
                critic=critic,
                optimizer=actor_opts[i],
                market=market,
                S_i=S_i,
                p_base_i=p_base_i,
                p_noisy_i=p_noisy_i,
                D_i=D_i,
                i=i,
                avg_sizes= avg_sizes,
                Sigma=Sigma,
                batch_size=actor_batch_size,
                n_epochs=n_epochs_actor,
                device=device,
                ub= UB,
                lb = LB
                
            )

    # ---------- plot ----------
    plt.figure()
    plt.plot(avg_reward_long_hist)
    plt.xlabel("Training step")
    plt.ylabel("Average reward per RFQ (long rollout)")
    plt.title("Average reward per RFQ during training")
    plt.grid(True, alpha=0.3)
    plt.show()

    return {
        "avg_reward_long": np.asarray(avg_reward_long_hist, dtype=float),
        "critic_loss": np.asarray(critic_loss_hist, dtype=float),
        "LB_final": LB.copy(),
        "UB_final": UB.copy(),
    }