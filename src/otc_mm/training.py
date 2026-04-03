from __future__ import annotations

import numpy as np
import torch

from .constants import DISCOUNT_RL, GAMMA, NU, RF
from .market import rollout
from .models import Critic


def update_actor_i(
    actor_i,
    critic,
    optimizer,
    market,
    S_i,
    D_i,
    i,
    Sigma,
    lb,
    ub,
    avg_sizes,
    batch_size=2048,
    n_epochs=1,
    eps_half_width=0.05,
    max_grad_norm=1.0,
    device=None,
):
    """
    - one symmetric actor per bond
    - bid uses actor(q), ask uses actor(-q)
    - perturbation-based update 
    """
    if device is None:
        try:
            device = next(actor_i.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    actor_i = actor_i.to(device)
    critic = critic.to(device)

    S_i = np.asarray(S_i, dtype=np.float32)
    D_i = np.asarray(D_i, dtype=np.int64).reshape(-1)

    n = S_i.shape[0]
    if n == 0:
        return {"loss": 0.0, "n": 0}

    avg_sizes_vec = np.asarray(avg_sizes, dtype=np.float32).reshape(-1)
    size_i = float(avg_sizes_vec[i])

    sign_np = np.where(D_i == 1, +1.0, -1.0).astype(np.float32)

    # Raw post-trade state
    S_trade_raw = S_i.copy()
    S_trade_raw[:, i] += sign_np * size_i

    # Feasibility mask
    valid_np = (
        (S_trade_raw[:, i] >= float(lb[i]) - 1e-6) &
        (S_trade_raw[:, i] <= float(ub[i]) + 1e-6)
    ).astype(np.float32)

    # Clipped version kept for numerical safety in critic / penalty eval
    S_trade_i = S_trade_raw.copy()
    S_trade_i[:, i] = np.clip(S_trade_i[:, i], float(lb[i]), float(ub[i]))

    lambdas = np.asarray(market.lambda_RFQs, dtype=np.float32).reshape(-1)
    Lambda = 2.0 * float(lambdas.sum())
    denom = float(RF) + Lambda
    gamma_rl = Lambda / denom
    grad_scaler = 1.0 / (((2.0 * eps_half_width) ** 2) / 12.0)

    S_t = torch.as_tensor(S_i, dtype=torch.float32, device=device)
    S_trade_t = torch.as_tensor(S_trade_i, dtype=torch.float32, device=device)
    sign_t = torch.as_tensor(sign_np, dtype=torch.float32, device=device)
    valid_t = torch.as_tensor(valid_np, dtype=torch.float32, device=device)

    avg_sizes_t = torch.as_tensor(avg_sizes_vec, dtype=torch.float32, device=device)
    Sigma_t = torch.as_tensor(np.asarray(Sigma, dtype=np.float32), dtype=torch.float32, device=device)

    denom_t = torch.tensor(denom, dtype=torch.float32, device=device)
    gamma_t = torch.tensor(gamma_rl, dtype=torch.float32, device=device)

    with torch.inference_mode():
        critic.eval()

        S_norm_t = S_t / avg_sizes_t
        S_trade_norm_t = S_trade_t / avg_sizes_t

        V_q = critic(S_norm_t)
        V_trade = critic(S_trade_norm_t)

        if V_q.dim() == 2:
            V_q = V_q[:, 0]
        if V_trade.dim() == 2:
            V_trade = V_trade[:, 0]

        psi_q = 0.5 * float(GAMMA) * torch.sqrt(torch.einsum("bi,ij,bj->b", S_t, Sigma_t, S_t))
        psi_trade = 0.5 * float(GAMMA) * torch.sqrt(torch.einsum("bi,ij,bj->b", S_trade_t, Sigma_t, S_trade_t))

        X_glob = (S_t * sign_t.unsqueeze(1)) / avg_sizes_t
        p_curr_glob = actor_i(X_glob)
        if p_curr_glob.dim() == 2:
            p_curr_glob = p_curr_glob[:, 0]

        eps_glob = (torch.rand(n, device=device) * (2.0 * eps_half_width)) - eps_half_width
        p_eps_glob = torch.clamp(p_curr_glob + eps_glob, NU, 1.0 - NU)

        delta_curr_glob = torch.as_tensor(
            np.asarray(market.inv_f(i, p_curr_glob.detach().cpu().numpy()), dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        delta_eps_glob = torch.as_tensor(
            np.asarray(market.inv_f(i, p_eps_glob.detach().cpu().numpy()), dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )

        no_trade_val = -psi_q / denom_t + gamma_t * V_q
        trade_curr_val = size_i * delta_curr_glob - psi_trade / denom_t + gamma_t * V_trade
        trade_eps_val = size_i * delta_eps_glob - psi_trade / denom_t + gamma_t * V_trade

        p_eff_curr_glob = p_curr_glob * valid_t
        p_eff_eps_glob = p_eps_glob * valid_t

        v_curr_glob = p_eff_curr_glob * trade_curr_val + (1.0 - p_eff_curr_glob) * no_trade_val
        v_eps_glob = p_eff_eps_glob * trade_eps_val + (1.0 - p_eff_eps_glob) * no_trade_val

        global_adv_std = (v_eps_glob - v_curr_glob).std(unbiased=False).clamp_min(1e-8)

    actor_i.train()

    total_loss = 0.0
    nb = 0

    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]

            X_b = (S_t[idx] * sign_t[idx].unsqueeze(1)) / avg_sizes_t
            valid_b = valid_t[idx]

            with torch.no_grad():
                p_curr = actor_i(X_b)
                if p_curr.dim() == 2:
                    p_curr = p_curr[:, 0]

                eps = (torch.rand(idx.shape[0], device=device) * (2.0 * eps_half_width)) - eps_half_width
                p_eps = torch.clamp(p_curr + eps, NU, 1.0 - NU)

                delta_curr = torch.as_tensor(
                    np.asarray(market.inv_f(i, p_curr.detach().cpu().numpy()), dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                )
                delta_eps = torch.as_tensor(
                    np.asarray(market.inv_f(i, p_eps.detach().cpu().numpy()), dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                )

                no_trade_val_b = -psi_q[idx] / denom_t + gamma_t * V_q[idx]
                trade_curr_val_b = size_i * delta_curr - psi_trade[idx] / denom_t + gamma_t * V_trade[idx]
                trade_eps_val_b = size_i * delta_eps - psi_trade[idx] / denom_t + gamma_t * V_trade[idx]

                p_eff_curr = p_curr * valid_b
                p_eff_eps = p_eps * valid_b

                v_curr = p_eff_curr * trade_curr_val_b + (1.0 - p_eff_curr) * no_trade_val_b
                v_eps = p_eff_eps * trade_eps_val_b + (1.0 - p_eff_eps) * no_trade_val_b

                dv = (v_eps - v_curr) / global_adv_std
                dp = p_eps - p_curr

            p_for_grad = actor_i(X_b)
            if p_for_grad.dim() == 2:
                p_for_grad = p_for_grad[:, 0]

            loss = -(p_for_grad * dv * dp).mean() * grad_scaler

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_i.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            total_loss += float(loss.detach().item())
            nb += 1

    return {
        "loss": total_loss / max(nb, 1),
        "n": int(n),
        "global_adv_std": float(global_adv_std.detach().item()),
        "valid_rate": float(valid_np.mean()),
    }


def update_critic_td(
    critic: Critic,
    target_critic: Critic,
    optimizer,
    S,
    A,
    R,
    S_next,
    market,
    avg_sizes,
    batch_size=50,
    n_epochs=1,
    r_bar=None,
    device="cpu",
):
    """
    critic TD update.
    """
    critic = critic.to(device)
    target_critic = target_critic.to(device)

    S_np = np.asarray(S, dtype=np.float32)
    Sp_np = np.asarray(S_next, dtype=np.float32)
    A_np = np.asarray(A, dtype=np.float32)
    R_np = np.asarray(R, dtype=np.float32).reshape(-1)

    N = S_np.shape[0]

    if r_bar is None:
        r_bar = float(R_np.mean())

    R_center = (R_np - float(r_bar)).astype(np.float32)

    bond_ids = A_np[:, 0].astype(np.int64)
    a_raw = A_np[:, 1].astype(np.float32)
    a_clip = np.clip(a_raw, NU, 1.0 - NU)

    size_vec = np.asarray(avg_sizes, dtype=np.float32)

    deltas = np.empty(N, dtype=np.float32)
    for k in range(N):
        i = int(bond_ids[k])
        deltas[k] = float(market.inv_f(i, float(a_clip[k])))

    p = np.empty(N, dtype=np.float32)
    for i in np.unique(bond_ids):
        mask = (bond_ids == i)
        p[mask] = np.asarray(market.f(int(i), deltas[mask]), dtype=np.float32)

    p = np.clip(p, NU, 1.0 - NU)

    denom_sizes = size_vec.reshape(1, -1)

    S_t = torch.as_tensor(S_np / denom_sizes, dtype=torch.float32, device=device)
    Sp_t = torch.as_tensor(Sp_np / denom_sizes, dtype=torch.float32, device=device)
    r_t = torch.as_tensor(R_center, dtype=torch.float32, device=device).unsqueeze(1)
    p_t = torch.as_tensor(p, dtype=torch.float32, device=device).unsqueeze(1)

    critic.train()
    target_critic.eval()

    total_loss, n_batches = 0.0, 0

    for _ in range(n_epochs):
        perm = torch.randperm(N, device=device)

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]

            s = S_t[idx]
            sp = Sp_t[idx]
            r = r_t[idx]
            pb = p_t[idx]

            with torch.no_grad():
                V_trade = target_critic(sp)
                V_stay = target_critic(s)

                if V_trade.dim() == 1:
                    V_trade = V_trade.unsqueeze(1)
                if V_stay.dim() == 1:
                    V_stay = V_stay.unsqueeze(1)

                y = r + DISCOUNT_RL * (pb * V_trade + (1.0 - pb) * V_stay)

            v = critic(s)
            if v.dim() == 1:
                v = v.unsqueeze(1)

            loss = torch.mean((v - y) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.detach().item()
            n_batches += 1

    return total_loss / max(n_batches, 1), float(r_bar)


def train_final(
    actors,
    critic,
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
    seed=42,
    device=None,
):
    """
    - reverse Matryoshka risk schedule
    - long rollout from zero inventory
    - short rollouts from random inventories inside current bounds
    - critic update on long + short
    - actor updates on long + short, split by bond
    """
    rng = np.random.default_rng(seed)

    policy = list(actors)
    d = len(policy)

    critic_opt = torch.optim.SGD(critic.parameters(), lr=critic_lr)
    actor_opts = {i: torch.optim.SGD(policy[i].parameters(), lr=actor_lr) for i in range(d)}


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

    avg_reward_long_hist = []
    critic_loss_hist = []

    for step in range(nb_steps):
        # reverse Matryoshka schedule
        if step > 0 and update_risk_after is not None and (step % update_risk_after == 0):
            LB = np.maximum(LB - np.asarray(avg_sizes, dtype=float), LB_max)
            UB = np.minimum(UB + np.asarray(avg_sizes, dtype=float), UB_max)

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

        # short rollouts from random inventories inside current box
        if nb_short_rollouts > 0:
            low_steps = np.ceil(LB / avg_sizes).astype(int)
            high_steps = np.floor(UB / avg_sizes).astype(int)
            q0_batch = rng.integers(low_steps, high_steps + 1, size=(nb_short_rollouts, d)) * avg_sizes

            short_seed = int(rng.integers(0, 2**32 - 1))
            rng = np.random.default_rng(short_seed)

            S_s, A_base_s, D_s, R_s, S_trade_s = rollout(
                policy=policy,
                starting_inventory=q0_batch,
                horizon=short_horizon,
                market=market,
                r=r,
                avg_sizes=avg_sizes,
                Sigma=Sigma,
                lb=LB,
                ub=UB,
            )

            S_all = np.vstack([np.asarray(S_long, dtype=float), np.asarray(S_s, dtype=float)])
            R_all = np.concatenate(
                [np.asarray(R_long, dtype=float).reshape(-1), np.asarray(R_s, dtype=float).reshape(-1)],
                axis=0,
            )
            S_trade_all = np.vstack([np.asarray(S_trade_long, dtype=float), np.asarray(S_trade_s, dtype=float)])
            A_base_all = np.vstack([np.asarray(A_base_long, dtype=float), np.asarray(A_base_s, dtype=float)])
            D_all = np.concatenate(
                [np.asarray(D_long, dtype=int).reshape(-1), np.asarray(D_s, dtype=int).reshape(-1)],
                axis=0,
            )
        else:
            S_all = np.asarray(S_long, dtype=float)
            R_all = np.asarray(R_long, dtype=float).reshape(-1)
            S_trade_all = np.asarray(S_trade_long, dtype=float)
            A_base_all = np.asarray(A_base_long, dtype=float)
            D_all = np.asarray(D_long, dtype=int).reshape(-1)

        bond_ids = A_base_all[:, 0].astype(int)

        critic_out = update_critic_td(
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
        critic_loss = critic_out[0] if isinstance(critic_out, (tuple, list)) else critic_out
        critic_loss_hist.append(float(critic_loss))

        for i in range(d):
            m = (bond_ids == i)
            if not np.any(m):
                continue

            S_i = S_all[m]
            D_i = D_all[m]

            update_actor_i(
                actor_i=policy[i],
                critic=critic,
                optimizer=actor_opts[i],
                market=market,
                S_i=S_i,
                D_i=D_i,
                i=i,
                avg_sizes=avg_sizes,
                Sigma=Sigma,
                batch_size=actor_batch_size,
                n_epochs=n_epochs_actor,
                device=device,
                ub=UB,
                lb=LB,
            )

    return {
        "avg_reward_long": np.asarray(avg_reward_long_hist, dtype=float),
        "critic_loss": np.asarray(critic_loss_hist, dtype=float),
    }