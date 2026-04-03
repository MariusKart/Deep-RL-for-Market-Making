from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d

from .constants import GAMMA, NU, RF
from .market import Market
from .utils import load_single_bond_targets


def pretrain_actor(
    actor,
    target_p_i,
    LB_risk,
    UB_risk,
    avg_sizes,
    batch_size=50,
    epochs=1000,
    lr=1e-3,
    device="cpu",
):
    """
    1-bond actor pretraining:
    pretrain one actor on the constant myopic probability p_i.
    """
    d = np.asarray(LB_risk).shape[0]

    LB = torch.tensor(LB_risk, dtype=torch.float32, device=device).reshape(1, d)
    UB = torch.tensor(UB_risk, dtype=torch.float32, device=device).reshape(1, d)
    avg = torch.tensor(avg_sizes, dtype=torch.float32, device=device).reshape(1, d)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(actor.parameters(), lr=lr)

    actor = actor.to(device)
    actor.train()

    for _ in range(epochs):
        optimizer.zero_grad()

        q_rand = LB + (UB - LB) * torch.rand(batch_size, d, device=device)
        q_rand = q_rand / avg

        pred = actor(q_rand)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)

        target = torch.full(
            (batch_size, 1),
            float(target_p_i),
            dtype=torch.float32,
            device=device,
        )

        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

    return actor


def solve_1d_value_fixed_policy(
    market: Market,
    i,
    LB_i,
    UB_i,
    rfq_size,
    delta_bid,
    delta_ask,
):
    """
    1D finite-difference value solver for a fixed policy.
    """
    grid = np.arange(LB_i, UB_i, rfq_size, dtype=float)
    n = grid.size

    p_bid = float(market.f(i, delta_bid))
    p_ask = float(market.f(i, delta_ask))

    gamma_rl = (2.0 * market.lambda_RFQs[i]) / (RF + 2.0 * market.lambda_RFQs[i])

    def psi(q):
        return 0.5 * GAMMA * np.sqrt(market.Sigma[i][i] * q * q)

    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    for k, q in enumerate(grid):
        can_up = (k + 1 < n)
        can_dn = (k - 1 >= 0)

        coeff_kk = 1.0
        rhs = 0.0

        # Bid side: prob 1/2
        if can_up:
            coeff_kk -= 0.5 * (1.0 - p_bid) * gamma_rl
            A[k, k + 1] += -0.5 * p_bid * gamma_rl

            rhs += 0.5 * p_bid * (rfq_size * delta_bid)
            rhs += 0.5 * p_bid * (-psi(q + rfq_size) / (RF + 2.0 * market.lambda_RFQs[i]))
            rhs += 0.5 * (1.0 - p_bid) * (-psi(q) / (RF + 2.0 * market.lambda_RFQs[i]))
        else:
            coeff_kk -= 0.5 * gamma_rl
            rhs += 0.5 * (-psi(q) / (RF + 2.0 * market.lambda_RFQs[i]))

        # Ask side: prob 1/2
        if can_dn:
            coeff_kk -= 0.5 * (1.0 - p_ask) * gamma_rl
            A[k, k - 1] += -0.5 * p_ask * gamma_rl

            rhs += 0.5 * p_ask * (rfq_size * delta_ask)
            rhs += 0.5 * p_ask * (-psi(q - rfq_size) / (RF + 2.0 * market.lambda_RFQs[i]))
            rhs += 0.5 * (1.0 - p_ask) * (-psi(q) / (RF + 2.0 * market.lambda_RFQs[i]))
        else:
            coeff_kk -= 0.5 * gamma_rl
            rhs += 0.5 * (-psi(q) / (RF + 2.0 * market.lambda_RFQs[i]))

        A[k, k] = coeff_kk
        b[k] = rhs

    V = np.linalg.solve(A, b)
    V = V - V[n // 2]
    return grid, V


def produce_initial_value_grid(
    market: Market,
    init_strategy,
    lb_risk,
    ub_risk,
    nb_bonds,
    sizes,
):
    """
    build one 1D grid/value function per bond.
    """
    grids = []
    Vs = []

    for i in range(nb_bonds):
        grid_i, V_i = solve_1d_value_fixed_policy(
            market=market,
            i=i,
            LB_i=float(lb_risk[i]),
            UB_i=float(ub_risk[i]),
            rfq_size=sizes[i],
            delta_bid=float(init_strategy[i]),
            delta_ask=float(init_strategy[i]),
        )
        grids.append(grid_i)
        Vs.append(V_i)

    return grids, Vs


def pretrain_critic(
    critic,
    grids,
    Vs,
    sizes,
    device="cpu",
    n_dense=100,
    lr=1e-2,
    max_steps=3000,
    tol=1e-4,
):
    """
    critic pretraining:
    - interpolate each 1D value function
    - Cartesian grid in original q-space
    - target = sum of 1D values
    - input to critic = q / sizes
    """
    nb_bonds = len(grids)

    interpolants = []
    dense_axes = []

    for i in range(nb_bonds):
        grid_i = np.asarray(grids[i], dtype=np.float32).reshape(-1)
        V_i = np.asarray(Vs[i], dtype=np.float32).reshape(-1)

        kind = "cubic" if grid_i.shape[0] >= 4 else "linear"
        f_i = interp1d(
            grid_i,
            V_i,
            kind=kind,
            bounds_error=False,
            fill_value=(V_i[0], V_i[-1]),
        )
        interpolants.append(f_i)

        dense_axis_i = np.linspace(grid_i.min(), grid_i.max(), n_dense, dtype=np.float32)
        dense_axes.append(dense_axis_i)

    mesh = np.meshgrid(*dense_axes, indexing="ij")
    X_orig = np.stack([m.ravel() for m in mesh], axis=1).astype(np.float32)

    Y_dense = np.zeros((X_orig.shape[0], 1), dtype=np.float32)
    for i in range(nb_bonds):
        Y_dense[:, 0] += interpolants[i](X_orig[:, i]).astype(np.float32)

    X_scaled = X_orig / np.asarray(sizes, dtype=np.float32).reshape(1, -1)

    X = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    Y = torch.tensor(Y_dense, dtype=torch.float32, device=device)

    critic = critic.to(device).float()
    critic.train()

    opt = optim.Adam(critic.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for step in range(max_steps):
        opt.zero_grad()
        pred = critic(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()

        if loss.item() < tol:
            print(f"stopped early at step {step}, loss={loss.item():.6f}")
            break

    return critic


def _sample_q_multi(LB_risk, UB_risk, n_samples, seed=42):
    rng = np.random.default_rng(seed)
    LB = np.asarray(LB_risk, dtype=np.float32).reshape(1, -1)
    UB = np.asarray(UB_risk, dtype=np.float32).reshape(1, -1)
    U = rng.uniform(size=(n_samples, LB.shape[1])).astype(np.float32)
    return LB + (UB - LB) * U


def _interp_saved_curve(x_grid, y_grid, x_new):
    x_grid = np.asarray(x_grid, dtype=np.float32).reshape(-1)
    y_grid = np.asarray(y_grid, dtype=np.float32).reshape(-1)
    x_new = np.asarray(x_new, dtype=np.float32).reshape(-1)
    return np.interp(x_new, x_grid, y_grid).astype(np.float32)


def build_multi_bond_critic_targets_from_saved_datasets(selected_bonds, q_multi, methodology="classic"):
    """
    Critical multi-bond warm start:
        V_target(q) = sum_{bond in selected_bonds} V_bond(q_bond)
    where each V_bond comes from the stored learned 1D dataset.
    """
    q_multi = np.asarray(q_multi, dtype=np.float32)
    n, d = q_multi.shape

    if len(selected_bonds) != d:
        raise ValueError("selected_bonds length must match q_multi dimension")

    y = np.zeros((n, 1), dtype=np.float32)

    for local_j, global_bond in enumerate(selected_bonds):
        tgt = load_single_bond_targets(int(global_bond), methodology=methodology)
        y[:, 0] += _interp_saved_curve(
            tgt["inventories"],
            tgt["values"],
            q_multi[:, local_j],
        )

    return y


def pretrain_critic_from_saved_1d_datasets(
    critic,
    selected_bonds,
    avg_sizes,
    LB_risk,
    UB_risk,
    methodology="classic",
    n_samples=20000,
    device="cpu",
    lr=1e-3,
    max_steps=3000,
    tol=1e-4,
    seed=42,
):
    """
    Multi-bond critic warm-start from stored learned 1D value functions.

    Input to critic: q / avg_sizes
    Target:
        sum_j V_j(q_j)
    """
    q_multi = _sample_q_multi(
        LB_risk=LB_risk,
        UB_risk=UB_risk,
        n_samples=n_samples,
        seed=seed,
    )

    y = build_multi_bond_critic_targets_from_saved_datasets(
        selected_bonds=selected_bonds,
        q_multi=q_multi,
        methodology=methodology,
    )

    x = q_multi / np.asarray(avg_sizes, dtype=np.float32).reshape(1, -1)

    X = torch.tensor(x, dtype=torch.float32, device=device)
    Y = torch.tensor(y, dtype=torch.float32, device=device)

    critic = critic.to(device).float()
    critic.train()

    opt = optim.Adam(critic.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for step in range(max_steps):
        opt.zero_grad()
        pred = critic(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()

        if loss.item() < tol:
            print(f"critic dataset pretraining stopped early at step {step}, loss={loss.item():.6f}")
            break

    return critic, q_multi, y


def pretrain_actor_from_saved_1d_dataset(
    actor,
    local_bond_idx,
    global_bond_idx,
    q_multi,
    avg_sizes,
    market: Market,
    methodology="classic",
    device="cpu",
    lr=1e-3,
    epochs=1000,
    batch_size=256,
):
    """
    Multi-bond actor warm-start from stored learned 1D quote curves.

    One actor per bond, symmetry preserved:
    - bid target:  actor( q / s )   should match stored bid quote curve
    - ask target:  actor(-q / s )   should match stored ask quote curve

    We train on probabilities, so stored deltas are converted back through market.f.
    """
    tgt = load_single_bond_targets(int(global_bond_idx), methodology=methodology)

    q_multi = np.asarray(q_multi, dtype=np.float32)
    x = q_multi / np.asarray(avg_sizes, dtype=np.float32).reshape(1, -1)

    q_j = q_multi[:, local_bond_idx]

    delta_bid = _interp_saved_curve(
        tgt["inventories"],
        tgt["deltas_bid"],
        q_j,
    )
    delta_ask = _interp_saved_curve(
        tgt["inventories"],
        tgt["deltas_ask"],
        q_j,
    )

    p_bid = np.asarray(market.f(local_bond_idx, delta_bid), dtype=np.float32).reshape(-1, 1)
    p_ask = np.asarray(market.f(local_bond_idx, delta_ask), dtype=np.float32).reshape(-1, 1)

    p_bid = np.clip(p_bid, NU, 1.0 - NU)
    p_ask = np.clip(p_ask, NU, 1.0 - NU)

    X_bid = x.astype(np.float32)
    X_ask = (-x).astype(np.float32)

    X_all = np.concatenate([X_bid, X_ask], axis=0)
    Y_all = np.concatenate([p_bid, p_ask], axis=0)

    X_t = torch.tensor(X_all, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_all, dtype=torch.float32, device=device)

    actor = actor.to(device).float()
    actor.train()

    opt = optim.Adam(actor.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = X_all.shape[0]

    for _ in range(epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]

            xb = X_t[idx]
            yb = Y_t[idx]

            opt.zero_grad()
            pred = actor(xb)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)

            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    return actor


def pretrain_multi_bond_from_single_bond_targets(
    actors,
    critic,
    selected_bonds,
    avg_sizes,
    LB_risk,
    UB_risk,
    market: Market,
    methodology="classic",
    n_samples=20000,
    critic_lr=1e-3,
    critic_max_steps=3000,
    critic_tol=1e-4,
    actor_lr=1e-3,
    actor_epochs=1000,
    actor_batch_size=256,
    device="cpu",
    seed=42,
):
    """
    Correct multi-bond warm-start:

    - critic target is additive:
          Critic_target(q) = sum_j Critic_j(q_j)
      using the stored learned 1D value datasets

    - actor j is trained from the stored learned 1D quote curves of bond j

    - symmetry is preserved because ask uses actor(-q)
    """
    critic, q_multi, y_critic = pretrain_critic_from_saved_1d_datasets(
        critic=critic,
        selected_bonds=selected_bonds,
        avg_sizes=avg_sizes,
        LB_risk=LB_risk,
        UB_risk=UB_risk,
        methodology=methodology,
        n_samples=n_samples,
        device=device,
        lr=critic_lr,
        max_steps=critic_max_steps,
        tol=critic_tol,
        seed=seed,
    )

    pretrained_actors = []
    for local_j, global_bond in enumerate(selected_bonds):
        actor_j = pretrain_actor_from_saved_1d_dataset(
            actor=actors[local_j],
            local_bond_idx=local_j,
            global_bond_idx=global_bond,
            q_multi=q_multi,
            avg_sizes=avg_sizes,
            market=market,
            methodology=methodology,
            device=device,
            lr=actor_lr,
            epochs=actor_epochs,
            batch_size=actor_batch_size,
        )
        pretrained_actors.append(actor_j)

    payload = {
        "q_multi": q_multi,
        "critic_targets": y_critic,
    }

    return pretrained_actors, critic, payload